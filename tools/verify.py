"""Check the C engine against the NumPy reference.

Runs the same token sequence through both and reports how far apart the logits
are. The C side is exercised through `needle_host --dump-logits`, so this tests
the exact binary that gets cross-compiled for the board.

    make -C host
    python3 tools/verify.py build/needle.nsp model/needle2.cact
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cact import parse_tokenizer_blob                     # noqa: E402
from chat import render_prompt                            # noqa: E402
from ref_needle import Weights, Needle                    # noqa: E402
from tok import Tokenizer                                 # noqa: E402


def q8(v):
    """Per-vector absmax int8, matching quant_i8() in needle.c."""
    m = np.abs(v).max()
    if m <= 0:
        return np.zeros_like(v)
    s = m / 127.0
    return np.clip(np.rint(v / s), -127, 127) * s


def patch_int8_kv(mod):
    """Make the reference use an int8 KV cache, as the runtime does."""
    import math

    def attention(self, layer, h):
        w = self.w
        hd, H, KV = w.head_dim, w.n_heads, w.n_kv
        p = f"layer{layer:02d}."
        q = (w.get(p + "q_proj") @ h).reshape(H, hd)
        k = (w.get(p + "k_proj") @ h).reshape(KV, hd)
        v = (w.get(p + "v_proj") @ h).reshape(KV, hd)
        q = mod.zcrms(q, w.get(p + "q_norm"))
        k = mod.zcrms(k, w.get(p + "k_norm"))
        q, k = self._rope(q), self._rope(k)
        k = np.stack([q8(k[g]) for g in range(KV)])
        v = np.stack([q8(v[g]) for g in range(KV)])
        q = np.stack([q8(q[i]) for i in range(H)])
        t = self.pos
        self.k_cache[layer, :, t] = k
        self.v_cache[layer, :, t] = v
        lo = 0 if not w.kv_window else max(0, t - w.kv_window + 1)
        keep = np.arange(lo, t + 1)
        K, V = self.k_cache[layer][:, keep], self.v_cache[layer][:, keep]
        reps = H // KV
        out = np.empty((H, hd), np.float32)
        for g in range(KV):
            sc = (K[g] @ q[g * reps:(g + 1) * reps].T).T / math.sqrt(hd)
            sc -= sc.max(-1, keepdims=True)
            a = np.exp(sc)
            a /= a.sum(-1, keepdims=True)
            out[g * reps:(g + 1) * reps] = a @ V[g]
        out = out.reshape(-1) * mod.sigmoid(w.get(p + "gate_proj") @ h)
        return w.get(p + "out_proj") @ out

    mod.Needle._attention = attention


def compare(tag, c, py):
    d = np.abs(c - py)
    rel = d.max(1) / (np.abs(py).max(1) + 1e-9)
    cos = (c * py).sum(1) / (np.linalg.norm(c, axis=1) * np.linalg.norm(py, axis=1))
    agree = 100.0 * (c.argmax(1) == py.argmax(1)).mean()
    print(f"{tag:<34} max|d|={d.max():7.4f}  median rel={np.median(rel)*100:6.4f}%  "
          f"min cos={cos.min():.8f}  argmax={agree:.1f}%")


def main():
    nsp = sys.argv[1] if len(sys.argv) > 1 else "build/needle.nsp"
    cact = sys.argv[2] if len(sys.argv) > 2 else "model/needle2.cact"
    host = os.path.join(os.path.dirname(nsp) or ".", "..", "host", "needle_host")
    host = os.path.normpath(host)
    if not os.path.exists(host):
        sys.exit(f"{host} not built; run: make -C host")

    import ref_needle

    w = Weights(cact)
    tk = Tokenizer(parse_tokenizer_blob(w.tokenizer_blob()))
    with open("tools.json") as f:
        tools = json.load(f)
    prompt = render_prompt("dim the living room to 30", tools)
    ids = [tk.bos_id] + tk.encode(prompt)
    print(f"{len(ids)} tokens")

    with tempfile.TemporaryDirectory() as tmp:
        idfile = os.path.join(tmp, "ids.txt")
        outfile = os.path.join(tmp, "logits.f32")
        with open(idfile, "w") as f:
            f.write(" ".join(map(str, ids)))
        subprocess.run([host, nsp, "--dump-logits", idfile, outfile],
                       check=True, capture_output=True)
        c = np.fromfile(outfile, np.float32).reshape(len(ids), -1)

    py = np.stack([Needle(w).step(t) for t in [ids[0]]])  # warm the caches
    m = Needle(w)
    py = np.stack([m.step(t) for t in ids]).astype(np.float32)
    compare("C vs fp32 reference", c, py)

    patch_int8_kv(ref_needle)
    m = ref_needle.Needle(w)
    py8 = np.stack([m.step(t) for t in ids]).astype(np.float32)
    compare("C vs reference with int8 KV", c, py8)


if __name__ == "__main__":
    main()
