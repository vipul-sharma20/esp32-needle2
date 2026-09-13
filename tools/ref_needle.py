"""NumPy reference for Needle-2 incremental decoding.

An independent implementation of the model in NumPy, so the C runtime has
something exact to be checked against. Weights come straight out of the
published `.cact`, so this reproduces the deployed quantised model.

One decode step, for token t:

    x[k] = emb[tok] * sqrt(d)                      for each of 4 lanes k
    for each layer l:
        nx    = rms_unit(flatten(x))                             R^2048
        hpre  = sigmoid(a_pre*nx@phi_pre + b_pre + pre_off)      R^4
        u     = sum_k hpre[k] * x[k]                             R^512
        b     = u + (engram fusion at layers 2 and 15)
        y     = block(b) - u
        hpost = 2*sigmoid(a_post*nx@phi_post + b_post + post_off)
        P     = sinkhorn(a_res*nx@phi_res + b_res)               4x4 doubly stochastic
        x[i]  = sum_j P[i,j] x[j] + hpost[i] * y
    logits = zcrms(mean_k x[k], final_norm) @ emb.T
"""

import math

import numpy as np

from cact import Cact, walsh, tensor_names

ENGRAM_SEED = 0x9E3779B9
ENGRAM_PRIME = 0x01000193
CONV_TAPS = 4


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def silu(x):
    return x * sigmoid(x)


def rms_unit(x, eps=1e-6):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)


def zcrms(x, scale, eps=1e-6):
    return (1.0 + scale) * x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)


def sinkhorn(logits, iters=20):
    lk = logits.astype(np.float32)
    for _ in range(iters):
        lk = lk - _logsumexp(lk, -1)
        lk = lk - _logsumexp(lk, -2)
    return np.exp(lk).astype(np.float32)


def _logsumexp(a, axis):
    m = np.max(a, axis=axis, keepdims=True)
    return m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))


def engram_indices(tokens, t, orders, heads, slots):
    """Hash the n-grams ending at position t. tokens[i] for i<0 reads as 0."""
    out = []
    for oi, order in enumerate(orders):
        for h in range(heads):
            seed = (ENGRAM_SEED * (oi * heads + h + 1)) & 0xFFFFFFFF
            acc = seed
            for j in range(order):
                u = tokens[t - j] if t - j >= 0 else 0
                acc = ((acc ^ u) * ENGRAM_PRIME) & 0xFFFFFFFF
            acc ^= acc >> 15
            out.append(acc % slots)
    return out


class Weights:
    """Dequantised tensors, addressed by canonical name."""

    def __init__(self, path, dequant_all=True):
        self.c = Cact(path)
        h = self.c.hdr
        self.names = tensor_names(h, self.c.sites)
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.d_model = h["d_model"]
        self.n_layers = h["num_layers"]
        self.n_heads = h["num_heads"]
        self.n_kv = h["num_kv_heads"]
        self.head_dim = h["head_dim"]
        self.vocab = h["vocab"]
        self.lanes = h["mhc_lanes"]
        self.kv_window = h["kv_window"]
        self.max_seq = h["max_seq_len"]
        self.slots = h["engram_slots"]
        self.sub_dim = h["engram_sub_dim"]
        self.n_tables = h["num_engram_tables"]
        self.dilation = h["engram_conv_dilation"]
        self.orders = self.c.orders
        self.sites = self.c.sites
        self.heads_per_order = self.n_tables // len(self.orders)
        self.rope_theta = h["rope_theta"]
        self._cache = {}
        if dequant_all:
            for n in self.names:
                if n != "tokenizer":
                    self.get(n)

    def get(self, name):
        if name not in self._cache:
            self._cache[name] = self.c.dequant(self.idx[name])
        return self._cache[name]

    def tokenizer_blob(self):
        return self.c.dequant(self.idx["tokenizer"])


class Needle:
    def __init__(self, w: Weights, max_len=None):
        self.w = w
        self.d = w.d_model
        self.max_len = max_len or w.max_seq
        hd = w.head_dim
        inv = 1.0 / (w.rope_theta ** (np.arange(0, hd, 2, dtype=np.float32) / hd))
        ang = np.outer(np.arange(self.max_len, dtype=np.float32), inv)
        self.cos = np.cos(ang).astype(np.float32)
        self.sin = np.sin(ang).astype(np.float32)
        self.H_hada = walsh(w.d_model)
        self.lane_onehot = np.eye(w.lanes, dtype=np.float32)
        self.reset()

    # -- state ------------------------------------------------------------
    def reset(self):
        w = self.w
        self.pos = 0
        self.tokens = []
        self.k_cache = np.zeros((w.n_layers, w.n_kv, self.max_len, w.head_dim), np.float32)
        self.v_cache = np.zeros_like(self.k_cache)
        # engram conv needs v at offsets 0,3,6,9 -> keep the last 10 rows
        span = (CONV_TAPS - 1) * w.dilation + 1
        self.ev_hist = np.zeros((len(w.sites), span, w.d_model), np.float32)
        self.sink_len = 0

    def pin_sinks(self):
        """Freeze everything decoded so far as always-attended KV sinks."""
        self.sink_len = self.pos

    # -- pieces -----------------------------------------------------------
    def _engram_kv(self, site):
        """(k, v) for the current position at one engram site."""
        w = self.w
        t = self.pos
        idx = engram_indices(self.tokens, t, w.orders, w.heads_per_order, w.slots)
        tables = w.get(f"engram{site}.tables").reshape(w.n_tables, w.slots, w.sub_dim)
        e = np.empty(w.d_model, np.float32)
        for i, table_idx in enumerate(idx):
            order = w.orders[i // w.heads_per_order]
            ok = 1.0 if t >= order - 1 else 0.0
            e[i * w.sub_dim:(i + 1) * w.sub_dim] = tables[i, table_idx] * ok
        k = w.get(f"engram{site}.key_proj") @ e
        v_now = w.get(f"engram{site}.value_proj") @ e
        span = self.ev_hist.shape[1]
        self.ev_hist[site, t % span] = v_now
        taps = w.get(f"engram{site}.taps")
        v = np.zeros(w.d_model, np.float32)
        for j in range(CONV_TAPS):
            src = t - j * w.dilation
            if src < 0:
                continue
            v += taps[j] * self.ev_hist[site, src % span]
        return k, v

    def _attention(self, layer, h):
        w = self.w
        hd, H, KV = w.head_dim, w.n_heads, w.n_kv
        p = f"layer{layer:02d}."
        q = (w.get(p + "q_proj") @ h).reshape(H, hd)
        k = (w.get(p + "k_proj") @ h).reshape(KV, hd)
        v = (w.get(p + "v_proj") @ h).reshape(KV, hd)
        q = zcrms(q, w.get(p + "q_norm"))
        k = zcrms(k, w.get(p + "k_norm"))
        q = self._rope(q)
        k = self._rope(k)

        t = self.pos
        self.k_cache[layer, :, t] = k
        self.v_cache[layer, :, t] = v

        lo = 0 if not w.kv_window else max(0, t - w.kv_window + 1)
        keep = np.arange(0, t + 1)
        if lo > 0:
            keep = np.concatenate([np.arange(0, min(self.sink_len, lo)), np.arange(lo, t + 1)])
        K = self.k_cache[layer][:, keep]          # (KV, n, hd)
        V = self.v_cache[layer][:, keep]
        reps = H // KV
        out = np.empty((H, hd), np.float32)
        for g in range(KV):
            scores = (K[g] @ q[g * reps:(g + 1) * reps].T).T / math.sqrt(hd)
            scores -= scores.max(axis=-1, keepdims=True)
            a = np.exp(scores)
            a /= a.sum(axis=-1, keepdims=True)
            out[g * reps:(g + 1) * reps] = a @ V[g]
        out = out.reshape(-1)
        out = out * sigmoid(w.get(p + "gate_proj") @ h)
        return w.get(p + "out_proj") @ out

    def _rope(self, x):
        hd = x.shape[-1]
        half = hd // 2
        c, s = self.cos[self.pos], self.sin[self.pos]
        x1, x2 = x[..., :half], x[..., half:]
        return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1)

    def _hadamard_mlp(self, layer, x):
        w = self.w
        p = f"layer{layer:02d}."
        z = (w.get(p + "d1") * x) @ self.H_hada
        z = silu(w.get(p + "d2") * z) @ self.H_hada
        return w.get(p + "d3") * z

    def _block(self, layer, x):
        w = self.w
        p = f"layer{layer:02d}."
        skip = x
        h = zcrms(x, w.get(p + "norm_in"))
        attn = self._attention(layer, h)
        attn = zcrms(attn, w.get(p + "post_norm"))
        x = skip + sigmoid(w.get(p + "attn_gate")[0]) * attn
        skip = x
        h = zcrms(x, w.get(p + "pre_hada"))
        return skip + self._hadamard_mlp(layer, h)

    # -- one step ---------------------------------------------------------
    def step(self, token):
        w = self.w
        self.tokens.append(int(token))
        emb = w.get("embedding")
        x = np.repeat((emb[token] * math.sqrt(self.d))[None, :], w.lanes, axis=0)

        ekv = [self._engram_kv(s) for s in range(len(w.sites))]

        phi_pre = w.get("mhc_phi_pre").reshape(w.n_layers, w.lanes, -1)
        phi_post = w.get("mhc_phi_post").reshape(w.n_layers, w.lanes, -1)
        phi_res = w.get("mhc_phi_res").reshape(w.n_layers, w.lanes * w.lanes, -1)
        a_pre, a_post, a_res = (w.get("mhc_a_pre"), w.get("mhc_a_post"), w.get("mhc_a_res"))
        b_pre, b_post, b_res = (w.get("mhc_b_pre"), w.get("mhc_b_post"), w.get("mhc_b_res"))

        for l in range(w.n_layers):
            lane = self.lane_onehot[l % w.lanes]
            nx = rms_unit(x.reshape(-1))
            hpre = sigmoid(a_pre[l] * (phi_pre[l] @ nx) + b_pre[l] + (8 * lane - 4))
            u = hpre @ x

            b = u
            if l in w.sites:
                site = w.sites.index(l)
                ek, ev = ekv[site]
                alpha = sigmoid(float(rms_unit(u) @ rms_unit(ek)) / math.sqrt(self.d))
                b = u + alpha * ev

            y = self._block(l, b) - u

            hpost = 2.0 * sigmoid(a_post[l] * (phi_post[l] @ nx) + b_post[l]
                                  + (-4 * (1 - lane)))
            A = a_res[l] * (phi_res[l] @ nx).reshape(w.lanes, w.lanes) + b_res[l]
            P = sinkhorn(A)
            x = P @ x + hpost[:, None] * y[None, :]

        self.pos += 1
        out = zcrms(x.mean(axis=0), w.get("final_norm"))
        return out @ emb.T

    def prefill(self, tokens):
        logits = None
        for t in tokens:
            logits = self.step(t)
        return logits
