"""Reader for the Needle `.cact` container.

Layout, all little-endian:

    header   120 bytes: 29 u32 fields then rope_theta as f32
    codebook codebook_len f32  (cb2[4] | cb3[8] | cb4[16])
    directory num_tensors records of 44 bytes, no names, fixed canon order
    blobs    64-byte aligned

A CQ tensor holds a logical [out, in] matrix: packed indices first
(out * in_pad*bits/8 bytes, LSB-first bitstream per row), then per-group L2
norms (out * in_pad/group fp16). One group reconstructs as

    w_group = (codebook[idx] * norm) @ H,  H = Walsh-Hadamard(group)/sqrt(group)
"""

import struct

import numpy as np

TAG = 0x05E12A83
ALIGN = 64
FP16, FP32, CQ, RAW = 1, 2, 3, 4
GROUP = 128

_HDR_FMT = "<29If"
_REC_FMT = "<BBHIIIIQQII"
REC_SIZE = struct.calcsize(_REC_FMT)

HDR_FIELDS = (
    "tag", "num_tensors", "codebook_len", "kv_window", "kv_bits",
    "vocab", "d_model", "num_heads", "num_kv_heads", "num_layers", "head_dim",
    "max_seq_len", "hada_n", "mhc_lanes", "engram_slots", "engram_sub_dim",
    "num_engram_tables", "engram_conv_taps", "engram_conv_dilation",
    "num_engram_orders", "o0", "o1", "o2", "o3",
    "num_engram_sites", "s0", "s1", "s2", "s3", "rope_theta",
)


def walsh(n):
    """Orthonormal Walsh-Hadamard matrix, natural (Hadamard) ordering."""
    h = np.array([[1.0]], np.float32)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return (h / np.sqrt(n)).astype(np.float32)


def wht_inplace(x):
    """Fast Walsh-Hadamard transform along the last axis, natural ordering.

    Equivalent to `x @ walsh(n)` up to the 1/sqrt(n) scale, in n log n.
    """
    n = x.shape[-1]
    step = 1
    while step < n:
        a = x[..., :].reshape(*x.shape[:-1], -1, 2 * step)
        lo = a[..., :step].copy()
        hi = a[..., step:].copy()
        a[..., :step] = lo + hi
        a[..., step:] = lo - hi
        step *= 2
    return x


class Cact:
    def __init__(self, path):
        self.raw = np.fromfile(path, dtype=np.uint8)
        buf = self.raw.tobytes()
        self.buf = buf
        vals = struct.unpack_from(_HDR_FMT, buf, 0)
        self.hdr = dict(zip(HDR_FIELDS, vals))
        if self.hdr["tag"] != TAG:
            raise ValueError("not a .cact file")
        n_cb = self.hdr["codebook_len"]
        cb = np.frombuffer(buf, np.float32, count=n_cb, offset=struct.calcsize(_HDR_FMT))
        # cb2[4] | cb3[8] | cb4[16]
        self.codebooks = {2: cb[0:4].copy(), 3: cb[4:12].copy(), 4: cb[12:28].copy()}
        self.orders = tuple(v for v in (self.hdr["o0"], self.hdr["o1"], self.hdr["o2"],
                                        self.hdr["o3"])[:self.hdr["num_engram_orders"]])
        self.sites = tuple(v for v in (self.hdr["s0"], self.hdr["s1"], self.hdr["s2"],
                                       self.hdr["s3"])[:self.hdr["num_engram_sites"]])

        off = struct.calcsize(_HDR_FMT) + 4 * n_cb
        self.recs = []
        for _ in range(self.hdr["num_tensors"]):
            dt, nd, _pad, s0, s1, s2, s3, boff, nbytes, group, bits = \
                struct.unpack_from(_REC_FMT, buf, off)
            self.recs.append(dict(dtype=dt, shape=(s0, s1, s2, s3)[:nd],
                                  off=boff, nbytes=nbytes, group=group, bits=bits))
            off += REC_SIZE

    # -- raw access -------------------------------------------------------
    def blob(self, i):
        r = self.recs[i]
        return self.raw[r["off"]:r["off"] + r["nbytes"]]

    def dequant(self, i):
        """Return tensor i as float32 (CQ tensors are reconstructed)."""
        r = self.recs[i]
        b = self.blob(i)
        if r["dtype"] == FP16:
            return b.view(np.float16).astype(np.float32).reshape(r["shape"])
        if r["dtype"] == FP32:
            return b.view(np.float32).reshape(r["shape"])
        if r["dtype"] == RAW:
            return b.tobytes()
        out, in_dim = r["shape"]
        return cq_unpack(b, out, in_dim, r["bits"], r["group"], self.codebooks[r["bits"]])

    def indices_and_norms(self, i):
        """Return (idx uint8 [out,in_pad], norms float32 [out,groups]) for a CQ tensor."""
        r = self.recs[i]
        assert r["dtype"] == CQ
        out, in_dim = r["shape"]
        bits, group = r["bits"], r["group"]
        in_pad = (in_dim + group - 1) // group * group
        row_bytes = in_pad * bits // 8
        b = self.blob(i)
        packed = b[:out * row_bytes].reshape(out, row_bytes)
        norms = b[out * row_bytes:].view(np.float16).reshape(out, in_pad // group)
        return unpack_lsb(packed, bits, in_pad), norms.astype(np.float32)


def unpack_lsb(packed, bits, in_pad):
    """Undo `_pack_lsb`: a continuous LSB-first bitstream of `bits` per index."""
    out = packed.shape[0]
    chunks = packed.reshape(out, in_pad // 8, bits).astype(np.uint64)
    word = np.zeros(chunks.shape[:-1], np.uint64)
    for b in range(bits):
        word |= chunks[..., b] << np.uint64(8 * b)
    mask = np.uint64((1 << bits) - 1)
    idx = np.empty((out, in_pad // 8, 8), np.uint8)
    for i in range(8):
        idx[..., i] = ((word >> np.uint64(i * bits)) & mask).astype(np.uint8)
    return idx.reshape(out, in_pad)


def cq_unpack(blob, out, in_dim, bits, group, codebook):
    in_pad = (in_dim + group - 1) // group * group
    row_bytes = in_pad * bits // 8
    packed = blob[:out * row_bytes].reshape(out, row_bytes)
    norms = blob[out * row_bytes:].view(np.float16).astype(np.float32)
    norms = norms.reshape(out, in_pad // group)
    idx = unpack_lsb(packed, bits, in_pad)
    unit = codebook[idx].reshape(out, in_pad // group, group)
    rot = unit * norms[:, :, None]
    w = (rot @ walsh(group)).reshape(out, in_pad)
    return w[:, :in_dim]


# -- canonical tensor order -----------------------------------------------
PER_LAYER = ("norm_in", "q_proj", "k_proj", "v_proj", "q_norm", "k_norm",
             "gate_proj", "out_proj", "post_norm", "attn_gate", "pre_hada",
             "d1", "d2", "d3")
MHC_NAMES = ("mhc_a_pre", "mhc_a_post", "mhc_a_res", "mhc_b_pre", "mhc_b_post",
             "mhc_b_res", "mhc_phi_pre", "mhc_phi_post", "mhc_phi_res")
PER_ENGRAM = ("tables", "key_proj", "value_proj", "taps")


def tensor_names(hdr, sites):
    """Positional name for every directory slot, in canon order."""
    names = ["embedding"]
    for i in range(hdr["num_layers"]):
        names += [f"layer{i:02d}.{n}" for n in PER_LAYER]
    names += list(MHC_NAMES)
    for s in range(len(sites)):
        names += [f"engram{s}.{n}" for n in PER_ENGRAM]
    names.append("final_norm")
    extra = hdr["num_tensors"] - len(names) - 1  # -1 for the tokenizer
    if extra > 0:
        names.append("heads.manifest")
        head_names = {1: "contrastive_head", 2: "confidence_head"}
        for h in range((extra - 1) // 3):
            names += [f"head{h}.probes", f"head{h}.proj", f"head{h}.bias"]
        del head_names
    names.append("tokenizer")
    return names


def parse_tokenizer_blob(blob):
    hdr_fmt, rec_fmt = "<IIIIIBBH", "<fBH"
    off = struct.calcsize(hdr_fmt)
    n, pad, eos, bos, unk, add_dummy, byte_fb, _ = struct.unpack_from(hdr_fmt, blob, 0)
    rec = struct.calcsize(rec_fmt)
    pieces, scores, types = [], [], []
    for _ in range(n):
        score, t, ln = struct.unpack_from(rec_fmt, blob, off)
        off += rec
        pieces.append(blob[off:off + ln].decode("utf-8"))
        scores.append(score)
        types.append(t)
        off += ln
    return dict(pieces=pieces, scores=scores, types=types, pad_id=pad, eos_id=eos,
                bos_id=bos, unk_id=unk, add_dummy_prefix=bool(add_dummy),
                byte_fallback=bool(byte_fb))
