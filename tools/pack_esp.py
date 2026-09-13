"""Repack a Needle `.cact` into the ESP32-S3 `.nsp` container.

Weight bytes are copied verbatim - same CQ indices, same fp16 group norms - so
the packed model is numerically the shipped one. What changes is the sectioning:
tensors streamed on every token go into one contiguous HOT block (destined for
PSRAM), the two engram tables stay COLD (four rows read per token, fine from
flash). The probe heads are dropped; the runtime does not use them.

    python3 tools/pack_esp.py ref/needle2.cact build/needle.nsp
"""

import argparse
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cact import Cact, tensor_names, unpack_lsb  # noqa: E402

MAGIC = 0x3150534E
VERSION = 1
ALIGN = 64
MMAP_ALIGN = 65536   # ESP32 flash MMU page: sections read via mmap start here

KIND_F16, KIND_CQ, KIND_CQ2P = 0, 1, 2
SEC_HOT, SEC_COLD = 0, 1

REC_FMT = "<IIIHBBB3x"
REC_SIZE = struct.calcsize(REC_FMT)
assert REC_SIZE == 20, REC_SIZE

HDR_FMT = ("<4I"        # magic version flags n_tensors
           "7I"         # vocab d_model n_heads n_kv_heads n_layers head_dim max_seq
           "3I"         # lanes kv_window kv_bits
           "5I"         # engram: slots sub_dim tables taps dilation
           "I4I"        # n_orders orders[4]
           "I4I"        # n_sites sites[4]
           "f"          # rope_theta
           "8Q"         # dir/hot/cold/tok (off,size)
           "I"          # n_cb
           "28f"        # cb
           "28b"        # cb_i8
           "5f"         # cb_scale
           "3I")        # tail
HDR_SIZE = struct.calcsize(HDR_FMT)


def align(n):
    return (n + ALIGN - 1) & ~(ALIGN - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--all-hot", action="store_true",
                    help="put the engram tables in the hot section too")
    ap.add_argument("--no-planar", action="store_true",
                    help="keep 2-bit tensors row-major (slower kernel)")
    args = ap.parse_args()

    c = Cact(args.src)
    h = c.hdr
    names = tensor_names(h, c.sites)
    idx = {n: i for i, n in enumerate(names)}

    # tensors the runtime walks, in canonical order
    keep = ["embedding"]
    for l in range(h["num_layers"]):
        keep += [f"layer{l:02d}.{s}" for s in
                 ("norm_in", "q_proj", "k_proj", "v_proj", "q_norm", "k_norm",
                  "gate_proj", "out_proj", "post_norm", "attn_gate", "pre_hada",
                  "d1", "d2", "d3")]
    keep += ["mhc_a_pre", "mhc_a_post", "mhc_a_res", "mhc_b_pre", "mhc_b_post",
             "mhc_b_res", "mhc_phi_pre", "mhc_phi_post", "mhc_phi_res"]
    for s in range(len(c.sites)):
        keep += [f"engram{s}.{n}" for n in ("tables", "key_proj", "value_proj", "taps")]
    keep.append("final_norm")

    def is_cold(name):
        return (not args.all_hot) and name.endswith(".tables")

    def to_planar(cact_idx):
        """Same indices and norms, re-laid-out group-major as two bit planes."""
        r = c.recs[cact_idx]
        out, in_dim = r["shape"]
        in_pad = (in_dim + 127) // 128 * 128
        groups = in_pad // 128
        row_bytes = in_pad * 2 // 8
        raw = c.blob(cact_idx)
        packed = raw[:out * row_bytes].reshape(out, row_bytes)
        norms = raw[out * row_bytes:].view(np.float16).reshape(out, groups)
        v = unpack_lsb(packed, 2, in_pad).reshape(out, groups, 128)
        p1 = np.packbits((v >> 1) & 1, axis=-1, bitorder="little")   # (out,g,16)
        p0 = np.packbits(v & 1, axis=-1, bitorder="little")
        planes = np.concatenate([p1, p0], axis=-1)                   # (out,g,32)
        planes = np.ascontiguousarray(planes.transpose(1, 0, 2))     # group-major
        return planes.tobytes() + np.ascontiguousarray(norms.T).tobytes()

    hot, cold = bytearray(), bytearray()
    recs = []
    for name in keep:
        r = c.recs[idx[name]]
        planar = (not args.no_planar) and r["dtype"] == 3 and r["bits"] == 2 \
                 and not name.endswith(".tables")   # gathered by row, not matvec'd
        blob = to_planar(idx[name]) if planar else c.blob(idx[name]).tobytes()
        sec = SEC_COLD if is_cold(name) else SEC_HOT
        dst = cold if sec == SEC_COLD else hot
        while len(dst) % ALIGN:
            dst += b"\0"
        off = len(dst)
        dst += blob
        shape = r["shape"]
        if r["dtype"] == 3:                      # CQ, [out, in]
            out, in_dim, bits = shape[0], shape[1], r["bits"]
            kind = KIND_CQ2P if planar else KIND_CQ
        else:                                    # fp16 vector or small matrix
            kind, bits = KIND_F16, 0
            out = shape[0] if shape else 1
            in_dim = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        if in_dim > 0xFFFF:
            raise ValueError(f"{name}: in={in_dim} does not fit u16")
        recs.append((off, len(blob), out, in_dim, kind, bits, sec))

    tok_blob = c.blob(idx["tokenizer"]).tobytes()

    dir_off = align(HDR_SIZE)
    dir_size = REC_SIZE * len(recs)
    def malign(n):
        return (n + MMAP_ALIGN - 1) & ~(MMAP_ALIGN - 1)

    hot_off = align(dir_off + dir_size)
    # cold and tokenizer are mmap'd straight from flash on the ESP32, so they
    # must land on flash-MMU page boundaries
    cold_off = malign(hot_off + len(hot))
    tok_off = malign(cold_off + len(cold))
    total = tok_off + len(tok_blob)

    cb = np.zeros(28, np.float32)
    cb[0:4] = c.codebooks[2]
    cb[4:12] = c.codebooks[3]
    cb[12:28] = c.codebooks[4]
    # int8 rescalings of the codebooks: carried in the header for format
    # stability, unused by this runtime (activations and kernels are fp32).
    cb_i8 = np.zeros(28, np.int8)
    cb_scale = np.zeros(5, np.float32)
    for bits, (lo, hi) in ((2, (0, 4)), (3, (4, 12)), (4, (12, 28))):
        v = cb[lo:hi]
        m = float(np.max(np.abs(v)))
        cb_i8[lo:hi] = np.clip(np.rint(v / m * 127.0), -127, 127).astype(np.int8)
        cb_scale[bits] = m / 127.0

    orders4 = list(c.orders) + [0] * (4 - len(c.orders))
    sites4 = list(c.sites) + [0] * (4 - len(c.sites))

    hdr = struct.pack(
        HDR_FMT,
        MAGIC, VERSION, 0, len(recs),
        h["vocab"], h["d_model"], h["num_heads"], h["num_kv_heads"],
        h["num_layers"], h["head_dim"], h["max_seq_len"],
        h["mhc_lanes"], h["kv_window"], h["kv_bits"],
        h["engram_slots"], h["engram_sub_dim"], h["num_engram_tables"],
        h["engram_conv_taps"], h["engram_conv_dilation"],
        len(c.orders), *orders4,
        len(c.sites), *sites4,
        h["rope_theta"],
        dir_off, dir_size, hot_off, len(hot), cold_off, len(cold),
        tok_off, len(tok_blob),
        28, *cb.tolist(), *cb_i8.tolist(), *cb_scale.tolist(), 0, 0, 0)

    out = bytearray(total)
    out[0:len(hdr)] = hdr
    p = dir_off
    for rec in recs:
        out[p:p + REC_SIZE] = struct.pack(REC_FMT, *rec)
        p += REC_SIZE
    out[hot_off:hot_off + len(hot)] = hot
    out[cold_off:cold_off + len(cold)] = cold
    out[tok_off:tok_off + len(tok_blob)] = tok_blob

    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    with open(args.dst, "wb") as f:
        f.write(out)

    mb = 1024 * 1024
    print(f"{args.dst}: {total/mb:.2f} MB total")
    print(f"  hot  (-> PSRAM) {len(hot)/mb:6.2f} MB   {len(recs)} tensors")
    print(f"  cold (-> flash) {len(cold)/mb:6.2f} MB")
    print(f"  tokenizer       {len(tok_blob)/1024:6.1f} KB")
    n_planar = sum(1 for r in recs if r[4] == KIND_CQ2P)
    print(f"  2-bit tensors in planar group-major layout: {n_planar}")
    src_mb = os.path.getsize(args.src) / mb
    print(f"  source .cact    {src_mb:6.2f} MB  (dropped probe heads: "
          f"{src_mb - total/mb:.2f} MB)")


if __name__ == "__main__":
    main()
