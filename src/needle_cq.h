/* Cactus-Quants kernels. */
#ifndef NEEDLE_CQ_H
#define NEEDLE_CQ_H

#include <stdint.h>
#include <string.h>

#include "nsp.h"

#ifndef NEEDLE_CQ2_EXACT
#define NEEDLE_CQ2_EXACT 1
#endif

typedef struct {
    float    cb[28];        /* the shipped Lloyd-Max codebooks, cb2|cb3|cb4 */
    float    lutf2[256][4]; /* a packed byte expanded to its four cb2 values */
    float    lutf4[256][2]; /* a packed byte expanded to its two  cb4 values */
    /* Multilinear form of the 2-bit codebook, exact:
     *   cb[v] = a + b*bit1(v) + g*bit0(v) + d*bit1(v)*bit0(v)
     * which turns a dot product into four subset sums of the activation. */
    float    m2_a, m2_b, m2_g, m2_d;
} nq_tables;

void nq_tables_init(nq_tables *t, const nsp_header *h);

static inline float nq_h2f(uint16_t h)
{
    uint32_t s = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1Fu;
    uint32_t m = h & 0x3FFu;
    uint32_t bits;
    if (e == 0) {
        if (m == 0) {
            bits = s;
        } else {
            int sh = 0;
            while (!(m & 0x400u)) { m <<= 1; sh++; }
            m &= 0x3FFu;
            bits = s | (uint32_t)((1 - sh + 127 - 15) << 23) | (m << 13);
        }
    } else if (e == 31) {
        bits = s | 0x7F800000u | (m << 13);
    } else {
        bits = s | ((e + 127 - 15) << 23) | (m << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

/* In-place fast Walsh-Hadamard transform, natural (Sylvester) ordering.
 * Equals v @ H_nat; the 1/sqrt(n) of the orthonormal H is left to the caller. */
void nq_fwht(float *v, int n);

/* Rotate and int8-quantise an activation, one 128-group at a time.
 * gscale[g] carries both the int8 step and the 1/sqrt(128) of the rotation. */
void nq_prep(const float *x, int in_dim, int in_pad, int8_t *xq, float *gscale);

/* Optional row sharding. When set, nq_matvec/nq_matvec_f32 hand their row
 * range to this callback, which must invoke `work(arg, lo, hi)` over disjoint
 * sub-ranges covering [lo, hi) and return once all are done. Rows are
 * independent, so any split gives bit-identical results. */
typedef void (*nq_work_fn)(void *arg, uint32_t lo, uint32_t hi);
typedef void (*nq_shard_fn)(nq_work_fn work, void *arg, uint32_t lo, uint32_t hi);
extern nq_shard_fn nq_shard;
extern uint32_t    nq_shard_min_rows;   /* below this, run inline. default 64 */

/* y[n] = W[n,:] . x for n in [row_lo, row_hi), W held as CQ. */
void nq_matvec(const uint8_t *blob, uint32_t out_rows, int in_pad, int bits,
               const nq_tables *t, const int8_t *xq, const float *gscale,
               float *y, uint32_t row_lo, uint32_t row_hi);

/* Precision path: keep the rotated activation in fp32 instead of int8.
 * The mHC projections need this - a_pre * (phi . nx) turns a 0.5% error in a
 * 2048-wide dot product into a gate flip. */
void nq_prep_f32(const float *x, int in_dim, int in_pad, float *xr);

void nq_matvec_f32(const uint8_t *blob, uint32_t out_rows, int in_pad, int bits,
                   const nq_tables *t, const float *xr, float *y,
                   uint32_t row_lo, uint32_t row_hi);

/* Subset sums of one rotated activation: S[g][k][m] = sum of xr[g*128+k*8+i]
 * over the bits i set in m. 16 KB of fp32 per 128-group, built once and reused
 * by every 2-bit matvec that reads this activation. */
void nq_build_sub(const float *xr, int ngroups, float *S);

/* 2-bit matvec over the planar group-major layout. No per-weight multiply:
 * each row costs 16 table lookups per bit plane. Exact in fp32. */
void nq_matvec_cq2p(const uint8_t *blob, uint32_t out_rows, int in_pad,
                    const nq_tables *t, const float *S, float *y,
                    uint32_t row_lo, uint32_t row_hi);

/* Reconstruct one CQ row to fp32 (embedding lookup, engram table gather). */
void nq_row(const uint8_t *blob, uint32_t out_rows, int in_dim, int in_pad,
            int bits, const nq_tables *t, uint32_t row, float *out);

#endif /* NEEDLE_CQ_H */
