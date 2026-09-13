#include "needle_cq.h"

#include <math.h>

#define INV_SQRT_128 0.08838834764831845f

nq_shard_fn nq_shard = 0;
uint32_t    nq_shard_min_rows = 64;

typedef struct {
    const uint8_t  *blob;
    const uint16_t *norms;
    const nq_tables *t;
    const float    *xr;
    float          *y;
    int             ng, row_bytes, grp_bytes, bits;
} mv_args;

void nq_tables_init(nq_tables *t, const nsp_header *h)
{
    memcpy(t->cb, h->cb, sizeof t->cb);

    for (int b = 0; b < 256; b++) {
        /* indices are packed LSB-first: four 2-bit fields or two nibbles */
        for (int k = 0; k < 4; k++)
            t->lutf2[b][k] = h->cb[(b >> (2 * k)) & 3];
        t->lutf4[b][0] = h->cb[12 + (b & 15)];
        t->lutf4[b][1] = h->cb[12 + ((b >> 4) & 15)];
    }
    t->m2_a = h->cb[0];
    t->m2_b = h->cb[2] - h->cb[0];
    t->m2_g = h->cb[1] - h->cb[0];
    t->m2_d = h->cb[3] - h->cb[2] - h->cb[1] + h->cb[0];
}

void nq_build_sub(const float *xr, int ngroups, float *S)
{
    for (int g = 0; g < ngroups; g++) {
        const float *x = xr + g * NSP_GROUP;
        float *Sg = S + (size_t)g * (16 * 256);
        for (int k = 0; k < 16; k++) {
            float *Sk = Sg + k * 256;
            const float *xk = x + k * 8;
            Sk[0] = 0.0f;
            for (int i = 0; i < 8; i++) {
                float v = xk[i];
                int half = 1 << i;
                for (int j = 0; j < half; j++) Sk[half + j] = Sk[j] + v;
            }
        }
    }
}

typedef struct {
    const uint8_t  *planes;
    const uint16_t *norms;
    const float    *S;
    float          *y;
    uint32_t        out_rows;
    int             ng;
    float           a, b, g, d;
} sub_args;

static void sub_rows(void *arg, uint32_t row_lo, uint32_t row_hi)
{
    const sub_args *A = (const sub_args *)arg;
    for (uint32_t n = row_lo; n < row_hi; n++) A->y[n] = 0.0f;

    for (int g = 0; g < A->ng; g++) {
        const float *Sg = A->S + (size_t)g * (16 * 256);
        float total = 0.0f;
        for (int k = 0; k < 16; k++) total += Sg[k * 256 + 255];
        const float base = A->a * total;

        const uint8_t *p = A->planes + ((size_t)g * A->out_rows + row_lo) * 32;
        const uint16_t *nm = A->norms + (size_t)g * A->out_rows;
        for (uint32_t n = row_lo; n < row_hi; n++, p += 32) {
            float s1 = 0.0f, s0 = 0.0f;
            const float *Sk = Sg;
#if NEEDLE_CQ2_EXACT
            float sd = 0.0f;
            for (int k = 0; k < 16; k++, Sk += 256) {
                unsigned a1 = p[k], a0 = p[16 + k];
                s1 += Sk[a1];
                s0 += Sk[a0];
                sd += Sk[a1 & a0];
            }
            A->y[n] += nq_h2f(nm[n]) *
                       (base + A->b * s1 + A->g * s0 + A->d * sd);
#else
            /* Drop the bilinear residual. delta is 0.00016 against codebook
             * levels of 0.13, i.e. 0.1% of one level on a quarter of the
             * weights - far below the 2-bit grid itself - and it removes a
             * third of the inner loop. */
            for (int k = 0; k < 16; k++, Sk += 256) {
                s1 += Sk[p[k]];
                s0 += Sk[p[16 + k]];
            }
            A->y[n] += nq_h2f(nm[n]) * (base + A->b * s1 + A->g * s0);
#endif
        }
    }
}

void nq_matvec_cq2p(const uint8_t *blob, uint32_t out_rows, int in_pad,
                    const nq_tables *t, const float *S, float *y,
                    uint32_t row_lo, uint32_t row_hi)
{
    const int ng = in_pad / NSP_GROUP;
    sub_args A;
    A.planes = blob;
    A.norms = (const uint16_t *)(blob + (size_t)out_rows * (size_t)ng * 32);
    A.S = S;
    A.y = y;
    A.out_rows = out_rows;
    A.ng = ng;
    A.a = t->m2_a; A.b = t->m2_b; A.g = t->m2_g; A.d = t->m2_d;
    if (nq_shard && row_hi - row_lo >= nq_shard_min_rows)
        nq_shard(sub_rows, &A, row_lo, row_hi);
    else
        sub_rows(&A, row_lo, row_hi);
}

void nq_prep_f32(const float *x, int in_dim, int in_pad, float *xr)
{
    for (int base = 0; base < in_pad; base += NSP_GROUP) {
        int n = in_dim - base;
        if (n > NSP_GROUP) n = NSP_GROUP;
        if (n < 0) n = 0;
        float *g = xr + base;
        if (n) memcpy(g, x + base, (size_t)n * sizeof(float));
        for (int i = n; i < NSP_GROUP; i++) g[i] = 0.0f;
        nq_fwht(g, NSP_GROUP);
        for (int i = 0; i < NSP_GROUP; i++) g[i] *= INV_SQRT_128;
    }
}

void nq_fwht(float *v, int n)
{
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += (len << 1)) {
            float *a = v + i, *b = v + i + len;
            for (int j = 0; j < len; j++) {
                float x = a[j], y = b[j];
                a[j] = x + y;
                b[j] = x - y;
            }
        }
    }
}

static void mv_rows_f32(void *arg, uint32_t row_lo, uint32_t row_hi)
{
    const mv_args *a = (const mv_args *)arg;
    for (uint32_t n = row_lo; n < row_hi; n++) {
        const uint8_t *p = a->blob + (size_t)n * (size_t)a->row_bytes;
        const uint16_t *nm = a->norms + (size_t)n * (size_t)a->ng;
        float acc = 0.0f;
        for (int g = 0; g < a->ng; g++) {
            const float *x = a->xr + g * NSP_GROUP;
            float t = 0.0f;
            if (a->bits == 4) {
                for (int b = 0; b < NSP_GROUP / 2; b++) {
                    const float *w = a->t->lutf4[p[b]];
                    t += w[0] * x[0] + w[1] * x[1];
                    x += 2;
                }
            } else {
                for (int b = 0; b < NSP_GROUP / 4; b++) {
                    const float *w = a->t->lutf2[p[b]];
                    t += w[0] * x[0] + w[1] * x[1] + w[2] * x[2] + w[3] * x[3];
                    x += 4;
                }
            }
            p += a->grp_bytes;
            acc += nq_h2f(nm[g]) * t;
        }
        a->y[n] = acc;
    }
}

static void mv_run(nq_work_fn work, mv_args *a, uint32_t lo, uint32_t hi)
{
    if (nq_shard && hi - lo >= nq_shard_min_rows) nq_shard(work, a, lo, hi);
    else work(a, lo, hi);
}

void nq_matvec_f32(const uint8_t *blob, uint32_t out_rows, int in_pad, int bits,
                   const nq_tables *t, const float *xr, float *y,
                   uint32_t row_lo, uint32_t row_hi)
{
    const int ng = in_pad / NSP_GROUP;
    const int row_bytes = in_pad * bits / 8;
    mv_args a = { blob,
                  (const uint16_t *)(blob + (size_t)out_rows * (size_t)row_bytes),
                  t, xr, y, ng, row_bytes, NSP_GROUP * bits / 8, bits };
    mv_run(mv_rows_f32, &a, row_lo, row_hi);
}

void nq_row(const uint8_t *blob, uint32_t out_rows, int in_dim, int in_pad,
            int bits, const nq_tables *t, uint32_t row, float *out)
{
    const int ng = in_pad / NSP_GROUP;
    const int row_bytes = in_pad * bits / 8;
    const uint16_t *norms =
        (const uint16_t *)(blob + (size_t)out_rows * (size_t)row_bytes);
    const uint8_t *p = blob + (size_t)row * (size_t)row_bytes;
    const uint16_t *nm = norms + (size_t)row * (size_t)ng;
    const float *cb = t->cb + (bits == 2 ? 0 : bits == 3 ? 4 : 12);
    const uint32_t mask = (1u << bits) - 1u;

    float g[NSP_GROUP];
    for (int grp = 0; grp < ng; grp++) {
        const uint8_t *q = p + (size_t)grp * (NSP_GROUP * bits / 8);
        for (int i = 0; i < NSP_GROUP; i++) {
            uint32_t bit = (uint32_t)i * (uint32_t)bits;
            uint32_t byte = bit >> 3, sh = bit & 7;
            uint32_t v = ((uint32_t)q[byte] | ((uint32_t)q[byte + 1] << 8)) >> sh;
            g[i] = cb[v & mask];
        }
        float nrm = nq_h2f(nm[grp]);
        for (int i = 0; i < NSP_GROUP; i++) g[i] *= nrm;
        nq_fwht(g, NSP_GROUP);
        int base = grp * NSP_GROUP;
        int n = in_dim - base;
        if (n > NSP_GROUP) n = NSP_GROUP;
        for (int i = 0; i < n; i++) out[base + i] = g[i] * INV_SQRT_128;
    }
}
