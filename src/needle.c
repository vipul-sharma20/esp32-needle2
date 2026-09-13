#include "needle.h"
#include "needle_cq.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define ENGRAM_SEED  0x9E3779B9u
#define ENGRAM_PRIME 0x01000193u
#define SINKHORN_ITERS 20
#define MAX_LANES 8
#define MAX_TABLES 8
#define NRM_EPS 1e-6f

/* Per-phase profiling. Inert unless the host sets needle_now_us. */
long long (*needle_now_us)(void) = 0;
double needle_profile[NEEDLE_P_COUNT];
const char *const needle_phase_name[NEEDLE_P_COUNT] = {
    "prep(rotate+subsums)", "matvec CQ2", "matvec CQ4", "attention core",
    "engram", "hadamard MLP", "sinkhorn", "norms(zcrms)", "lane mix", "TOTAL"
};
void needle_profile_reset(void) { memset(needle_profile, 0, sizeof needle_profile); }
#define PROF_BEG() long long _pt = needle_now_us ? needle_now_us() : 0
#define PROF_END(i) do { if (needle_now_us) \
        needle_profile[(i)] += (double)(needle_now_us() - _pt); } while (0)

struct needle_ctx {
    const needle_model *m;
    const nsp_header   *h;
    nq_tables           T;
    needle_config       cfg;

    int kv_window, n_slots, sink_slots, sink_len;
    int reps, span;
    int pos;
    uint32_t tok_hist[8];

    int8_t  *kc, *vc;        /* [L][KV][n_slots][hd] */
    float   *ks, *vs;        /* [L][KV][n_slots]     */
    int32_t *slot_pos;       /* [n_slots]            */

    float *ev_hist;          /* [sites][span][d]     */
    float *ek;               /* [sites][d]           */
    float *ev;               /* [sites][d]           */

    float *x;                /* [lanes][d]  residual lanes */
    float *xn;               /* [lanes][d]  scratch        */
    float *nx;               /* [lanes*d]                  */
    float *u, *hb, *yb, *ab, *tb, *mb;
    float rope_inv[64];
    float rope_cos[64], rope_sin[64];
    float *qh, *kh, *vh;     /* projections                */
    float *scores;
    float *phi_out;          /* max(L*lanes*lanes) rows    */
    float *logits;

    int8_t *xq;    /* block-local activation (512)  */
    float  *gs;
    float  *nxr;   /* mHC activation, fp32 rotated; survives block() */
    float  *axr;   /* block-local fp32 rotated activation */
    float  *sub;   /* subset sums of axr, for the 2-bit kernel */
    int8_t *qq;              /* [heads][hd] int8 queries   */
    float  *qs;              /* [heads]                    */

    size_t bytes_fast, bytes_big;
};

/* ---- helpers ----------------------------------------------------------- */

static void *alloc_with(needle_alloc_fn fn, void *user, size_t n, size_t *tally)
{
    void *p = fn ? fn(n, user) : malloc(n);
    if (p) {
        memset(p, 0, n);
        *tally += n;
    }
    return p;
}

static inline const nsp_rec *rec_of(const needle_ctx *c, uint32_t i)
{
    return &c->m->dir[i];
}

static inline const uint8_t *blob_of(const needle_ctx *c, const nsp_rec *r)
{
    return (r->section == NSP_SEC_COLD ? c->m->cold : c->m->hot) + r->off;
}

static inline const uint16_t *f16_of(const needle_ctx *c, uint32_t i)
{
    return (const uint16_t *)blob_of(c, rec_of(c, i));
}

static inline int in_pad_of(const nsp_rec *r)
{
    return ((int)r->in + NSP_GROUP - 1) / NSP_GROUP * NSP_GROUP;
}

static inline float sigmoidf_(float x)
{
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    float e = expf(x);
    return e / (1.0f + e);
}

static void zcrms(float *dst, const float *src, const uint16_t *scale, int n)
{
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += src[i] * src[i];
    float inv = 1.0f / sqrtf(s / (float)n + NRM_EPS);
    for (int i = 0; i < n; i++) dst[i] = (1.0f + nq_h2f(scale[i])) * src[i] * inv;
}

static void rms_unit(float *dst, const float *src, int n)
{
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += src[i] * src[i];
    float inv = 1.0f / sqrtf(s / (float)n + NRM_EPS);
    for (int i = 0; i < n; i++) dst[i] = src[i] * inv;
}

/* Doubly-stochastic projection of a small square matrix (Sinkhorn).
 *
 * Must be done in the log domain. The routing logits A span ~2000 here, so
 * exponentiating first (even with a global max subtraction) flushes every
 * entry but one to zero and collapses P to a single 1. Per-row and per-column
 * log-sum-exp keeps each normalisation in range. */
static void sinkhorn(float *P, const float *A, int n, int iters)
{
    float lk[MAX_LANES * MAX_LANES];
    memcpy(lk, A, sizeof(float) * (size_t)(n * n));
    for (int it = 0; it < iters; it++) {
        for (int i = 0; i < n; i++) {
            float *r = lk + i * n;
            float mx = r[0];
            for (int j = 1; j < n; j++) if (r[j] > mx) mx = r[j];
            float s = 0.0f;
            for (int j = 0; j < n; j++) s += expf(r[j] - mx);
            float lse = mx + logf(s);
            for (int j = 0; j < n; j++) r[j] -= lse;
        }
        for (int j = 0; j < n; j++) {
            float mx = lk[j];
            for (int i = 1; i < n; i++) if (lk[i * n + j] > mx) mx = lk[i * n + j];
            float s = 0.0f;
            for (int i = 0; i < n; i++) s += expf(lk[i * n + j] - mx);
            float lse = mx + logf(s);
            for (int i = 0; i < n; i++) lk[i * n + j] -= lse;
        }
    }
    for (int i = 0; i < n * n; i++) P[i] = expf(lk[i]);
}

static void quant_i8(const float *src, int n, int8_t *dst, float *scale)
{
    float m = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = fabsf(src[i]);
        if (a > m) m = a;
    }
    if (m <= 0.0f) {
        memset(dst, 0, (size_t)n);
        *scale = 0.0f;
        return;
    }
    float inv = 127.0f / m;
    for (int i = 0; i < n; i++) {
        float q = src[i] * inv;
        int v = (int)(q < 0.0f ? q - 0.5f : q + 0.5f);
        if (v > 127) v = 127;
        else if (v < -127) v = -127;
        dst[i] = (int8_t)v;
    }
    *scale = m / 127.0f;
}

/* ---- model binding ----------------------------------------------------- */

int needle_model_open(needle_model *m, const void *image, size_t size)
{
    const uint8_t *p = (const uint8_t *)image;
    if (size < sizeof(nsp_header)) return -1;
    const nsp_header *h = (const nsp_header *)p;
    if (h->magic != NSP_MAGIC || h->version != NSP_VERSION) return -2;
    if (h->tok_off + h->tok_size > size) return -3;
    m->hdr = h;
    m->dir = (const nsp_rec *)(p + h->dir_off);
    m->hot = p + h->hot_off;
    m->cold = p + h->cold_off;
    m->tok = p + h->tok_off;
    m->tok_size = (size_t)h->tok_size;
    return 0;
}

int needle_model_open_split(needle_model *m, const void *header_and_dir,
                            const void *hot, const void *cold,
                            const void *tok, size_t tok_size)
{
    const uint8_t *p = (const uint8_t *)header_and_dir;
    const nsp_header *h = (const nsp_header *)p;
    if (h->magic != NSP_MAGIC || h->version != NSP_VERSION) return -2;
    m->hdr = h;
    m->dir = (const nsp_rec *)(p + h->dir_off);
    m->hot = (const uint8_t *)hot;
    m->cold = (const uint8_t *)cold;
    m->tok = (const uint8_t *)tok;
    m->tok_size = tok_size;
    return 0;
}

/* ---- lifecycle --------------------------------------------------------- */

needle_ctx *needle_create(const needle_model *m, const needle_config *cfg)
{
    const nsp_header *h = m->hdr;
    needle_ctx *c = (needle_ctx *)calloc(1, sizeof *c);
    if (!c) return NULL;
    c->m = m;
    c->h = h;
    c->cfg = cfg ? *cfg : (needle_config){0};
    nq_tables_init(&c->T, h);

    c->kv_window = c->cfg.kv_window > 0 ? c->cfg.kv_window : (int)h->kv_window;
    if (c->kv_window <= 0) c->kv_window = (int)h->max_seq;
    c->sink_slots = c->cfg.sink_slots > 0 ? c->cfg.sink_slots : 0;
    c->n_slots = c->sink_slots + c->kv_window;
    c->reps = (int)(h->n_heads / h->n_kv_heads);
    c->span = (int)((h->engram_taps - 1) * h->engram_dilation + 1);

    const int L = (int)h->n_layers, KV = (int)h->n_kv_heads;
    const int hd = (int)h->head_dim, d = (int)h->d_model;
    const int lanes = (int)h->lanes, S = (int)h->n_sites;
    const size_t kv_elems = (size_t)L * KV * c->n_slots;

    needle_alloc_fn ab = c->cfg.alloc_big, af = c->cfg.alloc_fast;
    void *usr = c->cfg.user;

#define BIG(n)  alloc_with(ab, usr, (n), &c->bytes_big)
#define FAST(n) alloc_with(af, usr, (n), &c->bytes_fast)

    c->kc = (int8_t *)BIG(kv_elems * (size_t)hd);
    c->vc = (int8_t *)BIG(kv_elems * (size_t)hd);
    c->ks = (float *)BIG(kv_elems * sizeof(float));
    c->vs = (float *)BIG(kv_elems * sizeof(float));
    /* Order matters: internal SRAM is only ~355 KB and fragments fast. The
     * subset-sum tables take 48 random lookups per row per group - by far the
     * densest access in the engine - so they are claimed first, then the
     * rotated activations they are built from. */
    const int max_in0 = (int)(lanes * d);
    c->sub = (float *)FAST((size_t)NSP_SUB_FLOATS * sizeof(float));
    c->axr = (float *)FAST((size_t)max_in0 * sizeof(float));
    c->nxr = (float *)FAST((size_t)max_in0 * sizeof(float));
    c->logits = (float *)FAST((size_t)h->vocab * sizeof(float));

    c->slot_pos = (int32_t *)FAST((size_t)c->n_slots * sizeof(int32_t));

    c->ev_hist = (float *)FAST((size_t)S * c->span * d * sizeof(float));
    c->ek = (float *)FAST((size_t)S * d * sizeof(float));
    c->ev = (float *)FAST((size_t)S * d * sizeof(float));

    c->x = (float *)FAST((size_t)lanes * d * sizeof(float));
    c->xn = (float *)FAST((size_t)lanes * d * sizeof(float));
    c->nx = (float *)FAST((size_t)lanes * d * sizeof(float));
    c->u = (float *)FAST((size_t)d * sizeof(float));
    c->hb = (float *)FAST((size_t)d * sizeof(float));
    c->yb = (float *)FAST((size_t)d * sizeof(float));
    c->ab = (float *)FAST((size_t)d * sizeof(float));
    c->tb = (float *)FAST((size_t)d * sizeof(float));
    c->mb = (float *)FAST((size_t)d * sizeof(float));
    c->qh = (float *)FAST((size_t)h->n_heads * hd * sizeof(float));
    c->kh = (float *)FAST((size_t)KV * hd * sizeof(float));
    c->vh = (float *)FAST((size_t)KV * hd * sizeof(float));
    c->scores = (float *)FAST((size_t)c->n_slots * sizeof(float));
    c->phi_out = (float *)FAST((size_t)L * lanes * lanes * sizeof(float));
    
    for (int i = 0; i < hd / 2 && i < 64; i++)
        c->rope_inv[i] = powf(h->rope_theta, -(float)(2 * i) / (float)hd);

    const int max_in = (int)(lanes * d);
    c->xq = (int8_t *)FAST((size_t)max_in);
    c->gs = (float *)FAST((size_t)(max_in / NSP_GROUP + 1) * sizeof(float));

    c->qq = (int8_t *)FAST((size_t)h->n_heads * hd);
    c->qs = (float *)FAST((size_t)h->n_heads * sizeof(float));

#undef BIG
#undef FAST

    if (!c->kc || !c->vc || !c->ks || !c->vs || !c->logits || !c->xq) {
        needle_free(c);
        return NULL;
    }
    needle_reset(c);
    return c;
}

void needle_free(needle_ctx *c)
{
    if (!c) return;
    /* Allocator callbacks must return memory that free() accepts;
     * heap_caps_malloc does. */
    {
        free(c->kc); free(c->vc); free(c->ks); free(c->vs);
    }
    {
        free(c->slot_pos); free(c->ev_hist); free(c->ek); free(c->ev);
        free(c->x); free(c->xn); free(c->nx); free(c->u); free(c->hb);
        free(c->yb); free(c->ab); free(c->tb); free(c->mb);
        free(c->qh); free(c->kh);
        free(c->vh); free(c->scores); free(c->phi_out); free(c->logits);
        free(c->xq); free(c->gs); free(c->nxr); free(c->axr); free(c->sub);
        free(c->qq); free(c->qs);
    }
    free(c);
}

void needle_reset(needle_ctx *c)
{
    c->pos = 0;
    c->sink_len = 0;
    memset(c->tok_hist, 0, sizeof c->tok_hist);
    for (int i = 0; i < c->n_slots; i++) c->slot_pos[i] = -1;
    memset(c->ev_hist, 0,
           (size_t)c->h->n_sites * c->span * c->h->d_model * sizeof(float));
}

void needle_pin_sinks(needle_ctx *c)
{
    c->sink_len = c->pos < c->sink_slots ? c->pos : c->sink_slots;
}

int needle_pos(const needle_ctx *c) { return c->pos; }
size_t needle_bytes_fast(const needle_ctx *c) { return c->bytes_fast; }
size_t needle_bytes_big(const needle_ctx *c) { return c->bytes_big; }
const void *needle_sub_tables(const needle_ctx *c) { return c->sub; }

/* ---- pieces ------------------------------------------------------------ */

static inline int slot_for(const needle_ctx *c, int p)
{
    if (p < c->sink_slots) return p;
    return c->sink_slots + (p - c->sink_slots) % c->kv_window;
}

/* Rotate the activation once; 2-bit tensors then also want its subset sums.
 * Activations stay fp32 throughout - the subset-sum kernel is exact, so there
 * is nothing to gain by quantising them. */
static void prep(needle_ctx *c, const float *x, int in_dim, int in_pad)
{
    PROF_BEG();
    nq_prep_f32(x, in_dim, in_pad, c->axr);
    int ng = in_pad / NSP_GROUP;
    if (ng <= NSP_SUB_GROUPS) nq_build_sub(c->axr, ng, c->sub);
    PROF_END(NEEDLE_P_PREP);
}

static void mv(const needle_ctx *c, uint32_t ti, float *y,
               uint32_t lo, uint32_t hi)
{
    const nsp_rec *r = rec_of(c, ti);
    PROF_BEG();
    if (r->kind == NSP_KIND_CQ2P) {
        nq_matvec_cq2p(blob_of(c, r), r->out, in_pad_of(r), &c->T,
                       c->sub, y, lo, hi);
        PROF_END(NEEDLE_P_MV2);
    } else {
        nq_matvec_f32(blob_of(c, r), r->out, in_pad_of(r), r->bits, &c->T,
                      c->axr, y, lo, hi);
        PROF_END(NEEDLE_P_MV4);
    }
}

/* mHC projections read a 2048-wide activation, too wide for subset sums. */
static void mvf(const needle_ctx *c, uint32_t ti, const float *xr,
                float *y, uint32_t lo, uint32_t hi)
{
    const nsp_rec *r = rec_of(c, ti);
    PROF_BEG();
    nq_matvec_f32(blob_of(c, r), r->out, in_pad_of(r), r->bits, &c->T,
                  xr, y, lo, hi);
    PROF_END(NEEDLE_P_MV4);
}

static void engram_step(needle_ctx *c)
{
    const nsp_header *h = c->h;
    const int d = (int)h->d_model, sub = (int)h->engram_sub_dim;
    const int heads = (int)(h->engram_tables / h->n_orders);
    uint32_t slots[MAX_TABLES];

    for (uint32_t oi = 0, t = 0; oi < h->n_orders; oi++) {
        uint32_t order = h->orders[oi];
        for (int hh = 0; hh < heads; hh++, t++) {
            uint32_t acc = ENGRAM_SEED * (uint32_t)(oi * (uint32_t)heads + hh + 1);
            for (uint32_t j = 0; j < order; j++) {
                uint32_t tok = ((int)j <= c->pos)
                    ? c->tok_hist[(unsigned)(c->pos - (int)j) & 7u] : 0u;
                acc = (acc ^ tok) * ENGRAM_PRIME;
            }
            acc ^= acc >> 15;
            slots[t] = acc % h->engram_slots;
        }
    }

    for (uint32_t s = 0; s < h->n_sites; s++) {
        const nsp_rec *rt = rec_of(c, NSP_T_ENGRAM(h, s, NSP_E_TABLES));
        float *e = c->tb;
        for (uint32_t t = 0; t < h->engram_tables; t++) {
            uint32_t order = h->orders[t / (uint32_t)heads];
            float *dst = e + (size_t)t * sub;
            if (c->pos < (int)order - 1) {
                memset(dst, 0, (size_t)sub * sizeof(float));
                continue;
            }
            nq_row(blob_of(c, rt), rt->out, sub, sub, rt->bits, &c->T,
                   t * h->engram_slots + slots[t], dst);
        }

        prep(c, e, d, d);
        float *ek = c->ek + (size_t)s * d;
        mv(c, NSP_T_ENGRAM(h, s, NSP_E_KEY), ek, 0, (uint32_t)d);

        float *slot = c->ev_hist + ((size_t)s * c->span + (c->pos % c->span)) * d;
        mv(c, NSP_T_ENGRAM(h, s, NSP_E_VALUE), slot, 0, (uint32_t)d);

        const uint16_t *taps = f16_of(c, NSP_T_ENGRAM(h, s, NSP_E_TAPS));
        float *ev = c->ev + (size_t)s * d;
        memset(ev, 0, (size_t)d * sizeof(float));
        for (uint32_t j = 0; j < h->engram_taps; j++) {
            int src = c->pos - (int)(j * h->engram_dilation);
            if (src < 0) continue;
            const float *v = c->ev_hist + ((size_t)s * c->span + (src % c->span)) * d;
            const uint16_t *tp = taps + (size_t)j * d;
            for (int i = 0; i < d; i++) ev[i] += nq_h2f(tp[i]) * v[i];
        }
    }
}

/* The rotation depends only on the position, so the table is built once per
 * token rather than per head per layer - that was 20,736 sinf/cosf calls a
 * token for 32 distinct values. */
static void rope_begin(needle_ctx *c, int pos)
{
    const int half = (int)c->h->head_dim / 2;
    for (int i = 0; i < half; i++) {
        float ang = (float)pos * c->rope_inv[i];
        c->rope_cos[i] = cosf(ang);
        c->rope_sin[i] = sinf(ang);
    }
}

static void rope_pair(const needle_ctx *c, float *v, int hd)
{
    const int half = hd / 2;
    const float *cs = c->rope_cos, *sn = c->rope_sin;
    for (int i = 0; i < half; i++) {
        float a = v[i], b = v[i + half];
        v[i] = a * cs[i] - b * sn[i];
        v[i + half] = b * cs[i] + a * sn[i];
    }
}

static void attention(needle_ctx *c, int l, const float *h_in)
{
    const nsp_header *h = c->h;
    const int d = (int)h->d_model, hd = (int)h->head_dim;
    const int H = (int)h->n_heads, KV = (int)h->n_kv_heads;
    const int kvd = KV * hd;

    prep(c, h_in, d, d);
    mv(c, NSP_T_LAYER(h, l, NSP_L_Q), c->qh, 0, (uint32_t)(H * hd));
    mv(c, NSP_T_LAYER(h, l, NSP_L_K), c->kh, 0, (uint32_t)kvd);
    mv(c, NSP_T_LAYER(h, l, NSP_L_V), c->vh, 0, (uint32_t)kvd);
    mv(c, NSP_T_LAYER(h, l, NSP_L_GATE), c->ab, 0, (uint32_t)d);

    const uint16_t *qn = f16_of(c, NSP_T_LAYER(h, l, NSP_L_QNORM));
    const uint16_t *kn = f16_of(c, NSP_T_LAYER(h, l, NSP_L_KNORM));
    for (int i = 0; i < H; i++) {
        zcrms(c->qh + (size_t)i * hd, c->qh + (size_t)i * hd, qn, hd);
        rope_pair(c, c->qh + (size_t)i * hd, hd);
        quant_i8(c->qh + (size_t)i * hd, hd, c->qq + (size_t)i * hd, &c->qs[i]);
    }
    const int slot = slot_for(c, c->pos);
    for (int g = 0; g < KV; g++) {
        zcrms(c->kh + (size_t)g * hd, c->kh + (size_t)g * hd, kn, hd);
        rope_pair(c, c->kh + (size_t)g * hd, hd);
        size_t base = ((size_t)l * KV + (size_t)g) * (size_t)c->n_slots + (size_t)slot;
        quant_i8(c->kh + (size_t)g * hd, hd, c->kc + base * (size_t)hd, &c->ks[base]);
        quant_i8(c->vh + (size_t)g * hd, hd, c->vc + base * (size_t)hd, &c->vs[base]);
    }
    c->slot_pos[slot] = c->pos;

    PROF_BEG();
    const float inv_sqrt_hd = 1.0f / sqrtf((float)hd);
    const int oldest = c->pos - c->kv_window + 1;

    for (int g = 0; g < KV; g++) {
        const size_t gbase = ((size_t)l * KV + (size_t)g) * (size_t)c->n_slots;
        for (int r = 0; r < c->reps; r++) {
            const int head = g * c->reps + r;
            const int8_t *q = c->qq + (size_t)head * hd;
            const float qsc = c->qs[head] * inv_sqrt_hd;
            float mx = 0.0f;
            int nvalid = 0;
            for (int i = 0; i < c->n_slots; i++) {
                int p = c->slot_pos[i];
                if (p < 0 || p > c->pos) continue;
                if (p < oldest && p >= c->sink_len) continue;
                const int8_t *k = c->kc + (gbase + (size_t)i) * (size_t)hd;
                int32_t acc = 0;
                for (int t = 0; t < hd; t++) acc += (int32_t)q[t] * (int32_t)k[t];
                float sc = (float)acc * qsc * c->ks[gbase + (size_t)i];
                c->scores[i] = sc;
                if (!nvalid || sc > mx) mx = sc;
                nvalid++;
            }
            if (!nvalid) continue;
            float sum = 0.0f;
            for (int i = 0; i < c->n_slots; i++) {
                int p = c->slot_pos[i];
                if (p < 0 || p > c->pos) continue;
                if (p < oldest && p >= c->sink_len) continue;
                float e = expf(c->scores[i] - mx);
                c->scores[i] = e;
                sum += e;
            }
            float inv = 1.0f / sum;
            float *out = c->tb + (size_t)head * hd;
            memset(out, 0, (size_t)hd * sizeof(float));
            for (int i = 0; i < c->n_slots; i++) {
                int p = c->slot_pos[i];
                if (p < 0 || p > c->pos) continue;
                if (p < oldest && p >= c->sink_len) continue;
                float w = c->scores[i] * inv * c->vs[gbase + (size_t)i];
                const int8_t *v = c->vc + (gbase + (size_t)i) * (size_t)hd;
                for (int t = 0; t < hd; t++) out[t] += w * (float)v[t];
            }
        }
    }

    PROF_END(NEEDLE_P_ATTN);

    for (int i = 0; i < d; i++) c->tb[i] *= sigmoidf_(c->ab[i]);
    prep(c, c->tb, d, d);
    mv(c, NSP_T_LAYER(h, l, NSP_L_OUT), c->ab, 0, (uint32_t)d);
}

static void hadamard_mlp(needle_ctx *c, int l, const float *in, float *out)
{
    PROF_BEG();
    const int d = (int)c->h->d_model;
    const uint16_t *d1 = f16_of(c, NSP_T_LAYER(c->h, l, NSP_L_D1));
    const uint16_t *d2 = f16_of(c, NSP_T_LAYER(c->h, l, NSP_L_D2));
    const uint16_t *d3 = f16_of(c, NSP_T_LAYER(c->h, l, NSP_L_D3));
    const float inv = 1.0f / sqrtf((float)d);

    for (int i = 0; i < d; i++) out[i] = nq_h2f(d1[i]) * in[i];
    nq_fwht(out, d);
    for (int i = 0; i < d; i++) {
        float z = out[i] * inv * nq_h2f(d2[i]);
        out[i] = z * sigmoidf_(z);            /* SiLU */
    }
    nq_fwht(out, d);
    for (int i = 0; i < d; i++) out[i] *= inv * nq_h2f(d3[i]);
    PROF_END(NEEDLE_P_HADA);
}

static void block(needle_ctx *c, int l, float *b)
{
    const nsp_header *h = c->h;
    const int d = (int)h->d_model;

    { PROF_BEG(); zcrms(c->hb, b, f16_of(c, NSP_T_LAYER(h, l, NSP_L_NORM_IN)), d);
      PROF_END(NEEDLE_P_NORM); }
    attention(c, l, c->hb);
    zcrms(c->ab, c->ab, f16_of(c, NSP_T_LAYER(h, l, NSP_L_POST)), d);
    float g = sigmoidf_(nq_h2f(*f16_of(c, NSP_T_LAYER(h, l, NSP_L_ATTN_GATE))));
    for (int i = 0; i < d; i++) b[i] += g * c->ab[i];
    zcrms(c->hb, b, f16_of(c, NSP_T_LAYER(h, l, NSP_L_PRE_HADA)), d);
    hadamard_mlp(c, l, c->hb, c->mb);
    for (int i = 0; i < d; i++) b[i] += c->mb[i];
}

/* ---- one token --------------------------------------------------------- */

const float *needle_step(needle_ctx *c, int token)
{
    const nsp_header *h = c->h;
    const int d = (int)h->d_model, lanes = (int)h->lanes;
    const int L = (int)h->n_layers;
    const uint32_t mhc = NSP_T_MHC(h);

    PROF_BEG();
    rope_begin(c, c->pos);
    c->tok_hist[(unsigned)c->pos & 7u] = (uint32_t)token;

    const nsp_rec *re = rec_of(c, NSP_T_EMBEDDING);
    nq_row(blob_of(c, re), re->out, d, in_pad_of(re), re->bits, &c->T,
           (uint32_t)token, c->u);
    const float es = sqrtf((float)d);
    for (int i = 0; i < d; i++) c->u[i] *= es;
    for (int k = 0; k < lanes; k++)
        memcpy(c->x + (size_t)k * d, c->u, (size_t)d * sizeof(float));
    { PROF_BEG(); engram_step(c); PROF_END(NEEDLE_P_ENGRAM); }
    for (uint32_t s = 0; s < h->n_sites; s++) {
    }

    const uint16_t *a_pre = f16_of(c, mhc + NSP_M_A_PRE);
    const uint16_t *a_post = f16_of(c, mhc + NSP_M_A_POST);
    const uint16_t *a_res = f16_of(c, mhc + NSP_M_A_RES);
    const uint16_t *b_pre = f16_of(c, mhc + NSP_M_B_PRE);
    const uint16_t *b_post = f16_of(c, mhc + NSP_M_B_POST);
    const uint16_t *b_res = f16_of(c, mhc + NSP_M_B_RES);

    float hpre[MAX_LANES], hpost[MAX_LANES], A[MAX_LANES * MAX_LANES],
          P[MAX_LANES * MAX_LANES];

    for (int l = 0; l < L; l++) {
        const int lane = l % lanes;
        rms_unit(c->nx, c->x, lanes * d);
        nq_prep_f32(c->nx, lanes * d, lanes * d, c->nxr);
        const uint32_t r0 = (uint32_t)l * (uint32_t)lanes;
        mvf(c, mhc + NSP_M_PHI_PRE, c->nxr, c->phi_out, r0, r0 + lanes);
        float apre = nq_h2f(a_pre[l]);
        for (int k = 0; k < lanes; k++) {
            float off = (k == lane) ? 4.0f : -4.0f;
            hpre[k] = sigmoidf_(apre * c->phi_out[r0 + k]
                                + nq_h2f(b_pre[(size_t)l * lanes + k]) + off);
        }
        for (int i = 0; i < d; i++) {
            float s = 0.0f;
            for (int k = 0; k < lanes; k++) s += hpre[k] * c->x[(size_t)k * d + i];
            c->u[i] = s;
        }
        memcpy(c->yb, c->u, (size_t)d * sizeof(float));
        for (uint32_t s = 0; s < h->n_sites; s++) {
            if ((int)h->sites[s] != l) continue;
            const float *ek = c->ek + (size_t)s * d;
            const float *ev = c->ev + (size_t)s * d;
            rms_unit(c->hb, c->u, d);
            rms_unit(c->ab, ek, d);
            float dp = 0.0f;
            for (int i = 0; i < d; i++) dp += c->hb[i] * c->ab[i];
            float alpha = sigmoidf_(dp / sqrtf((float)d));
            for (int i = 0; i < d; i++) c->yb[i] += alpha * ev[i];
        }
        block(c, l, c->yb);
        for (int i = 0; i < d; i++) c->yb[i] -= c->u[i];

        mvf(c, mhc + NSP_M_PHI_POST, c->nxr, c->phi_out, r0, r0 + lanes);
        float apost = nq_h2f(a_post[l]);
        for (int k = 0; k < lanes; k++) {
            float off = (k == lane) ? 0.0f : -4.0f;
            hpost[k] = 2.0f * sigmoidf_(apost * c->phi_out[r0 + k]
                                        + nq_h2f(b_post[(size_t)l * lanes + k]) + off);
        }
        const uint32_t rr = (uint32_t)l * (uint32_t)(lanes * lanes);
        mvf(c, mhc + NSP_M_PHI_RES, c->nxr, c->phi_out, rr,
           rr + (uint32_t)(lanes * lanes));
        float ares = nq_h2f(a_res[l]);
        for (int i = 0; i < lanes * lanes; i++)
            A[i] = ares * c->phi_out[rr + i]
                 + nq_h2f(b_res[(size_t)l * lanes * lanes + i]);
        { PROF_BEG(); sinkhorn(P, A, lanes, SINKHORN_ITERS);
          PROF_END(NEEDLE_P_SINK); }

        PROF_BEG();
        for (int i = 0; i < lanes; i++) {
            float *dst = c->xn + (size_t)i * d;
            for (int t = 0; t < d; t++) dst[t] = hpost[i] * c->yb[t];
            for (int j = 0; j < lanes; j++) {
                float p = P[i * lanes + j];
                const float *src = c->x + (size_t)j * d;
                for (int t = 0; t < d; t++) dst[t] += p * src[t];
            }
        }
        memcpy(c->x, c->xn, (size_t)lanes * d * sizeof(float));
        PROF_END(NEEDLE_P_LANE);
    }

    for (int i = 0; i < d; i++) {
        float s = 0.0f;
        for (int k = 0; k < lanes; k++) s += c->x[(size_t)k * d + i];
        c->u[i] = s / (float)lanes;
    }
    zcrms(c->hb, c->u, f16_of(c, NSP_T_FINAL_NORM(h)), d);
    prep(c, c->hb, d, in_pad_of(re));
    uint32_t lo = c->cfg.row_hi > c->cfg.row_lo ? (uint32_t)c->cfg.row_lo : 0;
    uint32_t hi = c->cfg.row_hi > c->cfg.row_lo ? (uint32_t)c->cfg.row_hi : h->vocab;
    nq_matvec_f32(blob_of(c, re), re->out, in_pad_of(re), re->bits, &c->T,
                  c->axr, c->logits, lo, hi);

    PROF_END(NEEDLE_P_TOTAL);
    c->pos++;
    return c->logits;
}
