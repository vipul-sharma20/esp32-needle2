/* SentencePiece-BPE over the blob embedded in the model.
 *
 * blob:
 *   u32 n_pieces, u32 pad, u32 eos, u32 bos, u32 unk,
 *   u8 add_dummy_prefix, u8 byte_fallback, u16 _pad
 *   then n_pieces records: f32 score, u8 type, u16 len, len bytes
 */
#include "needle.h"

#include <stdlib.h>
#include <string.h>

enum { TK_NORMAL = 0, TK_UNKNOWN, TK_CONTROL, TK_USER_DEFINED, TK_BYTE };

#define SP_SPACE "\xE2\x96\x81"   /* U+2581 */
#define SP_SPACE_LEN 3

typedef struct {
    const char *s;
    uint16_t    len;
    float       score;
    uint8_t     type;
} piece_t;

struct needle_tok {
    piece_t  *p;
    int       n;
    int       pad_id, eos_id, bos_id, unk_id;
    int       add_dummy, byte_fallback;
    int       byte_id[256];
    /* open-addressed piece -> id */
    int32_t  *hash;
    uint32_t  hmask;
    int      *markers;   /* user-defined ids, longest surface first */
    int       n_markers;
};

static uint32_t hash_str(const char *s, int len)
{
    uint32_t h = 2166136261u;
    for (int i = 0; i < len; i++) h = (h ^ (uint8_t)s[i]) * 16777619u;
    return h ? h : 1u;
}

static int tok_lookup(const needle_tok *t, const char *s, int len)
{
    uint32_t i = hash_str(s, len) & t->hmask;
    for (;;) {
        int32_t id = t->hash[i];
        if (id < 0) return -1;
        const piece_t *p = &t->p[id];
        if (p->len == len && memcmp(p->s, s, (size_t)len) == 0) return id;
        i = (i + 1) & t->hmask;
    }
}

static void tok_insert(needle_tok *t, int id)
{
    const piece_t *p = &t->p[id];
    uint32_t i = hash_str(p->s, p->len) & t->hmask;
    while (t->hash[i] >= 0) i = (i + 1) & t->hmask;
    t->hash[i] = id;
}

static uint32_t rd32(const uint8_t *p) { uint32_t v; memcpy(&v, p, 4); return v; }
static uint16_t rd16(const uint8_t *p) { uint16_t v; memcpy(&v, p, 2); return v; }
static float rdf(const uint8_t *p) { float v; memcpy(&v, p, 4); return v; }

needle_tok *needle_tok_open(const uint8_t *blob, size_t size)
{
    if (size < 24) return NULL;
    needle_tok *t = (needle_tok *)calloc(1, sizeof *t);
    if (!t) return NULL;
    t->n = (int)rd32(blob);
    t->pad_id = (int)rd32(blob + 4);
    t->eos_id = (int)rd32(blob + 8);
    t->bos_id = (int)rd32(blob + 12);
    t->unk_id = (int)rd32(blob + 16);
    t->add_dummy = blob[20];
    t->byte_fallback = blob[21];

    t->p = (piece_t *)calloc((size_t)t->n, sizeof(piece_t));
    if (!t->p) { free(t); return NULL; }
    for (int i = 0; i < 256; i++) t->byte_id[i] = -1;

    size_t off = 24;
    int n_mark = 0;
    for (int i = 0; i < t->n; i++) {
        if (off + 7 > size) { needle_tok_free(t); return NULL; }
        t->p[i].score = rdf(blob + off);
        t->p[i].type = blob[off + 4];
        t->p[i].len = rd16(blob + off + 5);
        off += 7;
        if (off + t->p[i].len > size) { needle_tok_free(t); return NULL; }
        t->p[i].s = (const char *)(blob + off);
        off += t->p[i].len;
        if (t->p[i].type == TK_BYTE && t->p[i].len >= 5) {
            /* surface is "<0xAB>" */
            const char *h = t->p[i].s + 3;
            int hi = h[0] <= '9' ? h[0] - '0' : (h[0] | 32) - 'a' + 10;
            int lo = h[1] <= '9' ? h[1] - '0' : (h[1] | 32) - 'a' + 10;
            t->byte_id[(hi << 4) | lo] = i;
        }
        if (t->p[i].type == TK_USER_DEFINED) n_mark++;
    }

    uint32_t cap = 1;
    while (cap < (uint32_t)t->n * 2u) cap <<= 1;
    t->hmask = cap - 1;
    t->hash = (int32_t *)malloc((size_t)cap * sizeof(int32_t));
    if (!t->hash) { needle_tok_free(t); return NULL; }
    for (uint32_t i = 0; i < cap; i++) t->hash[i] = -1;
    for (int i = 0; i < t->n; i++) tok_insert(t, i);

    t->markers = (int *)calloc((size_t)(n_mark ? n_mark : 1), sizeof(int));
    if (!t->markers) { needle_tok_free(t); return NULL; }
    for (int i = 0; i < t->n; i++)
        if (t->p[i].type == TK_USER_DEFINED) t->markers[t->n_markers++] = i;
    for (int i = 1; i < t->n_markers; i++)       /* insertion sort, longest first */
        for (int j = i; j > 0 &&
             t->p[t->markers[j]].len > t->p[t->markers[j - 1]].len; j--) {
            int tmp = t->markers[j];
            t->markers[j] = t->markers[j - 1];
            t->markers[j - 1] = tmp;
        }
    return t;
}

void needle_tok_free(needle_tok *t)
{
    if (!t) return;
    free(t->p);
    free(t->hash);
    free(t->markers);
    free(t);
}

int needle_tok_bos(const needle_tok *t) { return t->bos_id; }
int needle_tok_eos(const needle_tok *t) { return t->eos_id; }
int needle_tok_count(const needle_tok *t) { return t->n; }

/* Merge adjacent symbols, highest vocabulary score first (SentencePiece BPE).
 * The symbol arrays are heap-allocated: one entry per UTF-8 character, which
 * for a rendered tool prompt runs to thousands - far past any task stack. */
static int bpe_segment(const needle_tok *t, const char *s, int len,
                       int *out, int max_out, int n_out)
{
    if (len <= 0) return n_out;
    const char **sym = (const char **)malloc((size_t)len * sizeof(char *));
    int *slen = (int *)malloc((size_t)len * sizeof(int));
    if (!sym || !slen) { free(sym); free(slen); return n_out; }
    int n = 0;
    for (int i = 0; i < len; ) {
        int c = (uint8_t)s[i], w = 1;
        if (c >= 0xF0) w = 4; else if (c >= 0xE0) w = 3; else if (c >= 0xC0) w = 2;
        if (i + w > len) w = 1;
        sym[n] = s + i;
        slen[n] = w;
        n++;
        i += w;
    }
    for (;;) {
        float best = 0.0f;
        int best_j = -1;
        for (int j = 0; j + 1 < n; j++) {
            int id = tok_lookup(t, sym[j], slen[j] + slen[j + 1]);
            if (id >= 0 && (best_j < 0 || t->p[id].score > best)) {
                best = t->p[id].score;
                best_j = j;
            }
        }
        if (best_j < 0) break;
        slen[best_j] += slen[best_j + 1];
        for (int j = best_j + 1; j + 1 < n; j++) {
            sym[j] = sym[j + 1];
            slen[j] = slen[j + 1];
        }
        n--;
    }
    for (int j = 0; j < n; j++) {
        int id = tok_lookup(t, sym[j], slen[j]);
        if (id >= 0) {
            if (n_out < max_out) out[n_out] = id;
            n_out++;
        } else if (t->byte_fallback) {
            for (int b = 0; b < slen[j]; b++) {
                int bid = t->byte_id[(uint8_t)sym[j][b]];
                if (n_out < max_out) out[n_out] = bid >= 0 ? bid : t->unk_id;
                n_out++;
            }
        } else {
            if (n_out < max_out) out[n_out] = t->unk_id;
            n_out++;
        }
    }
    free(sym);
    free(slen);
    return n_out;
}

int needle_tok_encode(const needle_tok *t, const char *text, int *out, int max_out)
{
    size_t raw = strlen(text);
    size_t cap = raw * SP_SPACE_LEN + SP_SPACE_LEN + 1;
    char *esc = (char *)malloc(cap);
    if (!esc) return -1;
    size_t e = 0;
    if (t->add_dummy) { memcpy(esc, SP_SPACE, SP_SPACE_LEN); e = SP_SPACE_LEN; }
    for (size_t i = 0; i < raw; i++) {
        if (text[i] == ' ') { memcpy(esc + e, SP_SPACE, SP_SPACE_LEN); e += SP_SPACE_LEN; }
        else esc[e++] = text[i];
    }
    esc[e] = 0;

    int n_out = 0;
    size_t seg = 0;
    for (size_t i = 0; i < e; ) {
        int hit = -1;
        for (int k = 0; k < t->n_markers; k++) {
            const piece_t *p = &t->p[t->markers[k]];
            if (e - i >= p->len && memcmp(esc + i, p->s, p->len) == 0) {
                hit = t->markers[k];
                break;
            }
        }
        if (hit < 0) { i++; continue; }
        if (i > seg) n_out = bpe_segment(t, esc + seg, (int)(i - seg), out, max_out, n_out);
        if (n_out < max_out) out[n_out] = hit;
        n_out++;
        i += t->p[hit].len;
        seg = i;
    }
    if (e > seg) n_out = bpe_segment(t, esc + seg, (int)(e - seg), out, max_out, n_out);
    free(esc);
    return n_out;
}

int needle_tok_piece(const needle_tok *t, int id, char *out, int max_out)
{
    if (id < 0 || id >= t->n) return 0;
    const piece_t *p = &t->p[id];
    if (p->type == TK_CONTROL || p->type == TK_UNKNOWN) return 0;
    if (p->type == TK_BYTE) {
        for (int b = 0; b < 256; b++)
            if (t->byte_id[b] == id) {
                if (max_out < 1) return 0;
                out[0] = (char)b;
                return 1;
            }
        return 0;
    }
    /* replace the SentencePiece space meta with a real space */
    int n = 0;
    for (int i = 0; i < p->len; ) {
        if (p->len - i >= SP_SPACE_LEN && memcmp(p->s + i, SP_SPACE, SP_SPACE_LEN) == 0) {
            if (n < max_out) out[n] = ' ';
            n++;
            i += SP_SPACE_LEN;
        } else {
            if (n < max_out) out[n] = p->s[i];
            n++;
            i++;
        }
    }
    return n;
}
