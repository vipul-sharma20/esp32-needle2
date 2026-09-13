/* Needle-2 inference for ESP32-S3 (and any host, for testing).
 *
 * Single conversation per context. Weights are read-only and shareable.
 */
#ifndef NEEDLE_H
#define NEEDLE_H

#include <stddef.h>
#include <stdint.h>

#include "nsp.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const nsp_header *hdr;
    const nsp_rec    *dir;
    const uint8_t    *hot;   /* base of the hot section  */
    const uint8_t    *cold;  /* base of the cold section */
    const uint8_t    *tok;   /* tokenizer blob           */
    size_t            tok_size;
} needle_model;

/* Bind a model to an in-memory .nsp image (the whole file). */
int needle_model_open(needle_model *m, const void *image, size_t size);

/* Bind a model whose sections live at separate addresses (ESP32: hot in
 * PSRAM, cold and tokenizer mapped from flash). `image` must still point at
 * the header + directory. */
int needle_model_open_split(needle_model *m, const void *header_and_dir,
                            const void *hot, const void *cold,
                            const void *tok, size_t tok_size);

typedef void *(*needle_alloc_fn)(size_t bytes, void *user);

typedef struct {
    int             kv_window;   /* 0 -> the value baked into the model */
    int             sink_slots;  /* KV slots pinned at the front (tool sinks) */
    needle_alloc_fn alloc_big;   /* KV cache; PSRAM on ESP32. NULL -> malloc */
    needle_alloc_fn alloc_fast;  /* activations; internal RAM. NULL -> malloc */
    void           *user;
    int             row_lo, row_hi; /* row shard, for splitting across cores  */
} needle_config;

typedef struct needle_ctx needle_ctx;

needle_ctx *needle_create(const needle_model *m, const needle_config *cfg);
void        needle_free(needle_ctx *c);
void        needle_reset(needle_ctx *c);

/* Freeze everything decoded so far as always-attended KV sinks. */
void needle_pin_sinks(needle_ctx *c);

/* Advance one token; returns logits[vocab], valid until the next call. */
const float *needle_step(needle_ctx *c, int token);

int needle_pos(const needle_ctx *c);
size_t needle_bytes_fast(const needle_ctx *c);
/* Pointer to the subset-sum tables, so a host can check where they landed. */
const void *needle_sub_tables(const needle_ctx *c);

/* Per-phase profiling. Set needle_now_us to a microsecond clock to enable;
 * needle_profile[] then accumulates microseconds per phase. */
enum { NEEDLE_P_PREP = 0, NEEDLE_P_MV2, NEEDLE_P_MV4, NEEDLE_P_ATTN,
       NEEDLE_P_ENGRAM, NEEDLE_P_HADA, NEEDLE_P_SINK, NEEDLE_P_NORM,
       NEEDLE_P_LANE, NEEDLE_P_TOTAL, NEEDLE_P_COUNT };
extern long long (*needle_now_us)(void);
extern double     needle_profile[NEEDLE_P_COUNT];
void needle_profile_reset(void);
extern const char *const needle_phase_name[NEEDLE_P_COUNT];
size_t needle_bytes_big(const needle_ctx *c);

/* ---- tokenizer ------------------------------------------------------- */
typedef struct needle_tok needle_tok;

needle_tok *needle_tok_open(const uint8_t *blob, size_t size);
void        needle_tok_free(needle_tok *t);
int         needle_tok_encode(const needle_tok *t, const char *text,
                              int *out, int max_out);
/* Append the text for one id; returns bytes written (0 for control ids). */
int         needle_tok_piece(const needle_tok *t, int id, char *out, int max_out);
int         needle_tok_bos(const needle_tok *t);
int         needle_tok_eos(const needle_tok *t);
int         needle_tok_count(const needle_tok *t);

#ifdef __cplusplus
}
#endif
#endif /* NEEDLE_H */
