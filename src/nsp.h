/* .nsp - Needle weights repacked for ESP32-S3.
 *
 * Holds the published model's CQ indices and fp16 group norms verbatim,
 * re-sectioned so the loader can make exactly two decisions:
 *
 *   HOT   every tensor streamed on every token  -> copy to PSRAM
 *   COLD  the two engram tables, read four rows at a time -> leave in flash
 *
 * Layout:
 *   header (nsp_header, 64-byte aligned)
 *   directory: n_tensors * nsp_rec, in the canonical order below
 *   hot section   (64-byte aligned)
 *   cold section  (64-byte aligned)
 *   tokenizer blob
 *
 * Canonical tensor order (matches .cact minus the probe heads):
 *   embedding
 *   per layer: norm_in q k v q_norm k_norm gate out post_norm attn_gate
 *              pre_hada d1 d2 d3
 *   mhc_a_pre mhc_a_post mhc_a_res mhc_b_pre mhc_b_post mhc_b_res
 *   mhc_phi_pre mhc_phi_post mhc_phi_res
 *   per engram site: tables key_proj value_proj taps
 *   final_norm
 */
#ifndef NSP_H
#define NSP_H

#include <stdint.h>

#define NSP_MAGIC   0x3150534EU /* "NSP1" */
#define NSP_VERSION 1
#define NSP_ALIGN   64
#define NSP_GROUP   128
#define NSP_MAX_GROUPS 32   /* in_pad <= 4096 */
#define NSP_SUB_GROUPS 4    /* groups for which subset-sum tables are built */
#define NSP_SUB_FLOATS (NSP_SUB_GROUPS * 16 * 256)   /* 64 KB of fp32 */

enum {
    NSP_KIND_F16  = 0,
    NSP_KIND_CQ   = 1,  /* row-major LSB-first indices, then fp16 group norms */
    NSP_KIND_CQ2P = 2,  /* 2-bit only: group-major bit planes, then norms.
                         * Per group, per row: 16 bytes of index-bit-1 then 16
                         * of index-bit-0, LSB-first. Same bytes, arranged so
                         * the subset-sum kernel can stream them. */
};
enum { NSP_SEC_HOT = 0, NSP_SEC_COLD = 1 };

/* per-layer slot numbers, relative to the layer's base index */
enum {
    NSP_L_NORM_IN = 0, NSP_L_Q, NSP_L_K, NSP_L_V, NSP_L_QNORM, NSP_L_KNORM,
    NSP_L_GATE, NSP_L_OUT, NSP_L_POST, NSP_L_ATTN_GATE, NSP_L_PRE_HADA,
    NSP_L_D1, NSP_L_D2, NSP_L_D3, NSP_L_COUNT
};
/* engram slot numbers, relative to the site's base index */
enum { NSP_E_TABLES = 0, NSP_E_KEY, NSP_E_VALUE, NSP_E_TAPS, NSP_E_COUNT };

typedef struct {
    uint32_t off;     /* byte offset within its section */
    uint32_t nbytes;
    uint32_t out;     /* rows */
    uint16_t in;      /* columns (reduction axis); 1 for vectors */
    uint8_t  kind;    /* NSP_KIND_* */
    uint8_t  bits;    /* 2/3/4 for CQ, 0 for fp16 */
    uint8_t  section; /* NSP_SEC_* */
    uint8_t  _pad[3];
} nsp_rec;            /* 20 bytes */

typedef struct {
    uint32_t magic, version, flags, n_tensors;

    uint32_t vocab, d_model, n_heads, n_kv_heads, n_layers, head_dim, max_seq;
    uint32_t lanes, kv_window, kv_bits;
    uint32_t engram_slots, engram_sub_dim, engram_tables, engram_taps,
             engram_dilation;
    uint32_t n_orders, orders[4];
    uint32_t n_sites, sites[4];
    float    rope_theta;

    uint64_t dir_off,  dir_size;
    uint64_t hot_off,  hot_size;
    uint64_t cold_off, cold_size;
    uint64_t tok_off,  tok_size;

    uint32_t n_cb;          /* 28: cb2[4] | cb3[8] | cb4[16] */
    float    cb[28];        /* the shipped fp32 codebooks */
    int8_t   cb_i8[28];     /* int8 rescalings, reserved; unused by this runtime */
    float    cb_scale[5];   /* their scales, likewise reserved */
    uint32_t _tail[3];
} nsp_header;

/* directory index helpers -------------------------------------------------- */
#define NSP_T_EMBEDDING             0
#define NSP_T_LAYER(h, l, slot)     (1u + (uint32_t)(l) * NSP_L_COUNT + (slot))
#define NSP_T_MHC(h)                (1u + (h)->n_layers * NSP_L_COUNT)
enum {
    NSP_M_A_PRE = 0, NSP_M_A_POST, NSP_M_A_RES, NSP_M_B_PRE, NSP_M_B_POST,
    NSP_M_B_RES, NSP_M_PHI_PRE, NSP_M_PHI_POST, NSP_M_PHI_RES, NSP_M_COUNT
};
#define NSP_T_ENGRAM(h, s, slot)    (NSP_T_MHC(h) + NSP_M_COUNT + \
                                     (uint32_t)(s) * NSP_E_COUNT + (slot))
#define NSP_T_FINAL_NORM(h)         (NSP_T_ENGRAM(h, (h)->n_sites, 0))

#endif /* NSP_H */
