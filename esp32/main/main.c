/* Needle-2 on ESP32-S3-BOX-3.
 *
 * Attach a console:  . ~/esp/esp-idf/export.sh
 *                    idf.py -p /dev/cu.usbmodem1101 monitor     (Ctrl+] exits)
 * or with no IDF:    screen /dev/cu.usbmodem1101 115200         (Ctrl+A K quits)
 *
 * Memory plan (see ../../DESIGN.md):
 *   hot weights  10.3 MB  flash partition -> PSRAM  (streamed every token)
 *   engram tables 2.1 MB  stays mmap'd in flash     (4 rows read per token)
 *   KV cache      3.6 MB  PSRAM
 *   activations   135 KB  internal SRAM
 *
 * Flash the model with:
 *   parttool.py --port /dev/cu.usbmodem* write_partition \
 *       --partition-name model --input ../build/needle.nsp
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_heap_caps.h"
#include "esp_partition.h"
#include "esp_timer.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_chip_info.h"
#include "esp_memory_utils.h"
#include "driver/usb_serial_jtag.h"
#if __has_include("driver/usb_serial_jtag_vfs.h")
#  include "driver/usb_serial_jtag_vfs.h"
#  define JTAG_VFS_USE_DRIVER   usb_serial_jtag_vfs_use_driver
#  define JTAG_VFS_RX_LE        usb_serial_jtag_vfs_set_rx_line_endings
#  define JTAG_VFS_TX_LE        usb_serial_jtag_vfs_set_tx_line_endings
#else
#  include "esp_vfs_dev.h"            /* esp_vfs_usb_serial_jtag_use_driver */
#  include "esp_vfs_usb_serial_jtag.h" /* ..._set_rx/tx_line_endings */
#  define JTAG_VFS_USE_DRIVER   esp_vfs_usb_serial_jtag_use_driver
#  define JTAG_VFS_RX_LE        esp_vfs_dev_usb_serial_jtag_set_rx_line_endings
#  define JTAG_VFS_TX_LE        esp_vfs_dev_usb_serial_jtag_set_tx_line_endings
#endif

/* Blocking line input over USB-Serial-JTAG. Without the driver behind the VFS,
 * fgets() returns immediately with nothing and the prompt is dead. Terminals
 * send CR, so translate it or a line never terminates. */
static void console_init(void)
{
    usb_serial_jtag_driver_config_t cfg = USB_SERIAL_JTAG_DRIVER_CONFIG_DEFAULT();
    cfg.rx_buffer_size = 1024;
    cfg.tx_buffer_size = 1024;
    if (usb_serial_jtag_driver_install(&cfg) != ESP_OK) return;
    JTAG_VFS_USE_DRIVER();
    JTAG_VFS_RX_LE(ESP_LINE_ENDINGS_CR);
    JTAG_VFS_TX_LE(ESP_LINE_ENDINGS_CRLF);
    setvbuf(stdin, NULL, _IONBF, 0);
}

#include "needle.h"
#include "needle_cq.h"

static const char *TAG = "needle";

static long long now_us(void) { return (long long)esp_timer_get_time(); }

#define LOAD_CHUNK 65536

/* ---- allocators -------------------------------------------------------- */
static void *alloc_psram(size_t n, void *user)
{
    (void)user;
    return heap_caps_malloc(n, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
}

static void *alloc_internal(size_t n, void *user)
{
    (void)user;
    void *p = heap_caps_malloc(n, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    return p ? p : heap_caps_malloc(n, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
}

/* ---- second core ------------------------------------------------------- */
static TaskHandle_t s_worker, s_owner;
static nq_work_fn   s_work;
static void        *s_arg;
static uint32_t     s_lo, s_hi;

static void worker_task(void *unused)
{
    (void)unused;
    for (;;) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        s_work(s_arg, s_lo, s_hi);
        xTaskNotifyGive(s_owner);
    }
}

/* Rows are independent, so any split is bit-identical to running inline. */
static void shard_dual_core(nq_work_fn work, void *arg, uint32_t lo, uint32_t hi)
{
    uint32_t mid = lo + (hi - lo) / 2;
    s_work = work;
    s_arg = arg;
    s_lo = mid;
    s_hi = hi;
    xTaskNotifyGive(s_worker);
    work(arg, lo, mid);
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
}

/* ---- model loading ----------------------------------------------------- */
typedef struct {
    needle_model model;
    void        *hot;
    const void  *mapped;
    esp_partition_mmap_handle_t map_handle;
    uint8_t     *head;      /* header + directory, in internal RAM */
} loaded_t;

static int load_model(loaded_t *out)
{
    const esp_partition_t *part = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x40, "model");
    if (!part) {
        ESP_LOGE(TAG, "no 'model' partition");
        return -1;
    }

    nsp_header hdr;
    if (esp_partition_read(part, 0, &hdr, sizeof hdr) != ESP_OK) return -2;
    if (hdr.magic != NSP_MAGIC || hdr.version != NSP_VERSION) {
        ESP_LOGE(TAG, "bad model image (magic %08x) - was it flashed?",
                 (unsigned)hdr.magic);
        return -3;
    }

    size_t head_bytes = (size_t)(hdr.dir_off + hdr.dir_size);
    out->head = heap_caps_malloc(head_bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!out->head) return -4;
    if (esp_partition_read(part, 0, out->head, head_bytes) != ESP_OK) return -5;

    ESP_LOGI(TAG, "model: %u layers, d=%u, vocab=%u, window=%u, %u tensors",
             (unsigned)hdr.n_layers, (unsigned)hdr.d_model, (unsigned)hdr.vocab,
             (unsigned)hdr.kv_window, (unsigned)hdr.n_tensors);
    ESP_LOGI(TAG, "hot %.2f MB -> PSRAM, cold %.2f MB stays in flash",
             hdr.hot_size / 1048576.0, hdr.cold_size / 1048576.0);

    out->hot = heap_caps_malloc((size_t)hdr.hot_size,
                                MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!out->hot) {
        ESP_LOGE(TAG, "could not reserve %.2f MB of PSRAM for weights "
                      "(largest free block %u KB)",
                 hdr.hot_size / 1048576.0,
                 (unsigned)(heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM) / 1024));
        return -6;
    }

    int64_t t0 = esp_timer_get_time();
    for (size_t off = 0; off < hdr.hot_size; off += LOAD_CHUNK) {
        size_t n = hdr.hot_size - off;
        if (n > LOAD_CHUNK) n = LOAD_CHUNK;
        if (esp_partition_read(part, (size_t)hdr.hot_off + off,
                               (uint8_t *)out->hot + off, n) != ESP_OK) return -7;
        if ((off & 0xFFFFF) == 0) vTaskDelay(1);   /* keep the watchdog happy */
    }
    double dt = (esp_timer_get_time() - t0) / 1e6;
    ESP_LOGI(TAG, "weights -> PSRAM in %.2f s (%.1f MB/s)",
             dt, hdr.hot_size / 1048576.0 / dt);

    /* cold section and tokenizer are read straight from flash */
    size_t map_off = (size_t)hdr.cold_off;
    size_t map_len = (size_t)(hdr.tok_off + hdr.tok_size - hdr.cold_off);
    if (esp_partition_mmap(part, map_off, map_len, ESP_PARTITION_MMAP_DATA,
                           &out->mapped, &out->map_handle) != ESP_OK) {
        ESP_LOGE(TAG, "mmap of the cold section failed");
        return -8;
    }

    const uint8_t *base = (const uint8_t *)out->mapped;
    needle_model_open_split(&out->model, out->head, out->hot, base,
                            base + (hdr.tok_off - hdr.cold_off),
                            (size_t)hdr.tok_size);
    return 0;
}

/* ---- what actually limits this chip ------------------------------------ */
/* One 32-bit read per 32-byte cache line, i.e. line-fill bandwidth, which is
 * what the matvec sees when it streams weights. */
static double stream_mbps(const void *p, size_t bytes)
{
    const volatile uint32_t *q = (const volatile uint32_t *)p;
    size_t words = bytes / 4;
    int64_t t0 = esp_timer_get_time();
    uint32_t sink = 0;
    for (size_t i = 0; i < words; i += 8) sink += q[i];
    int64_t dt = esp_timer_get_time() - t0;
    __asm__ __volatile__("" :: "r"(sink));
    return bytes / 1048576.0 / (dt / 1e6);
}

static void report_bandwidth(const loaded_t *L)
{
    const nsp_header *h = L->model.hdr;
    size_t probe = 4 * 1024 * 1024;
    if (probe > h->hot_size) probe = (size_t)h->hot_size;
    double psram = stream_mbps(L->hot, probe);
    double flash = stream_mbps(L->mapped, 1024 * 1024);

    /* weights streamed per token, straight out of the directory */
    double per_token = 0;
    for (uint32_t i = 0; i < h->n_tensors; i++) {
        const nsp_rec *r = &L->model.dir[i];
        if (r->section == NSP_SEC_COLD) continue;     /* gathered, not streamed */
        per_token += r->nbytes;
    }
    ESP_LOGI(TAG, "PSRAM stream %.1f MB/s, flash stream %.1f MB/s", psram, flash);
    ESP_LOGI(TAG, "weights touched per token: %.2f MB -> %.0f ms at PSRAM speed",
             per_token / 1048576.0, per_token / 1048576.0 / psram * 1000.0);
}

/* ---- reporting --------------------------------------------------------- */
static void report_memory(const char *when)
{
    ESP_LOGI(TAG, "%s: internal free %u KB (largest %u KB), "
                  "PSRAM free %u KB (largest %u KB)",
             when,
             (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024),
             (unsigned)(heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL) / 1024),
             (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024),
             (unsigned)(heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM) / 1024));
}

static int argmax_f(const float *v, int n)
{
    int b = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[b]) b = i;
    return b;
}

/* ---- one turn ---------------------------------------------------------- */
static void run_turn(needle_ctx *ctx, needle_tok *tk, const nsp_header *h,
                     const char *tools_json, const char *query, int max_new)
{
    static int ids[1024];
    static char rendered[3072];

    snprintf(rendered, sizeof rendered,
             "<|im_start|>user\n<tools>%s</tools>\n%s<|im_end|>\n"
             "<|im_start|>assistant\n", tools_json, query);

    needle_reset(ctx);
    needle_profile_reset();
    ids[0] = needle_tok_bos(tk);
    int n = 1 + needle_tok_encode(tk, rendered, ids + 1,
                                  (int)(sizeof ids / sizeof ids[0]) - 1);

    /* Prefill is ~0.7 s/token and prints nothing on its own, so a 116-token
     * prompt looks hung for over a minute. Show progress. */
    printf("reading %d tokens", n);
    fflush(stdout);
    int64_t t0 = esp_timer_get_time();
    const float *lg = NULL;
    for (int i = 0; i < n; i++) {
        if (i % 8 == 0) { putchar('.'); fflush(stdout); }
        lg = needle_step(ctx, ids[i]);
        /* One tick per token so IDLE0 runs. This app deliberately pegs both
         * cores; without this the task watchdog fires every 30 s and buries
         * the output. 1 ms against ~700 ms is free. */
        vTaskDelay(1);
    }
    double pre = (esp_timer_get_time() - t0) / 1e6;
    printf(" %.0fs\n", pre);

    t0 = esp_timer_get_time();
    int made = 0;
    char piece[64];
    printf("\n");
    for (int i = 0; i < max_new; i++) {
        int nxt = argmax_f(lg, (int)h->vocab);
        if (nxt == needle_tok_eos(tk)) break;
        int w = needle_tok_piece(tk, nxt, piece, sizeof piece - 1);
        piece[w] = 0;
        fputs(piece, stdout);
        fflush(stdout);
        made++;
        lg = needle_step(ctx, nxt);
        vTaskDelay(1);
    }
    double dec = (esp_timer_get_time() - t0) / 1e6;
    printf("\n\n[prefill %d tok in %.2fs = %.2f tok/s | decode %d tok in %.2fs "
           "= %.2f tok/s]\n",
           n, pre, n / pre, made, dec, made / (dec > 0 ? dec : 1e-9));

    int steps = n + made;
    double total = needle_profile[NEEDLE_P_TOTAL];
    printf("per-token breakdown over %d steps (%.1f ms/token):\n",
           steps, total / 1000.0 / steps);
    double named = 0;
    for (int i = 0; i < NEEDLE_P_TOTAL; i++) {
        named += needle_profile[i];
        printf("  %-22s %7.1f ms  %5.1f%%\n", needle_phase_name[i],
               needle_profile[i] / 1000.0 / steps,
               100.0 * needle_profile[i] / total);
    }
    printf("  %-22s %7.1f ms  %5.1f%%\n", "everything else",
           (total - named) / 1000.0 / steps, 100.0 * (total - named) / total);
    needle_profile_reset();
}

static const char *DEMO_TOOLS =
    "[{\"name\":\"set_lights\",\"description\":\"Turn a room's lights on or off "
    "and set brightness\",\"parameters\":{\"type\":\"object\",\"properties\":"
    "{\"room\":{\"type\":\"string\"},\"on\":{\"type\":\"boolean\"},"
    "\"brightness\":{\"type\":\"integer\",\"description\":\"0 to 100\"}},"
    "\"required\":[\"room\",\"on\"]}},"
    "{\"name\":\"set_timer\",\"description\":\"Start a countdown timer\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{\"minutes\":"
    "{\"type\":\"integer\"}},\"required\":[\"minutes\"]}}]";

void app_main(void)
{
    esp_chip_info_t chip;
    esp_chip_info(&chip);
    ESP_LOGI(TAG, "ESP32-S3 rev %d, %d cores, %u KB PSRAM",
             chip.revision, chip.cores,
             (unsigned)(heap_caps_get_total_size(MALLOC_CAP_SPIRAM) / 1024));
    report_memory("boot");

    static loaded_t L;
    int rc = load_model(&L);
    if (rc != 0) {
        ESP_LOGE(TAG, "load_model failed (%d)", rc);
        return;
    }

    needle_tok *tk = needle_tok_open(L.model.tok, L.model.tok_size);
    if (!tk) { ESP_LOGE(TAG, "tokenizer failed"); return; }

    needle_config cfg = {0};
    cfg.alloc_big = alloc_psram;
    cfg.alloc_fast = alloc_internal;
    needle_ctx *ctx = needle_create(&L.model, &cfg);
    if (!ctx) { ESP_LOGE(TAG, "needle_create failed"); return; }

    ESP_LOGI(TAG, "subset-sum tables (64 KB, the hottest structure) are in %s",
             esp_ptr_internal(needle_sub_tables(ctx)) ? "INTERNAL SRAM" : "PSRAM <-- slow");
    ESP_LOGI(TAG, "state: %.2f MB KV (PSRAM) + %u KB scratch (internal)",
             needle_bytes_big(ctx) / 1048576.0,
             (unsigned)(needle_bytes_fast(ctx) / 1024));
    report_memory("ready");
    report_bandwidth(&L);

    needle_now_us = now_us;
    s_owner = xTaskGetCurrentTaskHandle();
    if (chip.cores > 1 &&
        xTaskCreatePinnedToCore(worker_task, "needle_w", 4096, NULL,
                                configMAX_PRIORITIES - 3, &s_worker, 1) == pdPASS) {
        nq_shard = shard_dual_core;
        ESP_LOGI(TAG, "row sharding across both cores: on");
    }

    printf("\n> dim the living room to 30\n");
    run_turn(ctx, tk, L.model.hdr, DEMO_TOOLS, "dim the living room to 30", 160);

    console_init();
    printf("\nReady. Type a request and press Enter (~25 s per answer).\n"
           "Tools: set_lights(room, on, brightness) | set_timer(minutes)\n");

    report_memory("after decode");

    /* line-oriented REPL on the console */
    static char line[512];
    for (;;) {
        printf("\n> ");
        fflush(stdout);
        if (!fgets(line, sizeof line, stdin)) { vTaskDelay(pdMS_TO_TICKS(100)); continue; }
        size_t n = strlen(line);
        while (n && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = 0;
        if (!n) continue;
        run_turn(ctx, tk, L.model.hdr, DEMO_TOOLS, line, 160);
    }
}
