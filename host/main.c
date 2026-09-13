/* Host driver: runs the exact same engine the ESP32 runs, so tools and prompts
 * can be iterated on - and the numerics checked with tools/verify.py - without
 * flashing anything.
 *
 *   needle_host model.nsp --prompt "..." [--tools tools.json] [-n N]
 *   needle_host model.nsp --dump-logits tokens.txt out.f32
 */
#include "needle.h"
#include "needle_cq.h"
#include <pthread.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>
#include <unistd.h>

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void *map_file(const char *path, size_t *size)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return NULL; }
    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) return NULL;
    *size = (size_t)st.st_size;
    return p;
}

static char *read_text(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *b = (char *)malloc((size_t)n + 1);
    if (b && fread(b, 1, (size_t)n, f) != (size_t)n) { free(b); b = NULL; }
    if (b) {
        b[n] = 0;
        while (n > 0 && (b[n - 1] == '\n' || b[n - 1] == '\r')) b[--n] = 0;
    }
    fclose(f);
    return b;
}


/* --- row sharding ------------------------------------------------------- */
/* Sequential three-way split: proves rows are independent and the contract
 * holds, without any threading in the picture. */
static void shard_seq3(nq_work_fn work, void *arg, uint32_t lo, uint32_t hi)
{
    uint32_t a = lo + (hi - lo) / 3, b = lo + 2 * (hi - lo) / 3;
    work(arg, lo, a);
    work(arg, a, b);
    work(arg, b, hi);
}

typedef struct { nq_work_fn work; void *arg; uint32_t lo, hi; } job_t;
static void *run_job(void *p){ job_t *j = (job_t *)p; j->work(j->arg, j->lo, j->hi); return 0; }

static void shard_2threads(nq_work_fn work, void *arg, uint32_t lo, uint32_t hi)
{
    uint32_t mid = lo + (hi - lo) / 2;
    job_t j = { work, arg, mid, hi };
    pthread_t th;
    pthread_create(&th, NULL, run_job, &j);
    work(arg, lo, mid);
    pthread_join(th, NULL);
}

/* Needle is trained on compact JSON (separators=(",",":")). Pretty-printed
 * schemas re-tokenise and the model starts inventing tool names, so strip
 * whitespace outside string literals before rendering the prompt. */
static void json_minify(char *s)
{
    char *w = s;
    int in_str = 0, esc = 0;
    for (char *r = s; *r; r++) {
        if (in_str) {
            *w++ = *r;
            if (esc) esc = 0;
            else if (*r == '\\') esc = 1;
            else if (*r == '"') in_str = 0;
            continue;
        }
        if (*r == '"') { in_str = 1; *w++ = *r; continue; }
        if (*r == ' ' || *r == '\t' || *r == '\n' || *r == '\r') continue;
        *w++ = *r;
    }
    *w = 0;
}

static int argmax(const float *v, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[best]) best = i;
    return best;
}

int main(int argc, char **argv)
{
    const char *model_path = NULL, *prompt = NULL, *tools_path = NULL;
    const char *dump_tokens = NULL, *dump_out = NULL;
    int max_new = 128, raw = 0, verbose = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "--tools") && i + 1 < argc) tools_path = argv[++i];
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) max_new = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--raw")) raw = 1;
        else if (!strcmp(argv[i], "-v")) verbose = 1;
        else if (!strcmp(argv[i], "--shard-test")) nq_shard = shard_seq3;
        else if (!strcmp(argv[i], "--threads2")) nq_shard = shard_2threads;
        else if (!strcmp(argv[i], "--dump-logits") && i + 2 < argc) {
            dump_tokens = argv[++i];
            dump_out = argv[++i];
        } else if (!model_path) model_path = argv[i];
    }
    if (!model_path) {
        fprintf(stderr, "usage: needle_host model.nsp --prompt \"...\" "
                        "[--tools t.json] [-n N] [--raw]\n");
        return 2;
    }

    size_t size = 0;
    void *image = map_file(model_path, &size);
    if (!image) { perror(model_path); return 1; }

    needle_model m;
    int rc = needle_model_open(&m, image, size);
    if (rc != 0) { fprintf(stderr, "bad model (%d)\n", rc); return 1; }

    needle_tok *tk = needle_tok_open(m.tok, m.tok_size);
    if (!tk) { fprintf(stderr, "bad tokenizer\n"); return 1; }

    needle_config cfg = {0};
    needle_ctx *ctx = needle_create(&m, &cfg);
    if (!ctx) { fprintf(stderr, "out of memory\n"); return 1; }

    fprintf(stderr, "model %s  %u layers  d=%u  vocab=%u  window=%u\n",
            model_path, m.hdr->n_layers, m.hdr->d_model, m.hdr->vocab,
            m.hdr->kv_window);
    fprintf(stderr, "state: %.2f MB kv + %.0f KB scratch\n",
            needle_bytes_big(ctx) / 1048576.0, needle_bytes_fast(ctx) / 1024.0);

    /* --- logits dump mode: one line of space-separated token ids ---------- */
    if (dump_tokens) {
        char *txt = read_text(dump_tokens);
        if (!txt) { perror(dump_tokens); return 1; }
        FILE *out = fopen(dump_out, "wb");
        if (!out) { perror(dump_out); return 1; }
        char *save = NULL;
        for (char *t = strtok_r(txt, " \t\n", &save); t;
             t = strtok_r(NULL, " \t\n", &save)) {
            const float *lg = needle_step(ctx, atoi(t));
            fwrite(lg, sizeof(float), m.hdr->vocab, out);
        }
        fclose(out);
        fprintf(stderr, "wrote %s\n", dump_out);
        return 0;
    }

    if (!prompt) prompt = "dim the living room to 30";

    /* --- build the prompt ------------------------------------------------- */
    char *full;
    if (raw) {
        full = strdup(prompt);
    } else {
        char *tools = tools_path ? read_text(tools_path) : strdup("[]");
        if (!tools) { perror(tools_path); return 1; }
        json_minify(tools);
        size_t n = strlen(prompt) + strlen(tools) + 128;
        full = (char *)malloc(n);
        snprintf(full, n,
                 "<|im_start|>user\n<tools>%s</tools>\n%s<|im_end|>\n"
                 "<|im_start|>assistant\n", tools, prompt);
        free(tools);
    }

    int *ids = (int *)malloc(sizeof(int) * (size_t)m.hdr->max_seq);
    ids[0] = needle_tok_bos(tk);
    int n_ids = 1 + needle_tok_encode(tk, full, ids + 1, (int)m.hdr->max_seq - 1);
    if (verbose) fprintf(stderr, "prompt: %d tokens\n", n_ids);

    double t0 = now();
    const float *lg = NULL;
    for (int i = 0; i < n_ids; i++) lg = needle_step(ctx, ids[i]);
    double t_pre = now() - t0;
    fprintf(stderr, "prefill %d tok in %.2fs (%.1f tok/s)\n",
            n_ids, t_pre, n_ids / t_pre);

    t0 = now();
    int n_out = 0;
    char buf[64];
    for (int i = 0; i < max_new; i++) {
        int nxt = argmax(lg, (int)m.hdr->vocab);
        if (nxt == needle_tok_eos(tk)) break;
        int w = needle_tok_piece(tk, nxt, buf, sizeof buf);
        fwrite(buf, 1, (size_t)w, stdout);
        fflush(stdout);
        n_out++;
        lg = needle_step(ctx, nxt);
    }
    double t_dec = now() - t0;
    printf("\n");
    fprintf(stderr, "decode %d tok in %.2fs (%.2f tok/s)\n",
            n_out, t_dec, n_out / (t_dec > 0 ? t_dec : 1e-9));

    needle_free(ctx);
    needle_tok_free(tk);
    munmap(image, size);
    return 0;
}
