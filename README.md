# Needle-2 on the ESP32-S3-BOX-3

A C99 inference engine for [Cactus-Compute/needle-2](https://huggingface.co/Cactus-Compute/needle-2),
a 45M-parameter tool-calling model, running on an ESP32-S3-BOX-3:
16 MB flash, 16 MB octal PSRAM, dual Xtensa LX7 at 240 MHz.

Text goes in and a JSON tool call comes out, computed on the microcontroller
with no network access and no runtime dependencies.

```
> turn the bedroom lights on at 75
<tool_call>[{"name":"set_lights","arguments":{"room":"bedroom","brightness":75}}]</tool_call>
```

https://github.com/user-attachments/assets/d9cbcb73-4fb4-48f3-ac4a-e2637ef56c78

The board answering `dim the living room to 30`. The 78-second prefill is cut,
so what plays is decode at 1.28 tok/s.

## Quick start

```sh
tools/fetch_model.sh                                   # 13 MB from Hugging Face
python3 tools/pack_esp.py model/needle2.cact build/needle.nsp
```

That leaves a packed model at `build/needle.nsp`, ready to write to the board.

## On the board

```sh
cd esp32
. ~/esp/esp-idf/export.sh          # ESP-IDF v5.2 or newer
idf.py set-target esp32s3
idf.py -p /dev/cu.usbmodem1101 flash

# the weights, once (about two minutes)
parttool.py -p /dev/cu.usbmodem1101 write_partition \
    --partition-name model --input ../build/needle.nsp

idf.py -p /dev/cu.usbmodem1101 monitor      # Ctrl+] to exit
```

Talking to a flashed board needs no ESP-IDF: `screen /dev/cu.usbmodem1101 115200`
is enough (`Ctrl+A K` to quit).

The board prints its memory and bandwidth figures, runs one demo turn, then
gives you a prompt. Answers take about 25 s.

Two build settings are required, and `sdkconfig.defaults` already carries them:
`CONFIG_SPIRAM_MODE_OCT` and `CONFIG_SPIRAM_SPEED_80M`. The BOX-3's PSRAM is
octal; in quad mode it fails silently at boot. The USB-C port is wired to the
SoC's own USB, so the console is `CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG` and the
port enumerates as `/dev/cu.usbmodem*`.

## Defining tools

`DEMO_TOOLS` in `esp32/main/main.c` carries the board's schema. Standard JSON
schema:

```json
[{"name":"set_lights",
  "description":"Turn a room's lights on or off and set brightness",
  "parameters":{"type":"object",
    "properties":{"room":{"type":"string"},"on":{"type":"boolean"},
                  "brightness":{"type":"integer","description":"0 to 100"}},
    "required":["room","on"]}}]
```

Keep it **compact**. The model is trained on `json.dumps(separators=(",",":"))`;
a pretty-printed schema re-tokenises and the model starts inventing tool names.
A schema compiled into firmware has no minifier in front of it, so write it
compact.

A request no declared tool can serve comes back as the empty call `[]`. The
model is trained to emit that, so handle it as an answer rather than an error.

## Measured on hardware

ESP32-S3 rev 2, 16 MB octal PSRAM at 80 MHz, both cores:

```
weights -> PSRAM in 1.20 s
state:  3.59 MB KV cache (PSRAM) + 206 KB scratch (internal), 1931 KB PSRAM spare
PSRAM stream 60.3 MB/s, flash stream 25.8 MB/s
691 ms/token  ->  1.49 tok/s prefill, 1.28 tok/s decode
```

Per-token, with the phase breakdown the firmware prints after every turn:

| phase | ms | |
| --- | ---: | ---: |
| matvec CQ2 | 266.2 | 38.5% |
| attention core | 116.8 | 16.9% |
| matvec CQ4 | 42.1 | 6.1% |
| Hadamard MLP | 41.3 | 6.0% |
| prep (rotate + subset sums) | 38.5 | 5.6% |
| Sinkhorn | 23.7 | 3.4% |
| engram | 12.6 | 1.8% |
| lane mix | 9.7 | 1.4% |
| norms | 5.8 | 0.8% |
| unattributed | 134.8 | 19.5% |

Budgets:

| | |
| --- | --- |
| flash | 12.61 MB model + 0.29 MB app, of 16 MB |
| PSRAM | 10.33 MB weights + 3.59 MB KV cache |
| internal SRAM | 206 KB, 64 KB of it the subset-sum tables |


<details>
<summary><b>Running the same engine on a host</b></summary>

`host/` builds the engine for macOS or Linux, which is how you iterate on tools
and prompts without flashing, and how `tools/verify.py` drives the C code:

```sh
make -C host
./host/needle_host build/needle.nsp --tools tools.json --prompt "dim the living room to 30"
```

`--tools` takes the same schema as `DEMO_TOOLS` and is minified before it
reaches the model, so `tools.json` can stay readable. `-n N` caps the generated
tokens, `--raw` skips the chat template, `--threads2` shards rows across two
threads, and `--dump-logits ids.txt out.f32` writes raw logits.

</details>

## Verification

`tools/ref_needle.py` is an independent NumPy implementation of the same model.
`tools/verify.py` runs a prompt through both and diffs the logits:

```sh
make -C host
python3 tools/verify.py build/needle.nsp model/needle2.cact
```

```
116 tokens
C vs fp32 reference           max|d|=4.5802  median rel=0.7169%  min cos=0.99997222  argmax=98.3%
C vs reference with int8 KV   max|d|=0.6667  median rel=0.1027%  min cos=0.99999768  argmax=98.3%
```

The residual is the int8 KV cache, which is what the model was post-trained for
(`kv_bits=8` in its header). Activations are fp32 throughout: the 2-bit kernel
is exact, so quantising them would buy nothing.

Row sharding across the two cores is bit-identical to running inline. Rows are
independent, so any split gives the same answer.

## Layout

```
src/                the engine: C99, no dependencies, shared by host and ESP32
  nsp.h               packed weight container
  needle.h            public API
  needle_cq.c         quantised matvec kernels
  needle.c            the forward pass
  needle_tok.c        SentencePiece BPE over the embedded tokenizer
tools/
  fetch_model.sh      download the weights
  cact.py             reader for the published model container
  pack_esp.py         repack it for the ESP32
  ref_needle.py       NumPy implementation of the model, the correctness oracle
  verify.py           diff the C engine against it
host/                 host build, for iteration and verification
esp32/                ESP-IDF project
```

