# Qwen3-VL-2B vs Qwen3.5-0.8B for Cosma indexing

**Date**: 2026-05-04
**Hardware**: Apple Silicon Mac (Metal)
**Stack**: llama.cpp Metal build (brew, 9010), llama-server :8001/:8002 for the Qwen3.5 quants; Ollama :11434 for the production Qwen3-VL-2B baseline.

## TL;DR

**Stay on Qwen3-VL-2B-Instruct (Q4_K_M).** The 0.8B is 37% faster but hallucinates content from filenames — it indexed a lightbulb as "a black handgun and steel target" when the file was named `bullet.png`. For an indexer that ships search results to users, that's a hard fail.

## Aggregate metrics

| Model | Mean | p95 | json_ok | flagged | avg keywords |
|---|---:|---:|---:|---:|---:|
| `qwen3-vl:2b-instruct` (Q4_K_M, 1.9 GB) — **production** | 6.74s | 9.44s | 12/12 | 0 | 9.2 |
| `qwen3.5-0.8b-Q4_K_M` (530 MB) | **4.27s** | **5.98s** | 12/12 | 0 | 7.0 |
| `qwen3.5-0.8b-Q8_0` (810 MB) | 5.75s | 9.35s | 11/12 | 1 | 7.2 |

All runs use the **production system prompt** verbatim from
`cosma_backend/summarizer/base.py` and Unsloth's recommended
non-thinking sampling profile (temp 0.7, top_p 0.8, presence 1.5).

## Why the 0.8B's speed is misleading

Aggregate speed numbers say "swap it." Looking at content says don't.
Three representative outputs:

### `bullet.png` — a stylized orange lightbulb

| Model | title | summary |
|---|---|---|
| 2B | "Orange lightbulb" | "A yellowish-orange lightbulb with a rounded top and a tapered bottom, showing a soft glow." ✓ |
| 0.8B Q4 | "Bullet rifle and target" | "A close-up of a black handgun with a circular steel target on the ground..." ✗ |
| 0.8B Q8 | "Bullets and people" | "A group of people holding long white plastic bullets in the air..." ✗ |

### `alian.png` — a single small green pixel-art creature

| Model | title | summary |
|---|---|---|
| 2B | "Pixelated green creature" | "A pixelated green creature with a black top hat and a white face..." ✓ |
| 0.8B Q4 | "Tennis match with players" | "Two tennis players in white shirts are standing on a green court..." ✗ |
| 0.8B Q8 | "Closeup of a person in concrete with glasses" | "A young person wearing glasses and holding a black bag sits on a dark grey brick wall..." ✗ |

### `Snipaste_2025-12-23_03-09-51.jpg` — screenshot of a model-picker UI listing Gemini 2.5 Pro / Claude 4 Sonnet / Claude 3.7 / Claude 3.5 / Auto

| Model | keywords (concrete content captured) |
|---|---|
| 2B | `Gemini`, `Claude`, `Sonnet`, `3.7`, `3.5`, `Auto`, `prompt`, `count` ✓ |
| 0.8B Q4 | `AI models`, `prompt count`, `interface design`, `model comparison` ✗ — generic, lost the names |
| 0.8B Q8 | `Gemini 2.5 Pro`, `Claude 4`, `Sonnet prompts` ✓ — kept the names |

Photos and CJK-text screenshots came out roughly equivalent across all three — the gap shows up specifically on simple icons + heavy-OCR screenshots, where the 0.8B substitutes filename-derived guesses for what's actually pixel-visible. (See `results.json` for all 36 outputs.)

## Why this matters for indexing specifically

The cosma summarizer's output gets fed to the search index. A user later searches `lightbulb` or `green pixel` and expects to find these images. With the 0.8B:

- `bullet.png` would be findable by `gun`, `rifle`, `target`, `handgun` — none of which are in the file.
- `alian.png` would be findable by `tennis`, `tennis player`, `court` — none of which are in the file.

False positives in a personal-file index are worse than misses: the user can't trust the result without opening every match.

The 2B's failure mode is "miss things" (lower recall on rare content). The 0.8B's failure mode is "make things up" (low precision, wrong content). Recall is recoverable with re-indexing; precision is not — once garbage keywords are committed to the index, they pollute every related query until a full rebuild.

## Why the bench's first run looked very different

The first iteration (Ollama-only with `qwen3.5:0.8b`) reported the 0.8B at **38s** mean and a notorious "chicken breast" caption for the same lightbulb. Two separate fixes brought it down to "merely hallucinating":

1. **`enable_thinking=false`** via `--reasoning-budget 0` on llama-server. The Ollama-packaged `qwen3.5:0.8b` had thinking on by default, which inflates latency 3-4× without helping captions.
2. **Resize input to 1024 longest side** before send. llama-server returns HTTP 400 on raw camera photos (~6000×4000 px). Production vision pipelines all downscale upstream, so the resized input is the realistic apples-to-apples test.

Without #1, the previous bench's "thinking loops" warning from the docs was dominating timings. Without #2, the 0.8B looked broken on photos via 400s. The current numbers reflect both fixes applied.

## What about Unsloth's other quants?

Tested only Q4_K_M and Q8_0 — both vision-capable variants Ollama auto-pulled from `hf.co/unsloth/Qwen3.5-0.8B-GGUF`. The smaller dynamic quants (UD-Q3, UD-IQ2_M, etc.) would only push quality down further; the larger ones (BF16) eliminate the size advantage that motivated the swap. Neither direction changes the recommendation.

## Pipeline-level notes uncovered

These came up while wiring the bench up; worth noting somewhere:

- **Ollama 0.23.0 can't load Qwen3.5 GGUFs from HF**: errors with `unknown model architecture: 'qwen35'`. Unsloth's docs explicitly say "currently no Qwen3.5 GGUF works in Ollama due to separate mmproj vision files — use llama.cpp compatible backends." If we ever do swap, the cosma backend's `LlamaCppSummarizer` (which uses the cosmasense/llama-cpp-python fork) would need a `Qwen35VLChatHandler` analogous to the existing `Qwen3VLChatHandler` — that handler does not exist yet in the fork as of 2026-05-04.
- **Resize-to-1024 belongs in the pipeline, not the bench**: the failure mode for full-resolution photos is a 400 from the inference server, which would manifest as a parser failure in production. cosma already handles this for some paths but worth confirming end-to-end.

## Files in this repo

- `bench.py` — the bench. Reads images from `~/Downloads`, hits Ollama and llama-server, scores outputs, prints per-image table + summary.
- `results.json` — full raw outputs from the run that produced this report (36 rows: 12 images × 3 models).
- `run.log` — captured stdout from `python3 bench.py` at the time of report.
- `README.md` — methodology, scoring rubric, how to reproduce.

## Reproducing

```bash
# 1. baseline (Ollama, already pulled)
ollama pull qwen3-vl:2b-instruct

# 2. unsloth quants (pulled into Ollama's blob cache for convenience)
ollama pull hf.co/unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M
ollama pull hf.co/unsloth/Qwen3.5-0.8B-GGUF:Q8_0

# 3. symlink the blobs to readable paths so llama-server can find them
mkdir -p /tmp/qwen35-models && cd /tmp/qwen35-models
# (see git history of bench.py for the symlink commands)

# 4. spin up llama-server for each quant
llama-server -m /tmp/qwen35-models/qwen3.5-0.8b-q4_k_m.gguf \
    --mmproj /tmp/qwen35-models/mmproj.gguf \
    --port 8001 --ctx-size 4096 -ngl 99 --jinja \
    --reasoning-budget 0 --alias qwen3.5-0.8b-Q4_K_M &
llama-server -m /tmp/qwen35-models/qwen3.5-0.8b-q8_0.gguf \
    --mmproj /tmp/qwen35-models/mmproj.gguf \
    --port 8002 --ctx-size 4096 -ngl 99 --jinja \
    --reasoning-budget 0 --alias qwen3.5-0.8b-Q8_0 &

# 5. run
python3 bench.py
```
