# bench v4 — design + hypotheses

Written **before** running so the analysis isn't post-hoc rationalization. Predictions are graded against actual results in `REPORT3.md`.

## Dataset (12 files, 4 categories of 3)

The Unsplash photo IDs picked for `objects/` and `ui/` turned out not to match their intended categories on inspection. Reframed honestly:

| Category | Intended | Actual content | What it actually tests |
|---|---|---|---|
| `objects/` | Sprites / game icons | 3 coffee photos: hands clinking cups, branded coffee bag with text, hands holding latte art | **Near-duplicate distinguishability.** Can a model produce 3 distinct keyword sets, or does it collapse them to "coffee, coffee, coffee"? |
| `ui/` | Product UI screenshots | 3 people-with-computers photos: meeting w/ laptop in bg, hands typing, hand drawing wireframe sketch on paper | **Subject vs. incidental content.** Does the model say "people working" (correct) or hallucinate "VS Code interface" (filename-style failure)? |
| `architecture/` | ✓ Architecture | 3 modern architecture photos: angular glass facade, curved-roof museum, brick low-angle | Architectural style recognition, color/texture description. |
| `pdfs/` | ✓ Long hard PDFs | arXiv: Attention Is All You Need, GPT-3, BERT (text-extracted via `pdftotext`, truncated to 8000 chars) | Long-context comprehension, technical-term extraction. |

## Models

Three, each at Q4_K_M to keep quant constant:

| Model | Approx size | Notes |
|---|---|---|
| `qwen3-vl:2b-instruct` | 1.9 GB | Production baseline (via Ollama). |
| `qwen3.5-0.8b-Q4_K_M` | 0.5 GB | The "is small enough?" candidate. |
| `qwen3.5-2b-Q4_K_M` | 1.2 GB | Same size class as 2B-VL — does the new family beat the old at parity? |

## Reps + total inferences

3 reps per (model, file) = `3 × 12 × 3 = 108` inferences. With v3's ~5-7s mean for the 2B models and ~4s for 0.8B, total wall-clock should be 12-15 minutes inference + ~30s of model swap overhead.

## What we measure

Per individual run:
- `latency_s` — wall-clock, end-to-end including image base64 + transport.
- `peak_rss_mb` — sampled at 250 ms via `ps -o rss=` on the model server's pid (llama-server for Qwen3.5; the `ollama runner` child for qwen3-vl). For Ollama the pid can churn between requests; we capture the value at start-of-loop and accept it might miss process restarts.
- `json_ok` — output parses as JSON after stripping ```json fences and `<think>` tags.
- `has_all_keys` — JSON has `title`, `summary`, `keywords`.
- `kw_count` — len(keywords).
- `summary_len` — len(summary in chars).
- `flag` — heuristic failure tag: `invalid_json` / `missing_keys` / `tiny_summary` / `few_keywords` / `leaked_think` / `empty_summary` / `exception`.

Per (model, file):
- mean + stdev of latency across 3 reps. Stdev tells us how stable each pair is — if it's huge, the mean is misleading.

Per (model, category):
- Mean latency, json_ok rate, kw_count avg.

Per model overall:
- Mean / stdev / p95 latency, json_ok %, kw avg, peak RSS observed.

## Hypotheses going in

1. **Latency ranking will be 0.8B < 2B-VL < Qwen3.5-2B** (consistent with v3). Specifically: 0.8B around 4 s, 2B-VL around 6 s, Qwen3.5-2B around 7 s.
2. **Coffee photos: 0.8B will produce near-identical keyword sets across the three** despite each photo having distinct content (cheers gesture vs. branded packaging vs. latte art). 2B-VL will produce three distinct sets. This is the v4 lightbulb canary equivalent — testing whether the small model's cheaper attention budget collapses similar-looking inputs.
3. **Workplace photos: 0.8B will be more likely to hallucinate "VS Code" / "code editor" / "IDE" content** than 2B models, because the filename `ui/01.jpg` in the prompt biases it. 2B models will correctly identify "people in a meeting" / "person typing" / "wireframe sketch" as primary subjects.
4. **Architecture photos: smallest gap between models.** All three should describe modern architecture, glass, brick, etc. correctly. This is the "easy" category for all sizes.
5. **PDFs: 2B-VL will produce the densest keyword set** (it consistently topped kw_avg in v3). Qwen3.5-2B may match or exceed on text since this is text-only inference where the unified Qwen3.5 family should not be disadvantaged by vision-specific training.
6. **JSON validity ≥ 95% across all (model, category) cells.** With `--reasoning-budget 0` and the simple schema, all three should reliably emit JSON.
7. **Per-pair stdev ≤ 1.0 s** on Apple Silicon Metal. Inference on quantized models is reasonably deterministic at temp 0.7.

## Where we expect the small model to make sense

Predictions for the analysis section of the final report:

- ✅ **Architecture photos** — the small model probably matches the big one on description quality, so 0.8B saves ~30% latency for free.
- ⚠️ **PDFs** — quality-likely-equivalent on text, but with 0.8B's cheaper context budget it may truncate keyword lists.
- ❌ **Coffee photos (near-duplicate test)** — the 0.8B will probably produce search-poisoning identical keywords across the three.
- ❌ **Workplace photos (subject test)** — the 0.8B will probably mislabel the subject from filename bias.

If hypotheses 3 & 2 are wrong (the small model handles those correctly), it would meaningfully change the recommendation toward "ship the 0.8B for indexing photos; keep 2B for screenshots/text." If they're right, the recommendation stays "2B-VL only" as in v3.

## What this bench does NOT measure

- Search-recall outcomes (would need a real index to query).
- Long-document handling beyond 8000 chars (PDFs are truncated).
- Realistic mid-pipeline conditions (caches warm, batched calls, etc.).
- Behavior with CJK or non-Latin OCR — none of the chosen images carry meaningful CJK content.

## What the next bench (v5) should test

These are open questions this bench *won't* close:

1. **Real product UI screenshots** with dense text. Replace the workplace-photos category with actual VS Code / Figma / dashboard captures and re-test the OCR-completeness gap between 2B-VL and 2B-Qwen3.5.
2. **True pixel-art sprites** (find an Open Game Art set with stable URLs) to retest the "filename hallucination on simple input" canary from v1/v3.
3. **CJK + Cyrillic + Arabic OCR** screenshots to see whether the new Qwen3.5 family's "201 languages" claim translates to better non-Latin OCR.
4. **Latency at higher reps** (≥5 reps per pair) on the chosen "winner" pair so we know how much to trust the recommendation under jitter.
5. **End-to-end indexer test**: feed both models' outputs into a small embedding index, run a fixed query set, compare recall@5. The proxy metrics (kw_count, summary_len) only weakly predict actual search quality.
