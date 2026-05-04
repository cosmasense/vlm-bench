# vlm-bench

Cosma indexing-pipeline vision-LLM bench. Used to evaluate whether
swapping the production summarizer model is worth it.

## Production model (baseline)

`unsloth/Qwen3-VL-2B-Instruct-GGUF` at **Q4_K_M** (1.9 GB), served
via Ollama as `qwen3-vl:2b-instruct`. Bound to the cosma backend by
`cosma_backend/settings.py:LlamacppConfig.repo_id`.

## What this bench measures

For each (image, model) pair the bench records:

| Metric | Why it matters |
|---|---|
| `latency_s` | Indexing throughput. The pipeline runs summarize in serial per file. |
| `json_ok` | Output must parse as JSON — the summarizer's caller does `json.loads`. |
| `has_all_keys` | Production schema requires `title` + `summary` + `keywords`. |
| `kw_count` | Search index richness. Fewer keywords → worse recall. |
| `summary_len` | Sanity: short summaries usually mean the model gave up. |
| `flag` | Heuristic failure tag: `invalid_json`, `missing_keys`, `tiny_summary`, `few_keywords`, `leaked_think`, `empty_summary`, `exception`. |

The system prompt is copied verbatim from
`cosma_backend/summarizer/base.py::_get_system_prompt(include_title=True, is_visual=True)`,
so we're testing the same input the indexer actually sends.

All runs include `/no_think` in the user message to suppress
chain-of-thought generation. The indexer is a batch job — we don't
benefit from CoT, and Qwen3.5's default thinking mode roughly
triples latency for our use case.

## Test image set

12 images from `~/Downloads`, selected to mirror typical end-user
indexing inputs:

- 4× **OCR-heavy screenshots** (Cursor dashboard, model list, zyBooks
  course page, a Chinese-text page) — tests legible-text extraction.
- 3× **icons / sprites** (alien, lightbulb, ship) — minimal scene,
  exposes hallucination tendency on simple inputs (the lightbulb is
  the canary: previous bench had Qwen3.5 call it a "chicken breast").
- 3× **photographs** (portrait, scene, campus) — caption quality.
- 2× **UI screenshots** (game guide, misc) — mixed text+visual.

## Models compared

| Tag | Model | Quant | Approx size |
|---|---|---|---|
| `qwen3-vl:2b-instruct` | Qwen3-VL-2B (production) | Q4_K_M | 1.9 GB |
| `hf.co/unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M` | Qwen3.5-0.8B | Q4_K_M | ~530 MB |
| `hf.co/unsloth/Qwen3.5-0.8B-GGUF:Q8_0` | Qwen3.5-0.8B | Q8_0 | ~810 MB |

Q4_K_M of the smaller model gives an apples-to-apples 4-bit
comparison with the current default. Q8_0 is included as the
"best 0.8B can do" upper bound — if even Q8 isn't competitive,
no smaller quant will be.

## How to run

```bash
ollama pull qwen3-vl:2b-instruct
ollama pull hf.co/unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M
ollama pull hf.co/unsloth/Qwen3.5-0.8B-GGUF:Q8_0
python3 bench.py
```

Writes per-image side-by-side to stdout, a ranked summary at the
end, and full raw outputs to `results.json`. The interpretation
+ recommendation lives in `REPORT.md`.

## Limitations

- 12 images is small for stable timings — re-run if numbers look
  noisy. Mean ± p95 helps but doesn't fully de-noise.
- Hallucination scoring is by hand inspection; the heuristic flags
  catch obvious failures (empty / tiny summaries, leaked
  `<think>` tokens) but a confident-but-wrong caption ("chicken
  breast" for a lightbulb) still parses fine.
- Hardware here is one machine. A different Mac may shift latency
  ratios but not quality conclusions.
