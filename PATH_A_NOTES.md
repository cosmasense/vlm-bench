# Path A — porting Qwen3.5 vision support into the cosmasense llama-cpp-python fork

Research notes (not code) for adding `Qwen35VLChatHandler` to
`cosmasense/llama-cpp-python` so cosma's in-process summarizer can
load Qwen3.5 vision GGUFs the same way it already loads Qwen3-VL.

## Why this is needed

Today, `cosma_backend/summarizer/providers.py` requires a chat handler
class named `Qwen3VLChatHandler` that lives in the maintainer's fork
of `llama-cpp-python` (v0.3.32-metal). Stock PyPI `llama-cpp-python`
has `Qwen25VLChatHandler` but not Qwen3-VL or Qwen3.5.

Qwen3.5 (released Feb/Mar 2026) uses a **different vision-encoder
architecture** than Qwen3-VL — GGUFs identify themselves as
`qwen35`, not `qwen3`. The image-token format and chat template
differ enough that `Qwen3VLChatHandler` cannot be reused as-is.

llama.cpp upstream supports it (we ran 0.8B/2B/4B/9B Q4_K_M directly
via `llama-server --mmproj`). The gap is purely in the in-process
Python handler.

## What "porting the handler" actually means

Looking at `Qwen3VLChatHandler` in the fork:

1. It inherits from `MTMDChatHandler` (Multi-Modal Token Decoder
   handler — the generic image+text bridge in llama.cpp).
2. It overrides the **chat template formatting** — Qwen3-VL uses a
   specific `<|vision_start|>...<|vision_end|>` token pattern that
   varies subtly between Qwen vision generations.
3. It overrides the **image-token packing** — how raw pixel patches
   from the mmproj file get inserted into the LLM context as
   placeholder tokens that the prompt template then interleaves.
4. It registers itself in `llama_chat_format` so callers can do
   `getattr(llama_chat_format, "Qwen3VLChatHandler")`.

For Qwen3.5:

1. Inherit from the same `MTMDChatHandler`.
2. Find the Qwen3.5 chat template — it's embedded in the GGUF and
   readable via `llama_model_meta_val_str(model, "tokenizer.chat_template")`.
   llama-server uses Jinja eval (`--jinja`) to apply it; the handler
   needs to do the equivalent.
3. Find the Qwen3.5 vision-token boundary tokens. Inspect the GGUF
   metadata (`gguf-py/scripts/gguf_dump.py`) — Qwen3.5 likely uses
   different special tokens than Qwen3-VL.
4. Register as `Qwen35VLChatHandler` (or `Qwen3_5VLChatHandler` —
   pick a name without a `.` since Python identifier constraints).

## Concrete next steps

1. **Read the upstream llama.cpp PR** that added Qwen3.5 vision
   support. The chat template + image-token logic for the C++ side
   is the source of truth — the Python handler is just a thin shim
   on top.
2. **Diff `Qwen3VLChatHandler` vs `Qwen25VLChatHandler` in the
   fork** — that delta tells you what changes between vision
   generations. The Qwen3.5 changes will be analogous.
3. **Build a wheel locally** before publishing:
   ```bash
   git clone https://github.com/cosmasense/llama-cpp-python
   cd llama-cpp-python
   git checkout v0.3.32-metal  # or wherever the branch is
   # add Qwen35VLChatHandler in llama_cpp/llama_chat_format.py
   # bump submodule llama.cpp to a commit that includes qwen35 support
   CMAKE_ARGS="-DGGML_METAL=on" pip wheel . -w dist/
   ```
4. **Smoke-test** by loading the same `unsloth/Qwen3.5-0.8B-GGUF`
   blobs we used in the bench:
   ```python
   from llama_cpp import Llama
   from llama_cpp.llama_chat_format import Qwen35VLChatHandler
   handler = Qwen35VLChatHandler(clip_model_path="/tmp/qwen35-models/mmproj.gguf")
   llm = Llama(model_path="/tmp/qwen35-models/qwen3.5-0.8b-q4_k_m.gguf",
               chat_handler=handler, n_ctx=4096)
   # repeat one of the bench's image+prompt requests, compare to
   # llama-server output for byte equivalence.
   ```
5. **Wire into cosma**: in `cosma_backend/summarizer/providers.py`
   change `_REQUIRED_HANDLER_CLASS` to be a list (`Qwen3VLChatHandler`,
   `Qwen35VLChatHandler`) and have `_resolve_handler_class` pick by
   inspecting GGUF arch. Add a settings field for which arch the user
   chose.
6. **Update `_FORK_INSTALL_HINT`** to point at the new wheel URL.

## Risk: this is a treadmill

Qwen3.6 is already announced (per the Unsloth page header). Every
new vision generation = another fork update + another release. If
the cadence stays at ~2 months/generation, this becomes maintenance
overhead.

The alternative ("Path B" — switch cosma to subprocess `llama-server`
+ HTTP) is more refactor up front but stops the treadmill cold:
upstream llama.cpp gets new vision support → cosma users `brew
upgrade llama.cpp` → done, no fork update needed.

I'd flag Path B as worth reconsidering at the next vision-generation
cycle. For now Path A is correct because (a) the handler exists for
Qwen3-VL and we're already paying that cost, (b) v1.0.7 already
ships with it, (c) bench shows Qwen3-VL-2B is the right production
model anyway — Qwen3.5 is "nice to have in the picker", not "ship
critical".

## What this repo gave us toward Path A

- `bench_v3.py` exercises Qwen3.5 GGUFs end-to-end with the production
  prompt, so once the handler is written, you can compare its in-
  process outputs to `results_v3.json` for byte-equivalence. The
  bench is the regression suite.
- `samples/` has the canary images. The lightbulb test catches
  filename-hallucination regressions in any new model.
