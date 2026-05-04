"""
Cosma indexing-pipeline vision A/B/C bench.

Use case mirror: feed images through the EXACT production system prompt
the cosma summarizer sends (cosma_backend/summarizer/base.py
_get_system_prompt(include_title=True, is_visual=True)). For each
image + model pair, measure:

  - latency: end-to-end Ollama call duration
  - json_ok: did the output parse as valid JSON?
  - has_keys: does it have title/summary/keywords?
  - kw_count: how many keywords (proxy for indexer richness)
  - flag: pre-flag obvious failures (empty summary, ALL_CAPS, etc.)

The 12 test images were picked to mirror what an end user actually
indexes: heavy-OCR screenshots, dashboards, sprites/icons, photos,
foreign-language text. Bench prints a side-by-side per image then a
ranked summary so we can pick a winner without eyeballing 36 outputs.

All runs use /no_think — Qwen3.5 ships with thinking-mode by default,
which inflates latency by ~3x for our use case. The indexer is a
batch job that doesn't benefit from chain-of-thought reasoning.
"""
from __future__ import annotations
import base64
import io
import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
import urllib.request

from PIL import Image  # resize before send — see _b64_resized for why

# Two backend protocols, two endpoints:
#  - Ollama for the 2B baseline (already a tagged model on the user's
#    machine, served from :11434).
#  - llama-server (OpenAI-compatible) for each Qwen3.5 quant. Ollama
#    0.23 can't load Qwen3.5 GGUFs ("unknown architecture qwen35") so
#    we run llama.cpp directly. Each quant gets its own port — the
#    bench dispatches by model name.
OLLAMA = "http://localhost:11434/api/chat"

# Maps "logical" model name → (backend, endpoint, real_model_id).
# Logical names are what the bench prints + groups by; backend tells
# us which request shape to send.
ENDPOINTS: dict[str, tuple[str, str, str]] = {
    "qwen3-vl:2b-instruct": ("ollama", OLLAMA, "qwen3-vl:2b-instruct"),
    "qwen3.5-0.8b-Q4_K_M":  ("llama-server", "http://localhost:8001/v1/chat/completions", "qwen3.5-0.8b"),
    "qwen3.5-0.8b-Q8_0":    ("llama-server", "http://localhost:8002/v1/chat/completions", "qwen3.5-0.8b"),
}

PROMPT_HEADER = (
    "You are an indexing-summary writer. Given a file's content, "
    "return ONLY a JSON object matching the schema below. No prose, "
    "no markdown fences, no commentary. The JSON's `summary` and "
    "`keywords` are what a search engine will index — make them dense "
    "with concrete nouns a person would actually type into a search "
    "box.\n\n"
)
SYSTEM_PROMPT = PROMPT_HEADER + (
    "This call has an attached image. Look at it and describe "
    "what is visible: concrete objects, people, places, "
    "scenes, actions, and any legible text. Treat any text "
    "in the user message (filename, dimensions, etc.) as "
    "context only — describe the picture, not the metadata.\n"
    "Schema:\n"
    '  {"title": "<1-5 word title naming the main subject>",\n'
    '   "summary": "<1-2 sentences naming concrete visible '
    'objects, people, scenes, and any legible text>",\n'
    '   "keywords": ["<5-12 distinctive visual nouns or '
    'noun-phrases, lowercase>"]}\n'
    "Example (a hiking photo):\n"
    '  {"title": "Group hike at sunset",\n'
    '   "summary": "Five hikers with backpacks on a mountain '
    'ridge under an orange sky; pine trees in the foreground.",\n'
    '   "keywords": ["hiking", "sunset", "mountain ridge", '
    '"backpacks", "pine trees", "group photo"]}'
)

DL = Path("/Users/ethanpan/Downloads")
IMAGES = [
    DL / "Snipaste_2025-12-23_03-09-51.jpg",                  # OCR: model list
    DL / "cursor-2025.png",                                    # OCR: dashboard metrics
    DL / "My activity - COMP SCI 577 001_ Introduction to Algorithms _ zyBooks.jpeg",  # OCR: course page
    DL / "alian.png",                                          # icon: pixel sprite
    DL / "bullet.png",                                         # icon: lightbulb (the hallucination canary)
    DL / "ship.png",                                           # icon: game sprite
    DL / "Ethan_Pan_ILCE-7C_20260405-_DSC9776.JPG",            # photo: person
    DL / "Ethan_Pan_ILCE-7C_20260405-_DSC9789.JPG",            # photo: misc scene
    DL / "2021_0407_campus-pictures_jc_0185 (1).jpg",          # photo: campus
    DL / "guide_MrCrab.png",                                   # UI: game guide
    DL / "Snipaste_2026-04-23_14-46-05.png",                   # screenshot: misc
    DL / "用户题目页.png",                                       # OCR: Chinese text
]

MODELS = list(ENDPOINTS.keys())


@dataclass
class Result:
    image: str
    model: str
    latency_s: float
    raw: str
    json_ok: bool
    has_all_keys: bool
    kw_count: int
    summary_len: int
    flag: str  # heuristic failure flag, empty if clean


def b64(p: Path) -> str:
    """Encode image bytes after resizing the longest side to 1024 px.
    Why resize: llama-server returns HTTP 400 when fed full-resolution
    camera photos (~6000px, ~1.5 MB after base64) — its default image
    handling doesn't downscale. Production vision pipelines all resize
    upstream of the model anyway, so this is the realistic input. PNG
    sprites and small icons fall through unmodified — Image.thumbnail
    is a no-op when the longest side is already below the cap. JPEG
    quality 90 keeps post-resize bytes well under the limit while
    preserving OCR-relevant detail.
    """
    MAX_SIDE = 1024
    img = Image.open(p)
    if max(img.size) > MAX_SIDE:
        img = img.copy()
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
        buf = io.BytesIO()
        # Force JPEG output for non-transparent images so we don't blow
        # up bytes on huge PNG photos. Keep PNG only when transparency
        # is in play (mode RGBA / P with alpha) — sprites that depend
        # on it would lose meaning otherwise.
        if img.mode in ("RGBA", "P") and "transparency" in img.info:
            img.save(buf, format="PNG", optimize=True)
        else:
            img.convert("RGB").save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    return base64.b64encode(p.read_bytes()).decode("ascii")


def _build_ollama_payload(model_id: str, image_path: Path) -> dict:
    """Ollama's /api/chat: image bytes go on the user message itself
    via a top-level `images` array. Append /no_think to the text so
    Qwen3.5 skips the thinking phase even on the Ollama-tagged build.
    """
    return {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"filename: {image_path.name} /no_think",
                "images": [b64(image_path)],
            },
        ],
        "stream": False,
        "options": {"temperature": 0.1, "num_ctx": 4096},
    }


def _build_oai_payload(model_id: str, image_path: Path) -> dict:
    """OpenAI-compatible /v1/chat/completions (what llama-server speaks):
    image goes inside the user message's content array as
    `{type:"image_url", image_url:{url:"data:image/...;base64,..."}}`.
    The mime tag matters for some servers — we honor the file
    extension. Pass enable_thinking=false explicitly even though 0.8B
    defaults to non-thinking, in case the server's default chat
    template doesn't honor it (we observed `thinking = 1` in the
    server log on the first run).
    Sampling values are Unsloth's recommended non-thinking general-
    task profile from https://unsloth.ai/docs/models/qwen3.5 — picking
    these matters because temp=0.1 (the value previous bench used)
    sits well outside what Qwen3.5 was trained for, which biases
    quality measurements unfairly against the model.
    """
    suffix = image_path.suffix.lower().lstrip(".") or "jpeg"
    if suffix == "jpg":
        suffix = "jpeg"
    return {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"filename: {image_path.name}"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{suffix};base64,{b64(image_path)}",
                        },
                    },
                ],
            },
        ],
        "temperature": 0.7,
        "top_p": 0.8,
        "presence_penalty": 1.5,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def call(model: str, image_path: Path) -> tuple[str, float]:
    backend, url, model_id = ENDPOINTS[model]
    if backend == "ollama":
        payload = _build_ollama_payload(model_id, image_path)
    elif backend == "llama-server":
        payload = _build_oai_payload(model_id, image_path)
    else:
        raise ValueError(f"unknown backend: {backend}")

    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read())
    dt = time.monotonic() - t0

    # Ollama puts the assistant message at .message.content;
    # OpenAI-compatible APIs put it at .choices[0].message.content.
    if backend == "ollama":
        return data["message"]["content"], dt
    return data["choices"][0]["message"]["content"], dt


def parse_and_score(raw: str) -> tuple[bool, bool, int, int, str]:
    """Parse the model's output; return (json_ok, has_all_keys,
    keyword_count, summary_len, flag) where flag is a short hint at
    obvious failure modes ("" if clean)."""
    # Tolerate models that wrap JSON in ```json fences despite the
    # prompt explicitly forbidding them. Production code handles this
    # too, so the bench should mirror that leniency.
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    # Strip <think>...</think> tags if the model leaked them despite
    # /no_think — counts as a soft flag.
    leaked_think = "<think>" in s.lower() or "</think>" in s.lower()
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL | re.IGNORECASE).strip()
    try:
        obj = json.loads(s)
    except Exception:
        return False, False, 0, 0, "invalid_json"
    has_all = all(k in obj for k in ("title", "summary", "keywords"))
    if not has_all:
        return True, False, 0, 0, "missing_keys"
    summary = obj.get("summary") or ""
    keywords = obj.get("keywords") or []
    flag = ""
    if leaked_think:
        flag = "leaked_think"
    elif not summary.strip():
        flag = "empty_summary"
    elif len(summary) < 30:
        flag = "tiny_summary"
    elif len(keywords) < 4:
        flag = "few_keywords"
    return True, has_all, len(keywords), len(summary), flag


def main() -> None:
    rows: list[Result] = []
    for img in IMAGES:
        if not img.exists():
            print(f"missing: {img}")
            continue
        print(f"\n=== {img.name} ===")
        for m in MODELS:
            try:
                raw, dt = call(m, img)
                ok, has_all, kc, sl, flag = parse_and_score(raw)
            except Exception as e:
                raw, dt, ok, has_all, kc, sl, flag = f"ERROR: {e}", 0.0, False, False, 0, 0, "exception"
            rows.append(Result(img.name, m, dt, raw, ok, has_all, kc, sl, flag))
            tag = "OK" if (ok and has_all and not flag) else (flag or "BAD")
            print(f"  {m:<50} {dt:6.2f}s  {tag}")

    print("\n\n=== summary ===")
    print(f"{'model':<50} {'mean_s':>7} {'p95_s':>7} {'json_ok':>8} {'all_keys':>9} {'flagged':>8} {'avg_kw':>7}")
    for m in MODELS:
        ms = [r for r in rows if r.model == m and r.latency_s > 0]
        if not ms:
            continue
        lats = sorted(r.latency_s for r in ms)
        p95 = lats[max(0, int(0.95 * len(lats)) - 1)]
        json_ok = sum(1 for r in ms if r.json_ok)
        all_keys = sum(1 for r in ms if r.has_all_keys)
        flagged = sum(1 for r in ms if r.flag)
        avg_kw = sum(r.kw_count for r in ms) / len(ms) if ms else 0
        print(f"{m:<50} {sum(lats)/len(lats):>7.2f} {p95:>7.2f} {json_ok:>3}/{len(ms):<4} {all_keys:>3}/{len(ms):<5} {flagged:>3}/{len(ms):<4} {avg_kw:>7.1f}")

    here = Path(__file__).parent
    out_file = here / "results.json"
    out_file.write_text(json.dumps([asdict(r) for r in rows], indent=2))
    print(f"\nfull outputs: {out_file}")


if __name__ == "__main__":
    main()
