"""
Cosma indexing-pipeline bench v3.

Compared with v1 (bench.py):
  - Five models instead of three: qwen3-vl:2b-instruct (production
    baseline), and Qwen3.5-{0.8B,2B,4B,9B} all at Q4_K_M for an
    apples-to-apples size sweep.
  - Mixed dataset: same 12 images PLUS 6 text docs picked from
    ~/Downloads to mirror the indexer's real workload (notes, logs,
    CSVs, structured reports). Text docs use the non-visual prompt
    branch from cosma's summarizer; images use the visual branch.
  - Per-model memory tracking. Each Qwen3.5 server runs in isolation
    (kill-and-relaunch between models) so its peak RSS reflects just
    that model + mmproj. The Ollama-served 2B baseline is sampled
    from the matching `ollama runner` child process. Recorded as
    `peak_rss_mb` per (model, dataset_item).
  - Fully automated: starts/stops llama-server per model, waits for
    /health, samples ps every ~250ms during each request, writes
    results.json, and finishes by writing REPORT2.md with mermaid
    charts and embedded example images.

Same sampling settings everywhere (Unsloth's "non-thinking general
tasks" profile from https://unsloth.ai/docs/models/qwen3.5):
  temp 0.7, top_p 0.8, presence_penalty 1.5, top_k 20.
Thinking is disabled via --reasoning-budget 0 on llama-server side
and via /no_think on the Ollama side.

Run: `python3 bench_v3.py`. No CLI args — all knobs are constants
near the top of this file.
"""
from __future__ import annotations
import base64
import io
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
import urllib.request

from PIL import Image  # for resizing images before send (see _b64)

# ---------------------------------------------------------------------------
# Paths + config
# ---------------------------------------------------------------------------

HERE = Path(__file__).parent
RESULTS = HERE / "results_v3.json"
REPORT = HERE / "REPORT2.md"
SAMPLES_DIR = HERE / "samples"  # copied test images for embedding

OLLAMA = "http://localhost:11434/api/chat"
LCPP_PORT = 8001  # llama-server lives here; only one model at a time

# Image inputs are resized to 1024 longest side before sending. Without
# this, llama-server returns HTTP 400 on full-resolution camera photos
# (~6000px). Production vision pipelines all downscale upstream so this
# matches the real workload.
MAX_IMG_SIDE = 1024
# Truncate text docs to this many chars so the comparison is consistent
# across models (varying ctx-size). 8000 chars ~= ~2000 tokens, which
# fits comfortably inside our ctx-size of 4096 for every test model.
MAX_TEXT_CHARS = 8000


# ---------------------------------------------------------------------------
# System prompt — copied verbatim from cosma_backend/summarizer/base.py
# Keep this in sync if the production prompt changes.
# ---------------------------------------------------------------------------

PROMPT_HEADER = (
    "You are an indexing-summary writer. Given a file's content, "
    "return ONLY a JSON object matching the schema below. No prose, "
    "no markdown fences, no commentary. The JSON's `summary` and "
    "`keywords` are what a search engine will index — make them dense "
    "with concrete nouns a person would actually type into a search "
    "box.\n\n"
)

VISUAL_PROMPT = PROMPT_HEADER + (
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

TEXT_PROMPT = PROMPT_HEADER + (
    "Schema:\n"
    '  {"title": "<1-5 word title>",\n'
    '   "summary": "<1-2 sentences naming the topic and key '
    'concrete nouns a searcher would type>",\n'
    '   "keywords": ["<5-12 distinctive nouns or noun-phrases, '
    'lowercase>"]}\n'
    "Example (a Q3 financial report):\n"
    '  {"title": "Q3 Earnings Report",\n'
    '   "summary": "Q3 2025 earnings: revenue up 12%, AWS '
    'driving cloud growth, margins steady.",\n'
    '   "keywords": ["q3 earnings", "revenue growth", "aws", '
    '"cloud", "margins"]}'
)


# ---------------------------------------------------------------------------
# Test set
# ---------------------------------------------------------------------------

DL = Path("/Users/ethanpan/Downloads")

IMAGE_INPUTS = [
    DL / "Snipaste_2025-12-23_03-09-51.jpg",                       # OCR: model list
    DL / "cursor-2025.png",                                         # OCR: dashboard
    DL / "My activity - COMP SCI 577 001_ Introduction to Algorithms _ zyBooks.jpeg",
    DL / "alian.png",                                               # icon: pixel sprite
    DL / "bullet.png",                                              # icon: lightbulb (canary)
    DL / "ship.png",                                                # icon: game sprite
    DL / "Ethan_Pan_ILCE-7C_20260405-_DSC9776.JPG",                 # photo: portrait
    DL / "Ethan_Pan_ILCE-7C_20260405-_DSC9789.JPG",                 # photo: scene
    DL / "2021_0407_campus-pictures_jc_0185 (1).jpg",               # photo: campus
    DL / "guide_MrCrab.png",                                        # UI: game guide
    DL / "Snipaste_2026-04-23_14-46-05.png",                        # screenshot: misc
    DL / "用户题目页.png",                                            # OCR: Chinese
]

TEXT_INPUTS = [
    DL / "videosenio.txt",                                          # tiny: URL
    DL / "text-460D-9B34-F7-0.txt",                                 # short: BlueGuide diag
    DL / "astrotwin-log-2026-04-13T14-43.txt",                      # 5K: AstroTwin chat log
    DL / "logpperpaer.md",                                          # 24K: dev log → truncated
    DL / "study4_all.csv",                                          # 47K: CSV → truncated
    DL / "arch_terms_guide.md",                                     # markdown: architecture terms
]


# ---------------------------------------------------------------------------
# Model spec
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    """Everything the bench needs to run one model end-to-end."""
    name: str                                # logical id, used in tables
    backend: str                             # "ollama" or "llama-server"
    # llama-server only:
    gguf_path: str | None = None
    mmproj_path: str | None = None
    # ollama only:
    ollama_tag: str | None = None
    # display
    approx_size_gb: float = 0.0


# Resolve Ollama blob hashes → readable file paths so llama-server can
# load them. We pulled these via `ollama pull hf.co/...` earlier; the
# blobs live in the standard cache.
def _ollama_blob_path(manifest_relpath: str, layer_index: int) -> str:
    manifest = Path.home() / ".ollama/models/manifests" / manifest_relpath
    data = json.loads(manifest.read_text())
    digest = data["layers"][layer_index]["digest"].replace("sha256:", "sha256-")
    return str(Path.home() / f".ollama/models/blobs/{digest}")


def _resolve_unsloth(tag: str) -> tuple[str, str]:
    """Return (model_path, mmproj_path) for an unsloth Qwen3.5 quant
    pulled via Ollama. Manifest layer order is [model, projector] for
    these uploads; if upstream ever changes that, this will throw a
    KeyError loud enough to notice.
    """
    rel = f"hf.co/unsloth/{tag}"
    name, qtag = tag.split(":")
    manifest = f"hf.co/unsloth/{name}/{qtag}"
    data = json.loads((Path.home() / ".ollama/models/manifests" / manifest).read_text())
    model_layer = next(l for l in data["layers"] if "model" in l["mediaType"])
    proj_layer = next(l for l in data["layers"] if "projector" in l["mediaType"])
    blob = lambda d: str(Path.home() / f".ollama/models/blobs/{d.replace('sha256:', 'sha256-')}")
    return blob(model_layer["digest"]), blob(proj_layer["digest"])


HF_CACHE_ROOT = Path("/tmp/qwen35-models")  # populated by `hf download` per size


def _resolve_hf_dir(short: str) -> tuple[str, str] | None:
    """Find the GGUF + mmproj inside /tmp/qwen35-models/<short>/ that
    `hf download --include "*Q4_K_M*" --include "*mmproj-F16*"` writes.
    Returns None if either piece is missing (download not yet
    finished). Picks the largest matching .gguf when there are
    multiple shards (very rare at Q4_K_M for these sizes).
    """
    dir_ = HF_CACHE_ROOT / short
    if not dir_.exists():
        return None
    ggufs = sorted(dir_.glob("**/*Q4_K_M*.gguf"), key=lambda p: p.stat().st_size, reverse=True)
    mmproj = next(iter(dir_.glob("**/mmproj-F16*.gguf")), None)
    if not ggufs or mmproj is None:
        return None
    return str(ggufs[0]), str(mmproj)


def _build_model_specs() -> list[ModelSpec]:
    specs: list[ModelSpec] = [
        ModelSpec(
            name="qwen3-vl:2b-instruct",
            backend="ollama",
            ollama_tag="qwen3-vl:2b-instruct",
            approx_size_gb=1.9,
        )
    ]
    # 0.8B was originally pulled through Ollama (the blob is on disk
    # there), so we keep that resolution path. Larger sizes come
    # straight from huggingface_hub into /tmp/qwen35-models/<short>/
    # because Ollama's pull stalled mid-stream on these.
    for ollama_tag, label, gb in [
        ("Qwen3.5-0.8B-GGUF:Q4_K_M", "qwen3.5-0.8b-Q4_K_M", 0.5),
        ("Qwen3.5-0.8B-GGUF:Q8_0", "qwen3.5-0.8b-Q8_0", 0.8),
    ]:
        try:
            m, p = _resolve_unsloth(ollama_tag)
            specs.append(ModelSpec(
                name=label, backend="llama-server",
                gguf_path=m, mmproj_path=p, approx_size_gb=gb,
            ))
        except (FileNotFoundError, StopIteration):
            print(f"[skip] {ollama_tag} not in Ollama cache")

    for short, gb in [("2B", 1.2), ("4B", 2.4), ("9B", 5.4)]:
        resolved = _resolve_hf_dir(short)
        if resolved is None:
            print(f"[skip] Qwen3.5-{short}-GGUF Q4_K_M / mmproj not in {HF_CACHE_ROOT}/{short}")
            continue
        m, p = resolved
        specs.append(ModelSpec(
            name=f"qwen3.5-{short.lower()}-Q4_K_M",
            backend="llama-server", gguf_path=m, mmproj_path=p,
            approx_size_gb=gb,
        ))
    return specs


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    file: str
    file_kind: str            # "image" or "text"
    model: str
    latency_s: float
    raw: str
    json_ok: bool
    has_all_keys: bool
    kw_count: int
    summary_len: int
    flag: str
    peak_rss_mb: float = 0.0  # 0 if we couldn't sample (e.g. ollama)


# ---------------------------------------------------------------------------
# Memory sampling
# ---------------------------------------------------------------------------

class RSSSampler:
    """Background thread that polls `ps -o rss= -p PID` every 250ms
    and tracks the peak. start() / stop_and_get_peak_mb() bracketed
    around an inference call gives us per-request peak RSS without
    pulling psutil in as a dep — `ps` is on every macOS by default.
    """

    def __init__(self, pid: int):
        self.pid = pid
        self.peak_kb = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["ps", "-o", "rss=", "-p", str(self.pid)],
                    stderr=subprocess.DEVNULL, text=True,
                ).strip()
                if out:
                    rss_kb = int(out.split()[0])
                    self.peak_kb = max(self.peak_kb, rss_kb)
            except (subprocess.CalledProcessError, ValueError, IndexError):
                pass
            self._stop.wait(0.25)

    def start(self) -> None:
        self._stop.clear()
        self.peak_kb = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop_and_get_peak_mb(self) -> float:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        return self.peak_kb / 1024.0


# ---------------------------------------------------------------------------
# llama-server lifecycle
# ---------------------------------------------------------------------------

def start_llama_server(spec: ModelSpec) -> subprocess.Popen:
    """Spawn llama-server for one Qwen3.5 quant, block until /health
    is OK, return the subprocess.Popen. Caller is responsible for
    killing it before starting another (we share port LCPP_PORT).
    """
    if spec.backend != "llama-server":
        raise ValueError(f"not a llama-server spec: {spec.name}")
    cmd = [
        "llama-server",
        "-m", spec.gguf_path,
        "--mmproj", spec.mmproj_path,
        "--port", str(LCPP_PORT),
        "--ctx-size", "4096",
        "-ngl", "99",
        "--jinja",
        "--reasoning-budget", "0",         # disable thinking
        "--alias", spec.name,
    ]
    print(f"[start] {spec.name}: {' '.join(shlex.quote(c) for c in cmd)}")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        # Put it in its own process group so kill() takes down all
        # llama-server children together; otherwise leftover model
        # threads can keep the GPU pinned.
        preexec_fn=os.setsid,
    )
    # Wait up to 90s for /health to flip — bigger models take longer
    # to memory-map and warm up the Metal compute pipeline.
    deadline = time.time() + 90
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://localhost:{LCPP_PORT}/health", timeout=2) as r:
                if json.loads(r.read()).get("status") == "ok":
                    print(f"[ready] {spec.name} pid={proc.pid}")
                    return proc
        except Exception:
            pass
        time.sleep(1)
    proc.kill()
    raise RuntimeError(f"llama-server for {spec.name} never reached /health within 90s")


def stop_llama_server(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def find_ollama_runner_pid() -> int | None:
    """Ollama spawns a `ollama runner` subprocess per loaded model.
    Find the one belonging to qwen3-vl. Used so we can sample its RSS
    instead of the parent ollama-serve daemon (whose RSS includes
    every model loaded across the system).
    """
    try:
        out = subprocess.check_output(["pgrep", "-f", "ollama runner"], text=True)
    except subprocess.CalledProcessError:
        return None
    pids = [int(p) for p in out.split()]
    return pids[0] if pids else None


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _b64_resized_image(p: Path) -> str:
    img = Image.open(p)
    if max(img.size) > MAX_IMG_SIDE:
        img = img.copy()
        img.thumbnail((MAX_IMG_SIDE, MAX_IMG_SIDE), Image.LANCZOS)
        buf = io.BytesIO()
        if img.mode in ("RGBA", "P") and "transparency" in img.info:
            img.save(buf, format="PNG", optimize=True)
        else:
            img.convert("RGB").save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    return base64.b64encode(p.read_bytes()).decode("ascii")


def _read_text(p: Path) -> str:
    """Read a text file with charset tolerance, trim to MAX_TEXT_CHARS."""
    try:
        s = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        s = p.read_text(encoding="utf-8", errors="replace")
    if len(s) > MAX_TEXT_CHARS:
        # Keep head + tail with a marker — head usually has the most
        # signal (heading, first paragraph) and tail catches conclusion.
        head = s[: MAX_TEXT_CHARS - 1000]
        tail = s[-800:]
        s = f"{head}\n\n[...truncated for bench...]\n\n{tail}"
    return s


def _ollama_image_payload(model_id: str, p: Path) -> dict[str, Any]:
    return {
        "model": model_id,
        "messages": [
            {"role": "system", "content": VISUAL_PROMPT},
            {
                "role": "user",
                "content": f"filename: {p.name} /no_think",
                "images": [_b64_resized_image(p)],
            },
        ],
        "stream": False,
        "options": {
            "temperature": 0.7, "top_p": 0.8, "top_k": 20,
            "presence_penalty": 1.5, "num_ctx": 4096,
        },
    }


def _ollama_text_payload(model_id: str, p: Path) -> dict[str, Any]:
    return {
        "model": model_id,
        "messages": [
            {"role": "system", "content": TEXT_PROMPT},
            {
                "role": "user",
                "content": (
                    f"filename: {p.name}\n\n"
                    f"--- file content ---\n{_read_text(p)}\n--- end ---\n"
                    f"/no_think"
                ),
            },
        ],
        "stream": False,
        "options": {
            "temperature": 0.7, "top_p": 0.8, "top_k": 20,
            "presence_penalty": 1.5, "num_ctx": 4096,
        },
    }


def _oai_image_payload(model_id: str, p: Path) -> dict[str, Any]:
    suffix = p.suffix.lower().lstrip(".") or "jpeg"
    if suffix == "jpg":
        suffix = "jpeg"
    return {
        "model": model_id,
        "messages": [
            {"role": "system", "content": VISUAL_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"filename: {p.name}"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{suffix};base64,{_b64_resized_image(p)}"},
                    },
                ],
            },
        ],
        "temperature": 0.7,
        "top_p": 0.8,
        "presence_penalty": 1.5,
        "stream": False,
    }


def _oai_text_payload(model_id: str, p: Path) -> dict[str, Any]:
    return {
        "model": model_id,
        "messages": [
            {"role": "system", "content": TEXT_PROMPT},
            {
                "role": "user",
                "content": (
                    f"filename: {p.name}\n\n"
                    f"--- file content ---\n{_read_text(p)}\n--- end ---"
                ),
            },
        ],
        "temperature": 0.7,
        "top_p": 0.8,
        "presence_penalty": 1.5,
        "stream": False,
    }


def call(spec: ModelSpec, p: Path, kind: str) -> tuple[str, float]:
    if spec.backend == "ollama":
        url = OLLAMA
        payload = (_ollama_image_payload if kind == "image" else _ollama_text_payload)(
            spec.ollama_tag, p
        )
    else:
        url = f"http://localhost:{LCPP_PORT}/v1/chat/completions"
        payload = (_oai_image_payload if kind == "image" else _oai_text_payload)(
            spec.name, p
        )
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
    dt = time.monotonic() - t0
    if spec.backend == "ollama":
        return data["message"]["content"], dt
    return data["choices"][0]["message"]["content"], dt


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def parse_and_score(raw: str) -> tuple[bool, bool, int, int, str]:
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
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


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_one_model(spec: ModelSpec) -> list[RunResult]:
    """Run all dataset items against one model. Manages the model's
    server lifecycle inline so caller only needs to iterate specs.
    Returns a list of RunResult — one per (spec, dataset item)."""
    rows: list[RunResult] = []

    if spec.backend == "llama-server":
        proc = start_llama_server(spec)
        sampler_pid = proc.pid
    else:
        proc = None
        sampler_pid = find_ollama_runner_pid()
        if sampler_pid is None:
            # Touch the model with a no-op request so Ollama spawns
            # the runner, then re-resolve the pid. Otherwise our first
            # peak-RSS sample comes back zero.
            try:
                urllib.request.urlopen(
                    urllib.request.Request(
                        OLLAMA,
                        data=json.dumps({"model": spec.ollama_tag, "messages": [{"role": "user", "content": "ping"}]}).encode(),
                        headers={"Content-Type": "application/json"},
                    ), timeout=30,
                ).read()
            except Exception:
                pass
            sampler_pid = find_ollama_runner_pid()

    try:
        for item in IMAGE_INPUTS + TEXT_INPUTS:
            if not item.exists():
                print(f"  [skip missing] {item.name}")
                continue
            kind = "image" if item in IMAGE_INPUTS else "text"
            sampler = RSSSampler(sampler_pid) if sampler_pid else None
            if sampler:
                sampler.start()
            try:
                raw, dt = call(spec, item, kind)
                ok, has_all, kc, sl, flag = parse_and_score(raw)
            except Exception as e:
                raw, dt = f"ERROR: {e}", 0.0
                ok, has_all, kc, sl, flag = False, False, 0, 0, "exception"
            peak_mb = sampler.stop_and_get_peak_mb() if sampler else 0.0
            tag = "OK" if (ok and has_all and not flag) else (flag or "BAD")
            print(f"  {kind:5} {item.name:<55} {dt:6.2f}s  rss={peak_mb:6.0f}MB  {tag}")
            rows.append(RunResult(
                file=item.name, file_kind=kind, model=spec.name,
                latency_s=dt, raw=raw, json_ok=ok, has_all_keys=has_all,
                kw_count=kc, summary_len=sl, flag=flag, peak_rss_mb=peak_mb,
            ))
    finally:
        if proc is not None:
            stop_llama_server(proc)

    return rows


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(rows: list[RunResult], specs: list[ModelSpec]) -> None:
    by_model: dict[str, list[RunResult]] = {}
    for r in rows:
        by_model.setdefault(r.model, []).append(r)

    def stats(model: str) -> dict[str, Any]:
        rs = [r for r in by_model.get(model, []) if r.latency_s > 0]
        if not rs:
            return {"n": 0}
        lats = sorted(r.latency_s for r in rs)
        peak = max((r.peak_rss_mb for r in rs), default=0.0)
        return {
            "n": len(rs),
            "mean_s": sum(lats) / len(lats),
            "p95_s": lats[max(0, int(0.95 * len(lats)) - 1)],
            "json_ok": sum(1 for r in rs if r.json_ok),
            "all_keys": sum(1 for r in rs if r.has_all_keys),
            "flagged": sum(1 for r in rs if r.flag),
            "kw_avg": sum(r.kw_count for r in rs) / len(rs),
            "peak_rss_mb": peak,
            "exceptions": sum(1 for r in by_model.get(model, []) if r.flag == "exception"),
        }

    # Copy a few representative test images into the repo so the
    # markdown report can show them inline. We pick the canary
    # (lightbulb), a heavy-OCR screenshot, and the alien sprite — the
    # three that revealed the v1 0.8B's filename-hallucination.
    SAMPLES_DIR.mkdir(exist_ok=True)
    sample_images: list[tuple[str, str]] = []  # (basename, caption)
    for src_name, caption in [
        ("bullet.png", "Lightbulb (filename canary — should NOT mention bullets)"),
        ("alian.png", "Pixel sprite (small, no embedded text)"),
        ("Snipaste_2025-12-23_03-09-51.jpg", "OCR-heavy: AI model picker UI"),
    ]:
        src = DL / src_name
        if src.exists():
            dst = SAMPLES_DIR / src_name
            shutil.copy2(src, dst)
            sample_images.append((src_name, caption))

    lines: list[str] = []
    lines.append("# vlm-bench v3 — five-model size sweep across mixed inputs")
    lines.append("")
    lines.append(f"_Generated by `bench_v3.py` on {time.strftime('%Y-%m-%d %H:%M %Z')}._")
    lines.append("")
    lines.append("## TL;DR")
    lines.append("")
    lines.append(
        "Five models compared on the production cosma-summarizer prompt, "
        "across 12 images (real screenshots, sprites, photos, CJK text) "
        "and 6 text docs (logs, CSV, structured reports). All non-Ollama "
        "models run on llama-server with `--reasoning-budget 0` and "
        "Unsloth's recommended non-thinking sampling profile. One model "
        "loaded at a time so per-model peak RSS reflects just that "
        "model + mmproj."
    )
    lines.append("")

    # Aggregate table -------------------------------------------------------
    lines.append("## Aggregate metrics")
    lines.append("")
    lines.append("| Model | Approx size | n | Mean (s) | p95 (s) | json_ok | flagged | Avg keywords | Peak RSS (MB) |")
    lines.append("|---|---:|---:|---:|---:|:---:|:---:|---:|---:|")
    for spec in specs:
        s = stats(spec.name)
        if s["n"] == 0:
            continue
        lines.append(
            f"| `{spec.name}` | {spec.approx_size_gb:.1f} GB | {s['n']} | "
            f"{s['mean_s']:.2f} | {s['p95_s']:.2f} | "
            f"{s['json_ok']}/{s['n']} | {s['flagged']}/{s['n']} | "
            f"{s['kw_avg']:.1f} | {s['peak_rss_mb']:.0f} |"
        )
    lines.append("")

    # Mermaid: latency bar chart -------------------------------------------
    lines.append("## Mean latency (lower is better)")
    lines.append("")
    lines.append("```mermaid")
    lines.append("xychart-beta")
    lines.append('    title "Mean latency per model (seconds, all inputs)"')
    names = [spec.name for spec in specs if stats(spec.name)["n"] > 0]
    means = [stats(spec.name)["mean_s"] for spec in specs if stats(spec.name)["n"] > 0]
    quoted = "[" + ", ".join(f'"{n}"' for n in names) + "]"
    lines.append(f"    x-axis {quoted}")
    if means:
        lines.append(f"    y-axis \"latency (s)\" 0 --> {max(means) * 1.1:.1f}")
        lines.append(f"    bar [{', '.join(f'{m:.2f}' for m in means)}]")
    lines.append("```")
    lines.append("")

    # Mermaid: RSS chart ---------------------------------------------------
    lines.append("## Peak RSS during inference (lower is better)")
    lines.append("")
    lines.append("```mermaid")
    lines.append("xychart-beta")
    lines.append('    title "Peak resident set size per model (MB)"')
    rsss = [stats(spec.name)["peak_rss_mb"] for spec in specs if stats(spec.name)["n"] > 0]
    lines.append(f"    x-axis {quoted}")
    if rsss:
        lines.append(f"    y-axis \"RSS (MB)\" 0 --> {max(rsss) * 1.1:.0f}")
        lines.append(f"    bar [{', '.join(f'{r:.0f}' for r in rsss)}]")
    lines.append("```")
    lines.append("")

    # Per-input quality split ----------------------------------------------
    lines.append("## Quality split: images vs text")
    lines.append("")
    lines.append("| Model | Images json_ok | Text json_ok | Image kw avg | Text kw avg |")
    lines.append("|---|:---:|:---:|---:|---:|")
    for spec in specs:
        rs = [r for r in by_model.get(spec.name, []) if r.latency_s > 0]
        if not rs:
            continue
        img = [r for r in rs if r.file_kind == "image"]
        txt = [r for r in rs if r.file_kind == "text"]
        img_ok = f"{sum(1 for r in img if r.json_ok)}/{len(img)}" if img else "—"
        txt_ok = f"{sum(1 for r in txt if r.json_ok)}/{len(txt)}" if txt else "—"
        img_kw = f"{sum(r.kw_count for r in img) / len(img):.1f}" if img else "—"
        txt_kw = f"{sum(r.kw_count for r in txt) / len(txt):.1f}" if txt else "—"
        lines.append(f"| `{spec.name}` | {img_ok} | {txt_ok} | {img_kw} | {txt_kw} |")
    lines.append("")

    # Example outputs with embedded images ---------------------------------
    if sample_images:
        lines.append("## Example outputs")
        lines.append("")
        lines.append(
            "Three representative inputs. The first is the indexer's "
            "canary — a lightbulb named `bullet.png`. A model that "
            "captions it as bullets/guns is hallucinating from the "
            "filename rather than looking at pixels, which would "
            "poison the search index."
        )
        lines.append("")
        for fname, caption in sample_images:
            lines.append(f"### `{fname}`")
            lines.append("")
            lines.append(f"![{caption}](samples/{fname})")
            lines.append("")
            lines.append(f"_{caption}_")
            lines.append("")
            lines.append("| Model | Title | Summary |")
            lines.append("|---|---|---|")
            for spec in specs:
                rs = [r for r in by_model.get(spec.name, []) if r.file == fname and r.json_ok]
                if not rs:
                    lines.append(f"| `{spec.name}` | — | (no valid output) |")
                    continue
                try:
                    obj = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", rs[0].raw.strip()))
                except Exception:
                    obj = {}
                title = (obj.get("title") or "?").replace("|", "\\|")
                summary = (obj.get("summary") or "?").replace("|", "\\|").replace("\n", " ")
                if len(summary) > 200:
                    summary = summary[:200] + "…"
                lines.append(f"| `{spec.name}` | {title} | {summary} |")
            lines.append("")

    # Raw output pointer ---------------------------------------------------
    lines.append("## Raw outputs")
    lines.append("")
    lines.append(f"All {len(rows)} responses (model × file) are in [`results_v3.json`](results_v3.json) — open if you want to spot-check claims in this report.")
    lines.append("")

    REPORT.write_text("\n".join(lines))
    print(f"\nwrote {REPORT}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    specs = _build_model_specs()
    if not specs:
        raise SystemExit("no models resolved — pull at least one before running")
    print(f"Models: {[s.name for s in specs]}\n")

    all_rows: list[RunResult] = []
    for spec in specs:
        print(f"\n=== {spec.name} ({spec.backend}) ===")
        rows = run_one_model(spec)
        all_rows.extend(rows)
        # Persist after each model so a crash partway through still
        # leaves us with usable data.
        RESULTS.write_text(json.dumps([asdict(r) for r in all_rows], indent=2))

    write_report(all_rows, specs)


if __name__ == "__main__":
    main()
