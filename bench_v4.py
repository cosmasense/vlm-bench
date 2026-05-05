"""
vlm-bench v4 — three models × twelve curated files × three reps.

Goal: get statistically usable per-pair timings (mean ± stdev across
3 runs) so the latency gap between Qwen3-VL-2B, Qwen3.5-0.8B Q4_K_M,
and Qwen3.5-2B Q4_K_M is real signal, not run-to-run noise. The v3
bench used one rep per pair; this run is the apples-to-apples
follow-up at fixed dataset.

Dataset is in `dataset_v4/`, fetched by `dataset_v4/fetch.sh`. Four
categories of three:
  objects/      — single-object Unsplash photographs
  ui/           — Unsplash photos of code/dashboards/screens
  architecture/ — Unsplash architecture photographs
  pdfs/         — long arXiv papers (Attention/BERT/GPT-3),
                  text-extracted via `pdftotext` and truncated to
                  MAX_TEXT_CHARS, since cosma's pipeline summarizes
                  PDF text not page renders.

Each (model, file, rep) records:
  latency_s, peak_rss_mb, json_ok, has_all_keys, kw_count, summary_len, raw

Per (model, file) we then aggregate mean + stdev so the report
shows real bars, not points. Output:
  results_v4.json   — every individual rep
  REPORT3.md        — markdown report with mermaid charts grouped by
                      file-kind + per-pair-stdev tables + verdict.
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
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any
import urllib.request

from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HERE = Path(__file__).parent
DATA = HERE / "dataset_v4"
RESULTS = HERE / "results_v4.json"
REPORT = HERE / "REPORT3.md"
SAMPLES_DIR = HERE / "samples_v4"

OLLAMA = "http://localhost:11434/api/chat"
LCPP_PORT = 8001

MAX_IMG_SIDE = 1024
MAX_TEXT_CHARS = 8000

REPS_PER_PAIR = 3  # number of inference runs per (model, file)

# ---------------------------------------------------------------------------
# Production prompt — verbatim from cosma_backend/summarizer/base.py
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
# Inputs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Input:
    path: Path
    kind: str        # "image" or "text"
    category: str    # "objects" | "ui" | "architecture" | "pdfs"


def load_inputs() -> list[Input]:
    items: list[Input] = []
    for cat in ("objects", "ui", "architecture"):
        for p in sorted((DATA / cat).glob("*.jpg")):
            items.append(Input(p, "image", cat))
    for p in sorted((DATA / "pdfs").glob("*.pdf")):
        items.append(Input(p, "text", "pdfs"))
    if len(items) != 12:
        raise SystemExit(f"expected 12 dataset files, found {len(items)}")
    return items


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    name: str
    backend: str  # "ollama" or "llama-server"
    gguf_path: str | None = None
    mmproj_path: str | None = None
    ollama_tag: str | None = None
    approx_size_gb: float = 0.0


def _resolve_unsloth_blob(tag: str) -> tuple[str, str]:
    name, qtag = tag.split(":")
    manifest = Path.home() / f".ollama/models/manifests/hf.co/unsloth/{name}/{qtag}"
    data = json.loads(manifest.read_text())
    layers = {l["mediaType"]: l for l in data["layers"]}
    blob = lambda d: str(Path.home() / f".ollama/models/blobs/{d.replace('sha256:', 'sha256-')}")
    model = next(v for k, v in layers.items() if "model" in k)
    proj = next(v for k, v in layers.items() if "projector" in k)
    return blob(model["digest"]), blob(proj["digest"])


def _resolve_hf_dir(short: str) -> tuple[str, str] | None:
    """For /tmp/qwen35-models/<short>/ — find the Q4_K_M GGUF +
    matching mmproj. Returns None if either piece is missing."""
    d = Path("/tmp/qwen35-models") / short
    if not d.exists():
        return None
    ggufs = sorted(d.glob("**/*Q4_K_M*.gguf"), key=lambda p: p.stat().st_size, reverse=True)
    mm = next(iter(d.glob("**/mmproj-F16*.gguf")), None)
    if not ggufs or mm is None:
        return None
    return str(ggufs[0]), str(mm)


MODELS: list[ModelSpec] = []
MODELS.append(ModelSpec(
    name="qwen3-vl:2b-instruct", backend="ollama",
    ollama_tag="qwen3-vl:2b-instruct", approx_size_gb=1.9,
))
m, p = _resolve_unsloth_blob("Qwen3.5-0.8B-GGUF:Q4_K_M")
MODELS.append(ModelSpec(
    name="qwen3.5-0.8b-Q4_K_M", backend="llama-server",
    gguf_path=m, mmproj_path=p, approx_size_gb=0.5,
))
resolved = _resolve_hf_dir("2B")
if resolved is not None:
    m, p = resolved
    MODELS.append(ModelSpec(
        name="qwen3.5-2b-Q4_K_M", backend="llama-server",
        gguf_path=m, mmproj_path=p, approx_size_gb=1.2,
    ))


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class Run:
    file: str
    file_kind: str
    category: str
    model: str
    rep: int
    latency_s: float
    raw: str
    json_ok: bool
    has_all_keys: bool
    kw_count: int
    summary_len: int
    flag: str
    peak_rss_mb: float = 0.0


# ---------------------------------------------------------------------------
# Memory sampling (background ps poller)
# ---------------------------------------------------------------------------

class RSSSampler:
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
                    self.peak_kb = max(self.peak_kb, int(out.split()[0]))
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
    cmd = [
        "llama-server",
        "-m", spec.gguf_path,
        "--mmproj", spec.mmproj_path,
        "--port", str(LCPP_PORT),
        "--ctx-size", "4096",
        "-ngl", "99",
        "--jinja",
        "--reasoning-budget", "0",
        "--alias", spec.name,
    ]
    print(f"[start] {spec.name}")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )
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
    raise RuntimeError(f"{spec.name} failed to start")


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
    try:
        out = subprocess.check_output(["pgrep", "-f", "ollama runner"], text=True)
    except subprocess.CalledProcessError:
        return None
    pids = [int(p) for p in out.split()]
    return pids[0] if pids else None


# ---------------------------------------------------------------------------
# Input prep
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


_pdf_text_cache: dict[str, str] = {}


def _read_pdf_text(p: Path) -> str:
    """Extract text from a PDF via `pdftotext` and truncate. Cached
    so we don't re-extract on every rep."""
    key = str(p)
    if key in _pdf_text_cache:
        return _pdf_text_cache[key]
    try:
        out = subprocess.check_output(
            ["pdftotext", "-q", str(p), "-"], text=True, errors="replace",
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        out = f"[failed to extract PDF text: {e}]"
    if len(out) > MAX_TEXT_CHARS:
        head = out[: MAX_TEXT_CHARS - 1000]
        tail = out[-800:]
        out = f"{head}\n\n[...truncated for bench...]\n\n{tail}"
    _pdf_text_cache[key] = out
    return out


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _ollama_image_payload(model_id: str, p: Path) -> dict[str, Any]:
    return {
        "model": model_id,
        "messages": [
            {"role": "system", "content": VISUAL_PROMPT},
            {"role": "user",
             "content": f"filename: {p.name} /no_think",
             "images": [_b64_resized_image(p)]},
        ],
        "stream": False,
        "options": {"temperature": 0.7, "top_p": 0.8, "top_k": 20,
                    "presence_penalty": 1.5, "num_ctx": 4096},
    }


def _ollama_text_payload(model_id: str, p: Path, text: str) -> dict[str, Any]:
    return {
        "model": model_id,
        "messages": [
            {"role": "system", "content": TEXT_PROMPT},
            {"role": "user",
             "content": f"filename: {p.name}\n\n--- file content ---\n{text}\n--- end ---\n/no_think"},
        ],
        "stream": False,
        "options": {"temperature": 0.7, "top_p": 0.8, "top_k": 20,
                    "presence_penalty": 1.5, "num_ctx": 4096},
    }


def _oai_image_payload(model_id: str, p: Path) -> dict[str, Any]:
    suffix = p.suffix.lower().lstrip(".") or "jpeg"
    if suffix == "jpg":
        suffix = "jpeg"
    return {
        "model": model_id,
        "messages": [
            {"role": "system", "content": VISUAL_PROMPT},
            {"role": "user",
             "content": [
                 {"type": "text", "text": f"filename: {p.name}"},
                 {"type": "image_url",
                  "image_url": {"url": f"data:image/{suffix};base64,{_b64_resized_image(p)}"}},
             ]},
        ],
        "temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5,
        "stream": False,
    }


def _oai_text_payload(model_id: str, p: Path, text: str) -> dict[str, Any]:
    return {
        "model": model_id,
        "messages": [
            {"role": "system", "content": TEXT_PROMPT},
            {"role": "user",
             "content": f"filename: {p.name}\n\n--- file content ---\n{text}\n--- end ---"},
        ],
        "temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5,
        "stream": False,
    }


def call(spec: ModelSpec, item: Input) -> tuple[str, float]:
    if item.kind == "text":
        text = _read_pdf_text(item.path)
    if spec.backend == "ollama":
        url = OLLAMA
        if item.kind == "image":
            payload = _ollama_image_payload(spec.ollama_tag, item.path)
        else:
            payload = _ollama_text_payload(spec.ollama_tag, item.path, text)
    else:
        url = f"http://localhost:{LCPP_PORT}/v1/chat/completions"
        if item.kind == "image":
            payload = _oai_image_payload(spec.name, item.path)
        else:
            payload = _oai_text_payload(spec.name, item.path, text)
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

def run_for_model(spec: ModelSpec, items: list[Input]) -> list[Run]:
    rows: list[Run] = []
    if spec.backend == "llama-server":
        proc = start_llama_server(spec)
        sampler_pid = proc.pid
    else:
        proc = None
        sampler_pid = find_ollama_runner_pid()
        if sampler_pid is None:
            try:
                urllib.request.urlopen(
                    urllib.request.Request(
                        OLLAMA,
                        data=json.dumps({
                            "model": spec.ollama_tag,
                            "messages": [{"role": "user", "content": "ping"}],
                            "stream": False,
                        }).encode(),
                        headers={"Content-Type": "application/json"},
                    ), timeout=30,
                ).read()
            except Exception:
                pass
            sampler_pid = find_ollama_runner_pid()

    try:
        for item in items:
            for rep in range(REPS_PER_PAIR):
                sampler = RSSSampler(sampler_pid) if sampler_pid else None
                if sampler:
                    sampler.start()
                try:
                    raw, dt = call(spec, item)
                    ok, has_all, kc, sl, flag = parse_and_score(raw)
                except Exception as e:
                    raw, dt = f"ERROR: {e}", 0.0
                    ok, has_all, kc, sl, flag = False, False, 0, 0, "exception"
                peak_mb = sampler.stop_and_get_peak_mb() if sampler else 0.0
                tag = "OK" if (ok and has_all and not flag) else (flag or "BAD")
                print(f"  [{spec.name:<22} | {item.category:<13}] rep{rep+1} {item.path.name:<40} {dt:6.2f}s  {peak_mb:6.0f}MB  {tag}")
                rows.append(Run(
                    file=item.path.name, file_kind=item.kind, category=item.category,
                    model=spec.name, rep=rep, latency_s=dt, raw=raw,
                    json_ok=ok, has_all_keys=has_all, kw_count=kc,
                    summary_len=sl, flag=flag, peak_rss_mb=peak_mb,
                ))
    finally:
        if proc is not None:
            stop_llama_server(proc)
    return rows


# ---------------------------------------------------------------------------
# Aggregation + report
# ---------------------------------------------------------------------------

def _agg(values: list[float]) -> tuple[float, float]:
    """Return (mean, stdev). stdev=0 if only one sample."""
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def _write_report(all_rows: list[Run]) -> None:
    by_model: dict[str, list[Run]] = {}
    for r in all_rows:
        by_model.setdefault(r.model, []).append(r)

    SAMPLES_DIR.mkdir(exist_ok=True)
    # Copy one representative image per category so the report can
    # show side-by-side outputs.
    representatives: list[tuple[str, str, str]] = []  # (basename, category, caption)
    for cat in ("objects", "ui", "architecture"):
        first = sorted((DATA / cat).glob("*.jpg"))
        if first:
            src = first[0]
            shutil.copy2(src, SAMPLES_DIR / f"{cat}_{src.name}")
            representatives.append((f"{cat}_{src.name}", cat, src.name))
    # PDF can't be embedded; we just refer to it.

    L: list[str] = []
    L.append("# vlm-bench v4 — three models × twelve files × three reps")
    L.append("")
    L.append(f"_Generated by `bench_v4.py` on {time.strftime('%Y-%m-%d %H:%M %Z')}._")
    L.append("")
    L.append("## TL;DR")
    L.append("")

    # Aggregate latency / RSS / quality per model
    rows_for_summary: list[tuple[str, dict[str, Any]]] = []
    for spec in MODELS:
        rs = [r for r in by_model.get(spec.name, []) if r.latency_s > 0]
        if not rs:
            continue
        lats = [r.latency_s for r in rs]
        peak = max((r.peak_rss_mb for r in rs), default=0.0)
        json_ok = sum(1 for r in rs if r.json_ok)
        all_keys = sum(1 for r in rs if r.has_all_keys)
        flagged = sum(1 for r in rs if r.flag)
        rows_for_summary.append((spec.name, {
            "n": len(rs), "mean": statistics.mean(lats),
            "stdev": statistics.stdev(lats) if len(lats) > 1 else 0.0,
            "p95": sorted(lats)[max(0, int(0.95 * len(lats)) - 1)],
            "json_ok": json_ok, "all_keys": all_keys, "flagged": flagged,
            "peak": peak, "kw_avg": sum(r.kw_count for r in rs) / len(rs),
        }))

    L.append(
        "Three models on a 12-file curated dataset (3 single-object photos, 3 UI/code-screen photos, 3 architecture photographs, 3 long arXiv PDFs text-extracted), three reps per pair. Per-pair stdev now visible — single-rep timings from v3 had ~1-2 s of jitter that's now contained in error bars.")
    L.append("")

    # Aggregate metrics ---------------------------------------------------
    L.append("## Aggregate metrics (all 36 inputs)")
    L.append("")
    L.append("| Model | Approx size | n | Mean (s) | Stdev (s) | p95 (s) | json_ok | flagged | Avg keywords | Peak RSS (MB) |")
    L.append("|---|---:|---:|---:|---:|---:|:---:|:---:|---:|---:|")
    for name, s in rows_for_summary:
        L.append(
            f"| `{name}` | {next(m.approx_size_gb for m in MODELS if m.name == name):.1f} GB | "
            f"{s['n']} | {s['mean']:.2f} | {s['stdev']:.2f} | {s['p95']:.2f} | "
            f"{s['json_ok']}/{s['n']} | {s['flagged']}/{s['n']} | "
            f"{s['kw_avg']:.1f} | {s['peak']:.0f} |"
        )
    L.append("")

    # Mermaid: latency bar ------------------------------------------------
    L.append("## Mean latency per model")
    L.append("")
    L.append("```mermaid")
    L.append("xychart-beta")
    L.append('    title "Mean latency across all 36 (file × rep) cells (seconds)"')
    names = [n for n, _ in rows_for_summary]
    means = [s["mean"] for _, s in rows_for_summary]
    L.append('    x-axis [' + ", ".join(f'"{n}"' for n in names) + ']')
    if means:
        L.append(f'    y-axis "latency (s)" 0 --> {max(means) * 1.2:.1f}')
        L.append(f'    bar [{", ".join(f"{m:.2f}" for m in means)}]')
    L.append("```")
    L.append("")

    # Per-category mean latency -------------------------------------------
    L.append("## Latency by file category")
    L.append("")
    L.append("Mean latency per (model, category), averaging across both files in each category and reps. The interesting question: where does the small model save you time, and where does it just match the big model?")
    L.append("")
    L.append("| Model | objects | ui | architecture | pdfs |")
    L.append("|---|---:|---:|---:|---:|")
    for spec in MODELS:
        rs = [r for r in by_model.get(spec.name, []) if r.latency_s > 0]
        if not rs:
            continue
        cells: list[str] = [f"`{spec.name}`"]
        for cat in ("objects", "ui", "architecture", "pdfs"):
            in_cat = [r.latency_s for r in rs if r.category == cat]
            if in_cat:
                m, sd = _agg(in_cat)
                cells.append(f"{m:.2f} ± {sd:.2f}")
            else:
                cells.append("—")
        L.append("| " + " | ".join(cells) + " |")
    L.append("")

    # Mermaid: RSS --------------------------------------------------------
    L.append("## Peak RSS")
    L.append("")
    L.append("```mermaid")
    L.append("xychart-beta")
    L.append('    title "Peak RSS observed across the run (MB)"')
    L.append('    x-axis [' + ", ".join(f'"{n}"' for n in names) + ']')
    rss = [s["peak"] for _, s in rows_for_summary]
    if rss:
        L.append(f'    y-axis "RSS (MB)" 0 --> {max(rss) * 1.2:.0f}')
        L.append(f'    bar [{", ".join(f"{r:.0f}" for r in rss)}]')
    L.append("```")
    L.append("")

    # Quality by category -------------------------------------------------
    L.append("## Quality by file category")
    L.append("")
    L.append("`json_ok` rate and average keyword count per (model, category). `kw_avg` is a proxy for indexer richness — fewer keywords usually means weaker recall.")
    L.append("")
    L.append("| Model | objects (json_ok / kw) | ui (json_ok / kw) | architecture (json_ok / kw) | pdfs (json_ok / kw) |")
    L.append("|---|:---:|:---:|:---:|:---:|")
    for spec in MODELS:
        rs = [r for r in by_model.get(spec.name, []) if r.latency_s > 0]
        if not rs:
            continue
        cells = [f"`{spec.name}`"]
        for cat in ("objects", "ui", "architecture", "pdfs"):
            in_cat = [r for r in rs if r.category == cat]
            if not in_cat:
                cells.append("—")
                continue
            ok = sum(1 for r in in_cat if r.json_ok)
            kw = sum(r.kw_count for r in in_cat) / len(in_cat) if in_cat else 0
            cells.append(f"{ok}/{len(in_cat)} / {kw:.1f}")
        L.append("| " + " | ".join(cells) + " |")
    L.append("")

    # Per-pair stdev sanity check ----------------------------------------
    L.append("## Per-pair latency stability (stdev of 3 reps)")
    L.append("")
    L.append("If a pair's stdev is high (>1 s) the mean is noisy. Most should be small — Apple Silicon Metal inference is reasonably deterministic at temp 0.7.")
    L.append("")
    L.append("| File | qwen3-vl:2b | qwen3.5-0.8b Q4 | qwen3.5-2b Q4 |")
    L.append("|---|---|---|---|")
    items = sorted({r.file for r in all_rows})
    for fname in items:
        cells = [fname]
        for spec in MODELS:
            reps = [r.latency_s for r in all_rows if r.file == fname and r.model == spec.name and r.latency_s > 0]
            if reps:
                m, sd = _agg(reps)
                cells.append(f"{m:.2f} ± {sd:.2f}")
            else:
                cells.append("—")
        L.append("| " + " | ".join(cells) + " |")
    L.append("")

    # Example outputs (one per category) ---------------------------------
    L.append("## Example outputs (one input per category)")
    L.append("")
    for sample_name, cat, original_name in representatives:
        L.append(f"### {cat} — `{original_name}`")
        L.append("")
        L.append(f"![{cat}](samples_v4/{sample_name})")
        L.append("")
        L.append("| Model | Title | Summary |")
        L.append("|---|---|---|")
        for spec in MODELS:
            # First successful rep
            rs = [r for r in all_rows if r.file == original_name and r.model == spec.name and r.json_ok]
            if not rs:
                L.append(f"| `{spec.name}` | — | (no valid output) |")
                continue
            try:
                s = re.sub(r"^```(?:json)?\s*|\s*```$", "", rs[0].raw.strip())
                obj = json.loads(s)
            except Exception:
                obj = {}
            t = (obj.get("title") or "?").replace("|", "\\|")
            su = (obj.get("summary") or "?").replace("|", "\\|").replace("\n", " ")
            if len(su) > 220:
                su = su[:220] + "…"
            L.append(f"| `{spec.name}` | {t} | {su} |")
        L.append("")

    # PDF outputs (no image to embed) ------------------------------------
    pdf_files = sorted({r.file for r in all_rows if r.category == "pdfs"})
    L.append("### pdfs")
    L.append("")
    L.append("PDFs are text-extracted via `pdftotext` and truncated to 8000 chars before sending — that's how cosma's pipeline actually feeds them to the summarizer.")
    L.append("")
    for pdf in pdf_files:
        L.append(f"#### `{pdf}`")
        L.append("")
        L.append("| Model | Title | Summary |")
        L.append("|---|---|---|")
        for spec in MODELS:
            rs = [r for r in all_rows if r.file == pdf and r.model == spec.name and r.json_ok]
            if not rs:
                L.append(f"| `{spec.name}` | — | (no valid output) |")
                continue
            try:
                s = re.sub(r"^```(?:json)?\s*|\s*```$", "", rs[0].raw.strip())
                obj = json.loads(s)
            except Exception:
                obj = {}
            t = (obj.get("title") or "?").replace("|", "\\|")
            su = (obj.get("summary") or "?").replace("|", "\\|").replace("\n", " ")
            if len(su) > 220:
                su = su[:220] + "…"
            L.append(f"| `{spec.name}` | {t} | {su} |")
        L.append("")

    L.append("## Raw outputs")
    L.append("")
    L.append(f"All {len(all_rows)} runs (model × file × rep) are in [`results_v4.json`](results_v4.json).")
    L.append("")

    REPORT.write_text("\n".join(L))
    print(f"\nwrote {REPORT}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    items = load_inputs()
    print(f"Dataset: {len(items)} files")
    for it in items:
        print(f"  {it.category:<13} {it.kind:<5} {it.path.name}")
    print(f"\nModels: {[m.name for m in MODELS]}")
    print(f"Reps per pair: {REPS_PER_PAIR}")
    print(f"Total inferences: {len(items) * len(MODELS) * REPS_PER_PAIR}\n")

    all_rows: list[Run] = []
    for spec in MODELS:
        print(f"\n=== {spec.name} ({spec.backend}) ===")
        rows = run_for_model(spec, items)
        all_rows.extend(rows)
        RESULTS.write_text(json.dumps([asdict(r) for r in all_rows], indent=2))

    _write_report(all_rows)


if __name__ == "__main__":
    main()
