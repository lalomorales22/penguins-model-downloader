#!/usr/bin/env python3
"""Penguin's Model Downloader: a small local web app that downloads Hugging Face models
to this computer or an external drive, sorted by format.

    MODELS/GGUF/<org>/<repo>/          llama.cpp, Ollama, LM Studio
    MODELS/MLX/<org>/<repo>/           Apple Silicon (MLX)
    MODELS/SAFETENSORS/<org>/<repo>/   original weights (transformers, vLLM, ...)
    MODELS/OTHER/<org>/<repo>/         anything that fits none of the above
"""
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import storage  # noqa: E402

PORT = int(os.environ.get("MODELS_PORT", "8765"))
HERE = Path(__file__).resolve().parent
HF_BIN = shutil.which("hf", path=str(Path(sys.executable).parent)) or shutil.which("hf") or "hf"
FORMATS = ("GGUF", "MLX", "SAFETENSORS", "OTHER")
SAFETY_MARGIN = 2 * 1024**3  # always leave 2 GB free
REPO_RE = re.compile(r"^[A-Za-z0-9][\w.\-]*/[\w.\-]+$")
OTHER_WEIGHTS = (".bin", ".pt", ".pth", ".ckpt", ".onnx", ".h5", ".msgpack", ".ot", ".tflite", ".mlmodel")
QUANT_RE = re.compile(r"(?i)(?<![a-z0-9])((?:i\d-)?(?:q\d+(?:_[a-z0-9]+)*|iq\d+(?:_[a-z0-9]+)*|f16|f32|bf16|fp16|fp32))(?![a-z0-9])")
SPLIT_RE = re.compile(r"-\d{5}-of-\d{5}")
XET_ONLY_OVER = 50 * 1000 ** 3  # huggingface_hub refuses plain HTTP downloads above 50 GB

def cache_env(root):
    """Keep every byte of cache next to the models, never on some other disk."""
    return {
        "HF_HUB_CACHE": str(root / ".hf-cache" / "hub"),
        "HF_XET_CACHE": str(root / ".hf-cache" / "xet"),
        "HF_HUB_DISABLE_TELEMETRY": "1",
        # Plain HTTP downloads grow the partial file byte by byte, which is what
        # makes the progress bar and resume trustworthy. run_job turns xet back on
        # for files over 50 GB, which cannot be fetched any other way.
        "HF_HUB_DISABLE_XET": "1",
    }


from huggingface_hub import HfApi, get_token  # noqa: E402

api = HfApi()
jobs = []
jobs_lock = threading.Lock()
queue_event = threading.Event()


def drive_ok():
    return storage.ready()


def free_bytes():
    return shutil.disk_usage(storage.root()).free if drive_ok() else 0


def parse_query(q):
    """Return a repo id if the text looks like one (or a HF URL), else None."""
    q = q.strip()
    if q.startswith("hf.co/"):
        q = "https://" + q
    if "huggingface.co" in q or q.startswith("https://hf.co/"):
        if "://" not in q:
            q = "https://" + q
        parts = [p for p in urlparse(q).path.split("/") if p]
        if parts and parts[0] == "models":
            parts = parts[1:]
        if len(parts) >= 2:
            q = parts[0] + "/" + parts[1]
    return q if REPO_RE.match(q) else None


MEDIA_EXT = (".mp4", ".mov", ".webm", ".gif", ".png", ".jpg", ".jpeg", ".webp", ".avif", ".svg",
             ".wav", ".mp3", ".flac", ".pdf")
VERSION_RE = re.compile(r"(?i)(?<![a-z0-9])v(\d+(?:\.\d+)*)(?![0-9])")

GGUF_NOTES = {
    1: "Extremely squeezed. Quality suffers badly; avoid unless nothing else fits.",
    2: "Tiny. Quality drops a lot; only if nothing else fits.",
    3: "Small. Noticeably less accurate than the original.",
    4: "Good balance of size and quality. The usual pick.",
    5: "A little better than Q4, a little bigger.",
    6: "Very close to the original quality.",
    8: "Practically identical to the original. Big.",
    16: "Uncompressed original. Largest, and rarely worth it over Q8.",
    32: "Full 32-bit precision. Twice the size of the original with no visible gain.",
}

# pipeline tag -> (family, plain-English name, one-line explanation)
PIPELINES = {
    "text-generation": ("chat", "Chat / writing model", "Answers questions, writes and codes, like ChatGPT."),
    "image-text-to-text": ("chat", "Chat model that can see images", "Chats like ChatGPT and can also look at pictures you give it."),
    "any-to-any": ("chat", "Multi-purpose chat model", "Chats and can handle more than text, such as images or audio."),
    "text-to-image": ("media", "Image generator", "Makes pictures from a text description."),
    "image-to-image": ("media", "Image editor", "Changes or restyles a picture you give it."),
    "text-to-video": ("media", "Video generator", "Makes video clips from a text description."),
    "image-to-video": ("media", "Video generator", "Turns a picture (plus a description) into a video clip."),
    "video-to-video": ("media", "Video editor", "Changes or restyles a video you give it."),
    "text-to-3d": ("media", "3D model generator", "Makes 3D objects from a description."),
    "image-to-3d": ("media", "3D model generator", "Turns a picture into a 3D object."),
    "automatic-speech-recognition": ("audio", "Speech-to-text", "Writes down what is said in an audio recording."),
    "text-to-speech": ("audio", "Voice generator", "Reads text out loud in a generated voice."),
    "text-to-audio": ("audio", "Sound / music generator", "Makes music or sound effects from a description."),
    "feature-extraction": ("embed", "Embedding model", "Turns text into numbers for search tools. It does not chat."),
    "sentence-similarity": ("embed", "Embedding model", "Compares pieces of text for search tools. It does not chat."),
}


def about_model(pipeline, tags, fmts):
    family, kind, blurb = PIPELINES.get(pipeline, ("other", (pipeline or "AI model").replace("-", " ").capitalize(),
                                                   "A specialised model. Its page explains what it does."))
    comfy = "comfyui" in tags
    runs = {}
    for fmt in fmts:
        warn = ""
        if family == "chat":
            if fmt == "GGUF":
                soft, comp = "Ollama, llama.cpp, LM Studio", "Mac, Windows or Linux. Any graphics card, or none at all."
            elif fmt == "MLX":
                soft, comp = "LM Studio or mlx-lm", "Apple Silicon Macs only (M1 and newer)."
            else:
                soft, comp = "transformers, vLLM, text-generation-webui", "Best with an NVIDIA graphics card. Works on Macs, but slower."
                warn = "Ollama and llama.cpp cannot open this format directly. For those, look for a GGUF version of this model."
        elif family == "media":
            if fmt == "GGUF":
                soft = "ComfyUI with the GGUF add-on"
            elif fmt == "MLX":
                soft = "MLX apps on a Mac"
            else:
                soft = "ComfyUI" if comfy else "ComfyUI or diffusers"
            comp = "Apple Silicon Macs only (M1 and newer)." if fmt == "MLX" else \
                "Best with an NVIDIA graphics card. Macs can run it, but slower."
            warn = "This is not a chat model, so it will not run in Ollama or llama.cpp."
        else:
            soft = {"GGUF": "llama.cpp-based apps (check the model page)", "MLX": "MLX apps on a Mac"}.get(
                fmt, "transformers (check the model page for the exact app)")
            comp = "Apple Silicon Macs only (M1 and newer)." if fmt == "MLX" else "Mac, Windows or Linux."
            if family in ("audio", "embed"):
                warn = "This is not a chat model."
        runs[fmt] = {"software": soft, "computer": comp, "warn": warn}
    return {"family": family, "kind": kind, "blurb": blurb, "runs": runs}


def file_kind(name):
    low = name.lower()
    if low.endswith(".gguf"):
        return "gguf"
    if low.endswith(".safetensors"):
        return "safetensors"
    if low.endswith(OTHER_WEIGHTS):
        return "other"
    return "aux"


def weight_meta(name, kind, size):
    """Plain-English description of one weight file, worked out from its name."""
    base = Path(name).name
    low, path_low = base.lower(), name.lower()
    toks = set(re.split(r"[^a-z0-9]+", low))
    m = VERSION_RE.search(base)
    meta = {"role": "main", "tier": None, "pruned": False, "quant": "",
            "version": [int(x) for x in m.group(1).split(".")] if m else None,
            "need": int(size * 1.2 + 1.5e9)}
    title, notes = [], []
    if meta["version"]:
        title.append("v" + m.group(1))
    if kind == "gguf":
        if "mmproj" in low:
            meta.update(role="addon", need=0)
            return meta | {"title": "Vision add-on", "note": "Lets the chat model see images. Download it together with one main file."}
        q = QUANT_RE.search(base)
        if q:
            quant = q.group(1).upper()
            meta["quant"] = quant
            d = re.search(r"\d+", quant.split("-")[-1])
            tier = int(d.group()) if d else None
            if quant.startswith(("F", "BF")):
                tier = 32 if "32" in quant else 16
            meta["tier"] = tier
            title = [quant]
            notes.append(GGUF_NOTES.get(tier, "A compressed version."))
            if quant.startswith("IQ"):
                notes.append("Uses a newer compression method that keeps a bit more quality for its size.")
    else:
        if "lora" in toks or "lora" in path_low:
            meta.update(role="addon", need=0)
            return meta | {"title": "LoRA add-on", "note": "A small add-on, not a full model. It needs the base model too."}
        if "vae" in toks or "/vae" in "/" + path_low:
            meta.update(role="helper", need=0)
            return meta | {"title": "VAE helper", "note": "A helper file the main model uses to turn its output into pictures."}
        if any(t in path_low for t in ("text_encoder", "clip", "umt5", "t5xxl", "t5_xxl")):
            meta.update(role="helper", need=0)
            return meta | {"title": "Text encoder", "note": "A helper file the main model uses to read your prompt."}
        if toks & {"w4a8", "int4", "4bit", "nf4", "q4", "fp4"}:
            meta["tier"] = 4
            title.append("4-bit")
            notes.append("4-bit: the smallest, with some loss in quality.")
        elif toks & {"int8", "fp8", "8bit", "q8"} or "fp8" in low:
            meta["tier"] = 8
            title.append("8-bit")
            notes.append("8-bit: about half the size, with quality very close to full.")
        elif toks & {"bf16", "fp16", "f16", "half"}:
            meta["tier"] = 16
            title.append("Full quality")
            notes.append("Full quality (16-bit). The best this model can look, and the biggest.")
        elif toks & {"fp32", "f32"}:
            meta["tier"] = 32
            title.append("32-bit")
            notes.append("Full 32-bit precision. Twice the size with no visible gain.")
        if "pruned" in toks:
            meta["pruned"] = True
            title.append("pruned")
            notes.append("Pruned: slimmed down to save space, so slightly lower quality.")
        if toks & {"distill", "distilled", "turbo", "lightning", "lcm", "schnell"}:
            title.append("fast")
            notes.append("Fast version: quicker results in fewer steps.")
    meta["title"] = " · ".join(title) if title else Path(base).stem
    meta["note"] = " ".join(notes)
    return meta


def build_view(fmt, files, whole):
    """Group a repo's files for one format tab: choice / required / helper / extra."""
    has_st = any(file_kind(n) == "safetensors" for n, _ in files)
    groups = {}
    for name, size in files:
        kind = file_kind(name)
        low = name.lower()
        if fmt == "GGUF":
            if kind == "gguf":
                group = "choice"
            elif low in ("readme.md", "license", "license.md", "license.txt"):
                group = "helper"
            else:
                continue
        else:
            if kind == "gguf":
                continue
            if low.endswith(MEDIA_EXT) or low.endswith(".gitkeep") or "tfevents" in low or low.startswith("runs/") \
                    or (kind == "other" and has_st):
                group = "extra"
            elif kind == "safetensors":
                group = "required" if whole else "choice"
            elif kind == "other":
                group = "required"
            else:
                group = "helper"
        key = SPLIT_RE.sub("", name) if kind == "gguf" else name
        g = groups.setdefault(key, {"label": key, "kind": kind, "group": group, "files": [], "size": 0})
        g["files"].append(name)
        g["size"] += size
    items = []
    for g in groups.values():
        g["parts"] = len(g["files"])
        if g["group"] == "choice":
            g.update(weight_meta(g["label"], g["kind"], g["size"]))
        items.append(g)
    items.sort(key=lambda g: (-g["size"] if g["group"] == "choice" else g["size"]))
    mode = "choose" if any(g["group"] == "choice" for g in items) else "whole"
    need = 0
    if mode == "whole":
        need = int(sum(g["size"] for g in items if g["group"] == "required") * 1.2 + 1.5e9)
    return {"mode": mode, "items": items, "need": need}


def repo_scan(repo):
    info = api.model_info(repo, files_metadata=True)
    files = [(s.rfilename, s.size or 0) for s in info.siblings or [] if s.rfilename != ".gitattributes"]
    return info, files


def gate_state(info):
    """"" for an open model, otherwise "ok" (this account may download it),
    "login" (nobody is logged in) or "locked" (terms not accepted yet)."""
    if not info.gated:
        return ""
    if not get_token():
        return "login"
    try:
        api.auth_check(info.id)
        return "ok"
    except Exception:
        return "locked"


def repo_details(repo):
    info, files = repo_scan(repo)
    names = {n.lower() for n, _ in files}
    kinds = {file_kind(n) for n, _ in files}
    st = [n for n, _ in files if file_kind(n) == "safetensors"]
    tags = list(info.tags or [])
    is_mlx = "mlx" in tags or repo.lower().startswith("mlx-community/")
    available = []
    if "gguf" in kinds:
        available.append("GGUF")
    if is_mlx:
        available.append("MLX")
    elif "safetensors" in kinds:
        available.append("SAFETENSORS")
    if not available:
        available.append("OTHER")
    detected = "MLX" if is_mlx else "SAFETENSORS" if "safetensors" in kinds else available[0]
    # One model split over several files, or several alternative versions of it?
    whole = (is_mlx or len(st) <= 1 or "config.json" in names or "model_index.json" in names
             or any(SPLIT_RE.search(n) for n in st))
    return {
        "type": "repo",
        "repo": info.id,
        "gated": gate_state(info),
        "downloads": info.downloads or 0,
        "likes": info.likes or 0,
        "pipeline": info.pipeline_tag or "",
        "license": next((t.split(":", 1)[1] for t in tags if t.startswith("license:")), ""),
        "available": available,
        "detected": detected,
        "about": about_model(info.pipeline_tag or "", tags, available),
        "views": {f: build_view(f, files, whole) for f in available},
    }


def search(q, fmt):
    kw = {"search": q, "sort": "downloads", "limit": 30}
    flt = {"GGUF": "gguf", "MLX": "mlx", "SAFETENSORS": "safetensors"}.get(fmt)
    if flt:
        kw["filter"] = flt
    out = []
    for m in api.list_models(**kw):
        tags = m.tags or []
        out.append({
            "repo": m.id,
            "downloads": m.downloads or 0,
            "likes": m.likes or 0,
            "kind": PIPELINES.get(m.pipeline_tag or "", ("", "", ""))[1],
            "formats": [f for f, t in (("GGUF", "gguf"), ("MLX", "mlx"), ("SAFETENSORS", "safetensors")) if t in tags],
        })
    return {"type": "search", "results": out}


def dest_for(fmt, repo):
    org, name = repo.split("/", 1)
    return storage.root() / fmt / org / name


def job_done_bytes(job):
    dest = Path(job["dest"])
    done = 0
    for f, size in job["files"].items():
        try:
            done += min((dest / f).stat().st_size, size)
        except OSError:
            pass
    inc = dest / ".cache" / "huggingface" / "download"
    if inc.is_dir():
        for p in inc.rglob("*.incomplete"):
            if p.name.startswith("._"):
                continue
            try:
                done += p.stat().st_size
            except OSError:
                pass
    return min(done, job["total"])


def job_view(job):
    done = job["total"] if job["state"] == "done" else job_done_bytes(job)
    now = time.time()
    speed = 0
    if job["state"] == "running":
        s = job["samples"]
        s.append((now, done))
        while len(s) > 2 and now - s[0][0] > 20:
            s.pop(0)
        if len(s) > 1 and s[-1][0] > s[0][0]:
            speed = max(0, (s[-1][1] - s[0][1]) / (s[-1][0] - s[0][0]))
    return {k: job[k] for k in ("id", "repo", "format", "dest", "total", "state", "error")} | {
        "done": done, "speed": speed, "count": len(job["files"]),
    }


def write_model_info(job):
    path = Path(job["dest"]) / "model-info.json"
    data = {}
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        pass
    files = dict(data.get("files", {}))
    files.update(job["files"])
    data.update({
        "repo": job["repo"],
        "format": job["format"],
        "source": "https://huggingface.co/" + job["repo"],
        "downloaded": datetime.now().isoformat(timespec="seconds"),
        "files": files,
    })
    path.write_text(json.dumps(data, indent=2))


def run_job(job):
    dest = Path(job["dest"])
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [HF_BIN, "download", job["repo"], *job["files"], "--local-dir", str(dest)]
    if shutil.which("caffeinate"):  # macOS: keep the computer awake while downloading
        cmd = ["caffeinate", "-i", *cmd]
    env = os.environ | cache_env(Path(job["root"]))
    if max(job["files"].values()) > XET_ONLY_OVER:
        del env["HF_HUB_DISABLE_XET"]  # Hugging Face only serves files this big through xet
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                            start_new_session=os.name == "posix")
    job["proc"] = proc
    tail = b""
    while True:
        chunk = proc.stdout.read1(65536)
        if not chunk:
            break
        tail = (tail + chunk)[-6000:]
    code = proc.wait()
    job["proc"] = None
    if job["cancel"]:
        job["state"] = "cancelled"
    elif code == 0:
        write_model_info(job)
        job["state"] = "done"
    else:
        lines = [l.strip() for l in re.split(r"[\r\n]+", tail.decode("utf-8", "replace")) if l.strip()]
        lines = [l for l in lines if "%|" not in l and "new version of huggingface_hub" not in l and "hf.co/cli/install" not in l]
        job["error"] = "\n".join(lines[-6:]) or f"hf exited with code {code}"
        job["state"] = "error"


def worker():
    while True:
        queue_event.wait()
        with jobs_lock:
            job = next((j for j in jobs if j["state"] == "queued"), None)
            if job is None:
                queue_event.clear()
                continue
            job["state"] = "running"
        try:
            if not drive_ok():
                raise RuntimeError("The save location is not available. Is the drive plugged in?")
            run_job(job)
        except Exception as e:  # keep the queue alive whatever happens
            job["state"], job["error"] = "error", str(e)


def start_download(body):
    repo, fmt, files = body.get("repo", ""), body.get("format", ""), body.get("files") or []
    if not REPO_RE.match(repo) or fmt not in FORMATS or not files:
        raise ValueError("Pick a model, a format and at least one file.")
    if not drive_ok():
        raise ValueError("Choose where to save first. If you use an external drive, check it is plugged in.")
    sizes = dict(repo_scan(repo)[1])
    unknown = [f for f in files if f not in sizes]
    if unknown:
        raise ValueError("Not a file in this repo: " + unknown[0])
    chosen = {f: sizes[f] for f in files}
    dest = dest_for(fmt, repo)
    total = sum(chosen.values())
    probe = {"dest": str(dest), "files": chosen, "total": total}
    with jobs_lock:
        pending = sum(j["total"] - job_done_bytes(j) for j in jobs if j["state"] in ("queued", "running"))
        need = total - job_done_bytes(probe) + pending
        if need > free_bytes() - SAFETY_MARGIN:
            raise ValueError(f"Not enough room there: this needs {need / 1e9:.1f} GB and only "
                             f"{free_bytes() / 1e9:.1f} GB is free (I always leave 2 GB spare).")
        if any(j["dest"] == str(dest) and j["state"] in ("queued", "running") for j in jobs):
            raise ValueError("That model is already downloading. Wait for it to finish, then add more files.")
        job = {"id": f"{int(time.time() * 1000):x}", "repo": repo, "format": fmt, "dest": str(dest),
               "root": str(storage.root()), "files": chosen, "total": total, "state": "queued", "error": "", "cancel": False,
               "proc": None, "samples": []}
        jobs.append(job)
    queue_event.set()
    return job_view(job)


def cancel_job(job_id):
    with jobs_lock:
        job = next((j for j in jobs if j["id"] == job_id), None)
        if not job:
            return
        job["cancel"] = True
        if job["state"] == "queued":
            job["state"] = "cancelled"
        proc = job["proc"]
    if proc and proc.poll() is None:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                if os.name == "posix":
                    os.killpg(proc.pid, sig)
                else:
                    proc.kill()
            except (ProcessLookupError, OSError):
                break
            try:
                proc.wait(timeout=3)
                break
            except subprocess.TimeoutExpired:
                continue


def library():
    out = []
    if not drive_ok():
        return out
    for fmt in FORMATS:
        base = storage.root() / fmt
        if not base.is_dir():
            continue
        for org in sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith(".")):
            for repo in sorted(p for p in org.iterdir() if p.is_dir() and not p.name.startswith(".")):
                size = n = 0
                for root, dirs, files in os.walk(repo):
                    dirs[:] = [d for d in dirs if d != ".cache"]
                    for f in files:
                        if f.startswith("._") or f in ("model-info.json", ".DS_Store"):
                            continue
                        try:
                            size += (Path(root) / f).stat().st_size
                            n += 1
                        except OSError:
                            pass
                out.append({"format": fmt, "repo": f"{org.name}/{repo.name}", "path": str(repo),
                            "size": size, "files": n})
    return out


def reveal(path):
    p = Path(path).resolve()
    roots = {Path(j["root"]).resolve() for j in jobs} | ({storage.root().resolve()} if storage.root() else set())
    if not p.exists() or not any(r in (p, *p.parents) for r in roots):
        raise ValueError("Not a folder inside the models folder.")
    if storage.SYSTEM == "Windows":
        os.startfile(str(p))
    else:
        subprocess.Popen(["open" if storage.SYSTEM == "Darwin" else "xdg-open", str(p)])


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send_json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def local_only(self):
        host = (self.headers.get("Host") or "").split(":")[0]
        if host not in ("127.0.0.1", "localhost"):
            self.send_json({"error": "forbidden"}, 403)
            return False
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).hostname not in ("127.0.0.1", "localhost"):
            self.send_json({"error": "forbidden"}, 403)
            return False
        return True

    def do_GET(self):
        if not self.local_only():
            return
        url = urlparse(self.path)
        qs = parse_qs(url.query)
        try:
            if url.path == "/":
                data = (HERE / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
            elif url.path == "/vendor/three.module.min.js":
                data = (HERE / "vendor" / "three.module.min.js").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/javascript")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "max-age=86400")
                self.end_headers()
                self.wfile.write(data)
            elif url.path == "/api/status":
                with jobs_lock:
                    views = [job_view(j) for j in jobs]
                ok = drive_ok()
                usage = shutil.disk_usage(storage.root()) if ok else None
                self.send_json({"drive": ok, "path": str(storage.root() or ""), "free": usage.free if ok else 0,
                                "total": usage.total if ok else 0, "jobs": views})
            elif url.path == "/api/resolve":
                q = (qs.get("q") or [""])[0].strip()
                fmt = (qs.get("format") or [""])[0]
                if not q:
                    raise ValueError("Type a model name, a search term or a Hugging Face link.")
                repo = parse_query(q)
                if repo:
                    try:
                        self.send_json(repo_details(repo))
                        return
                    except Exception as e:
                        if "huggingface.co" in q:
                            raise ValueError(f"Could not open {repo}: {type(e).__name__}")
                self.send_json(search(q, fmt))
            elif url.path == "/api/locations":
                self.send_json(storage.locations())
            elif url.path == "/api/library":
                self.send_json({"models": library()})
            else:
                self.send_json({"error": "not found"}, 404)
        except ValueError as e:
            self.send_json({"error": str(e)}, 400)
        except Exception as e:
            self.send_json({"error": f"{type(e).__name__}: {e}"[:400]}, 500)

    def do_POST(self):
        if not self.local_only():
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            if self.path == "/api/download":
                self.send_json(start_download(body))
            elif self.path == "/api/cancel":
                cancel_job(body.get("id", ""))
                self.send_json({"ok": True})
            elif self.path == "/api/location":
                storage.set_root(body.get("path", ""))
                self.send_json({"ok": True})
            elif self.path == "/api/format/prepare":
                self.send_json(storage.format_prepare(body.get("mount", "")))
            elif self.path == "/api/format/run":
                with jobs_lock:
                    if any(j["state"] in ("queued", "running") for j in jobs):
                        raise ValueError("Wait for the downloads to finish (or cancel them) before formatting a drive.")
                root = storage.format_run(body.get("token", ""), body.get("confirm", ""), body.get("fs", ""), body.get("name", ""))
                self.send_json({"ok": True, "path": str(root)})
            elif self.path == "/api/reveal":
                reveal(body.get("path", ""))
                self.send_json({"ok": True})
            else:
                self.send_json({"error": "not found"}, 404)
        except ValueError as e:
            self.send_json({"error": str(e)}, 400)
        except Exception as e:
            self.send_json({"error": f"{type(e).__name__}: {e}"[:400]}, 500)


def main():
    url = f"http://127.0.0.1:{PORT}/"
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", PORT)) == 0:
            print("Already running, opening", url)
            webbrowser.open(url)
            return
    storage.load()
    threading.Thread(target=worker, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    where = storage.root() or "(not chosen yet)"
    print(f"Penguin's Model Downloader running at {url}\nSaving to {where}\nClose this window to stop it.")
    if not os.environ.get("MODELS_NO_BROWSER"):
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for j in jobs:
            cancel_job(j["id"])


if __name__ == "__main__":
    main()
