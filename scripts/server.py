"""Local backend for the Flutter UI (stdlib only). Runs jobs detached, serves logs, scans folds,
and proxies the model-list call (browsers block cross-origin calls to vLLM/OpenAI).
Usage: python3 <project>/scripts/server.py [--port 8765]   (binds to 127.0.0.1 only)"""
import argparse, io, json, os, re, shlex, shutil, signal, subprocess, sys, threading, time, urllib.error, urllib.request, zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse, parse_qs

ROOT = Path(os.environ.get("FT_ROOT") or os.environ.get("OVIS_FT_ROOT") or Path(__file__).resolve().parents[2])
STORE = ROOT / ".ui_jobs"
STORE.mkdir(exist_ok=True)
INDEX = STORE / "jobs.json"
WORK = ROOT / ".ui_work"          # one workspace per browser: uploads, dataset, trained model
WORK.mkdir(exist_ok=True)
# FT_PUBLIC=1: only the workspace API (/api/ws/...) works; endpoints that run arbitrary commands, list server
# folders or fetch arbitrary URLs are refused, and jobs are only visible to the workspace that started them.
PUBLIC = os.environ.get("FT_PUBLIC") == "1"
MAX_UPLOAD = int(os.environ.get("FT_MAX_UPLOAD_MB", "100")) * 2**20       # Cloudflare free/Pro cap request bodies at 100 MB
MAX_UNZIPPED = int(os.environ.get("FT_MAX_UNZIPPED_MB", "2000")) * 2**20
MAX_FILES = 20000
TEXT_EXT = {".md", ".txt", ".json", ".csv", ".html", ".xml", ".yaml", ".yml", ".tsv", ".log"}
UPLOAD_EXT = TEXT_EXT | {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".pdf"}
SLOTS = ("inputs", "labels", "test_inputs", "test_labels", "teacher_inputs")
STEPS = ("all", "check", "build", "check-train", "check-infer")
WID_RE = re.compile(r"[a-f0-9]{32}")
ALLOWED_FILE = Path(__file__).with_name("allowed_models.json")
TEACHERS_FILE = Path(os.environ.get("FT_TEACHERS_FILE") or Path(__file__).with_name("teachers.json"))
TEACHER_MAX_FILES = int(os.environ.get("FT_TEACHER_MAX_FILES", "300"))


def load():
    try:
        return json.loads(INDEX.read_text())
    except Exception:
        return []


def save(jobs):
    INDEX.write_text(json.dumps(jobs))


def alive(pid):
    """True only for a live process (a zombie that nobody reaped does not count)."""
    try:
        st = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        return st not in ("Z", "X")
    except (OSError, IndexError, TypeError):
        return False


def view(j):
    ec = None
    p = Path(j["dir"]) / "exit_code"
    if p.exists():
        try:
            ec = int(p.read_text().strip())
        except ValueError:
            pass
    running = ec is None and bool(alive(j.get("pid")))
    status = "running" if running else ("killed" if ec is None else "done" if ec == 0 else f"failed ({ec})")
    j = {**j, "running": running, "exitCode": ec, "status": status}
    if PUBLIC:      # do not leak server paths/commands to browsers
        j = {k: v for k, v in j.items() if k not in ("cmd", "dir", "pid")}
        j["cmd"] = ""
    return j


def start(name, cmd, env):
    jid = str(int(time.time() * 1000))
    d = STORE / jid
    d.mkdir()
    log, ex = d / "log.txt", d / "exit_code"
    log.write_text(f"$ {cmd}\n\n")
    wrapper = (f'cd "{ROOT}"; [ -f venv/bin/activate ] && source venv/bin/activate; '
               f'( {cmd} ) >> "{log}" 2>&1; echo $? > "{ex}"')
    p = subprocess.Popen(["bash", "-c", wrapper], cwd=ROOT, start_new_session=True,
                         env={**os.environ, **{k: v for k, v in env.items() if isinstance(v, str)}},
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    threading.Thread(target=p.wait, daemon=True).start()   # reap, so finished jobs don't linger as zombies
    job = {"id": jid, "name": name, "cmd": cmd, "dir": str(d), "startedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "pid": p.pid}
    jobs = load(); jobs.append(job); save(jobs)
    return job


def hello(model, prefix):
    """Load the model in a subprocess (frees the GPU afterwards) and ask it to say hello."""
    venv_py = ROOT / "venv/bin/python"
    cmd = [str(venv_py if venv_py.exists() else "python3"), str(Path(__file__).with_name("hello_model.py")), "--model", model]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": prefix + "\nHello test timed out after 15 min while loading the model."}
    line = next((l for l in reversed(r.stdout.splitlines()) if l.startswith("HELLO_RESULT ")), None)
    if not line:
        tail = "\n".join((r.stderr or r.stdout).strip().splitlines()[-4:])
        return {"ok": False, "error": f"{prefix}\nHello test crashed (exit {r.returncode}):\n{tail}"}
    d = json.loads(line[len("HELLO_RESULT "):])
    if not d.get("ok"):
        return {"ok": False, "error": f"{prefix}\nModel failed to answer: {d.get('error')}"}
    return {"ok": True, "ms": int((time.time() - t0) * 1000),
            "info": f"{prefix}\nModel replied ({d['loader']}, load {d['load_s']}s, answer {d['gen_s']}s): \"{d['reply']}\""}


def allowed_models():
    try:
        return json.loads(ALLOWED_FILE.read_text())
    except Exception:
        return []


def teachers():
    try:
        return json.loads(TEACHERS_FILE.read_text())
    except Exception:
        return []


def teacher_public(t):
    """What browsers may see about a teacher: never its URL or key."""
    env = t.get("secret_env") if t["kind"] == "agent" else t.get("key_env")
    return {"id": t["id"], "kind": t["kind"], "label": t.get("label", t["id"]), "vision": bool(t.get("vision")),
            "configured": bool(os.environ.get(env)) if env else True, "note": t.get("note")}


def get_teacher(tid, kind=None):
    t = next((x for x in teachers() if x["id"] == tid), None)
    if not t or (kind and t["kind"] != kind):
        raise ValueError("Pick a teacher from the list.")
    return t


def fetch_agent_card(base):
    hdr = {"User-Agent": "finetune-studio/1.0"}
    out = {"ok": False}
    try:
        base = base.rstrip("/")
        with urllib.request.urlopen(urllib.request.Request(base + "/health", headers=hdr), timeout=15) as r:
            out["health"] = r.read().decode("utf-8", "replace")[:200]
        with urllib.request.urlopen(urllib.request.Request(base + "/agent.md", headers=hdr), timeout=15) as r:
            md = r.read().decode("utf-8", "replace")
        m = re.search(r"```json\s*(\{.*?\})\s*```", md, re.S)
        card = json.loads(m.group(1)) if m else {}
        out.update({"ok": True, "name": card.get("display_name") or card.get("agent_id"),
                    "description": card.get("description"), "version": card.get("version"),
                    "capability": (card.get("capabilities") or ["text-extract"])[0],
                    "invokePath": card.get("invoke_path"), "inputs": card.get("input_schema") or {}})
    except Exception as e:
        out["error"] = str(e)
    return out


def teacher_test(tid):
    """Check an allowlisted teacher from the server (model: say hello; agent: health + card)."""
    t = get_teacher(tid)
    if t["kind"] == "agent":
        c = fetch_agent_card(t["url"])
        return {"ok": c["ok"], "info": (f"{c.get('name')} v{c.get('version')}: {c.get('description') or ''}"
                                        if c["ok"] else c.get("error", "unreachable"))}
    t0 = time.time()
    key = os.environ.get(t.get("key_env") or "", "")
    try:
        req = urllib.request.Request(
            t["base_url"].rstrip("/") + "/chat/completions",
            data=json.dumps({"model": t["model"], "max_tokens": 256, "temperature": 0,
                             "chat_template_kwargs": {"enable_thinking": False},
                             "messages": [{"role": "user", "content": "Say: hello world"}]}).encode(),
            headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})})
        with urllib.request.urlopen(req, timeout=60) as r:
            msg = json.load(r)["choices"][0]["message"]
        reply = (msg.get("content") or msg.get("reasoning_content") or "").strip() or "(connected, empty reply)"
        return {"ok": True, "info": f'Connected in {int((time.time() - t0) * 1000)} ms. Model replied: "{reply}"'}
    except Exception as e:
        return {"ok": False, "info": f"Teacher is not reachable right now: {type(e).__name__}"}


def ws_teacher_run(wid, body):
    w = ws_dir(wid)
    t = get_teacher(body.get("teacher"))
    if not (w / "teacher_inputs").is_dir():
        raise ValueError("Upload the input files (.zip) first.")
    if any(j["running"] for j in (view(x) for x in load()) if j["name"].endswith(":teacher")):
        raise ValueError("A teacher job is already running; wait for it to finish.")
    py = ROOT / "venv/bin/python"
    py = str(py) if py.exists() else "python3"
    out, env = w / "teacher_out", {}
    shutil.rmtree(out, ignore_errors=True)
    if t["kind"] == "model":
        prompt = (body.get("prompt") or "").strip()
        if not prompt or len(prompt) > 8000:
            raise ValueError("Write a prompt (max 8000 characters).")
        a = [py, str(Path(__file__).with_name("teacher_run.py")), str(w / "teacher_inputs"), "--out", str(out),
             "--base-url", t["base_url"], "--model", t["model"],
             "--dpi", str(clamp(body, "dpi", 72, 300, 200, int)),
             "--max-tokens", str(clamp(body, "max_tokens", 256, 16384, 8192, int)),
             "--workers", str(int(t.get("workers", 2))), "--limit", str(TEACHER_MAX_FILES),
             "--postprocess", "markdown" if body.get("markdown", True) else "none"]
        env = {"TEACHER_PROMPT": prompt, "TEACHER_API_KEY": os.environ.get(t.get("key_env") or "", "")}
        if body.get("thinking"):
            a.append("--allow-thinking")
    else:
        # A secret typed by the user wins over the server's; it goes to the job via env only (never argv, log or disk).
        given = body.get("secret")
        secret = given.strip() if isinstance(given, str) and given.strip() else os.environ.get(t.get("secret_env") or "", "")
        if not secret or len(secret) > 512:
            raise ValueError("Enter this agent's secret (the server has none configured for it).")
        fields, instr = (body.get("fields") or "")[:2000], (body.get("instructions") or "")[:2000]
        a = [py, str(Path(__file__).with_name("teacher_agent.py")), str(w / "teacher_inputs"), "--out", str(out),
             "--url", t["url"], "--capability", t.get("capability", "text-extract"),
             "--content-field", t.get("content_field", "md"), "--in-ext", t.get("in_ext", "md"),
             "--workers", str(int(t.get("workers", 2))), "--timeout", str(clamp(body, "timeout", 30, 900, 600, int)),
             "--limit", str(TEACHER_MAX_FILES), "--fields", fields, "--instructions", instr]
        env = {"AGENT_SECRET": secret}
    return start(f"ws:{wid}:teacher", shlex.join(a), env)


def use_teacher_labels(wid):
    """Student step: teacher inputs become the training inputs, teacher outputs the labels."""
    w = ws_dir(wid)
    src, lab = w / "teacher_inputs", w / "teacher_out"
    if not src.is_dir() or not lab.is_dir():
        raise ValueError("Run the teacher first.")
    for name, d in (("inputs", src), ("labels", lab)):
        shutil.rmtree(w / name, ignore_errors=True)
        shutil.copytree(d, w / name, ignore=shutil.ignore_patterns("_raw", "flagged.txt"))
    return {"ok": True}


def ws_dir(wid):
    if not WID_RE.fullmatch(wid or ""):
        raise ValueError("bad workspace id")
    return WORK / wid


def extract_zip(data, dest):
    """Safely unpack an uploaded zip into dest (replacing it): no path escapes, symlinks, hidden files or odd types."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise ValueError("That is not a valid .zip file.")
    infos = [i for i in zf.infolist() if not i.is_dir()]
    if len(infos) > MAX_FILES:
        raise ValueError(f"Too many files in the zip (max {MAX_FILES}).")
    if sum(i.file_size for i in infos) > MAX_UNZIPPED:
        raise ValueError(f"The zip expands to more than {MAX_UNZIPPED // 2**20} MB.")
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    base = tmp.resolve()
    types, skipped = {}, 0
    for i in infos:
        parts = PurePosixPath(i.filename).parts
        ext = PurePosixPath(i.filename).suffix.lower()
        if (not parts or parts[0] == "/" or ".." in parts or "__MACOSX" in parts or parts[-1].startswith(".")
                or ext not in UPLOAD_EXT or (i.external_attr >> 16) & 0o170000 == 0o120000):
            skipped += 1
            continue
        target = (tmp / Path(*parts)).resolve()
        if base not in target.parents:
            skipped += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(i) as src, open(target, "wb") as out:
            shutil.copyfileobj(src, out)
        types[ext] = types.get(ext, 0) + 1
    if not types:
        shutil.rmtree(tmp, ignore_errors=True)
        raise ValueError("No usable files in the zip. Allowed: " + " ".join(sorted(UPLOAD_EXT)))
    shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)
    return {"files": sum(types.values()), "skipped": skipped, "types": types}


def slot_info(d):
    if not d.is_dir():
        return None
    types = {}
    for f in d.rglob("*"):
        if f.is_file():
            types[f.suffix.lower()] = types.get(f.suffix.lower(), 0) + 1
    return {"files": sum(types.values()), "types": types}


def dir_size(d):
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) if d.is_dir() else 0


def ws_status(wid):
    w = ws_dir(wid)
    out = w / "out"
    lora, merged = out / "outputs" / "lora", out / "outputs" / "merged_16bit"
    summary = None
    try:
        summary = json.loads((out / "predictions" / "summary.json").read_text())
    except Exception:
        pass
    try:
        params = json.loads((w / "params.json").read_text())
    except Exception:
        params = None
    tout = w / "teacher_out"
    flagged = 0
    try:
        flagged = len([l for l in (tout / "flagged.txt").read_text().splitlines() if l.strip()])
    except Exception:
        pass
    teacher = {"files": sum(1 for f in tout.rglob("*") if f.is_file() and "_raw" not in f.parts and f.name != "flagged.txt"),
               "flagged": flagged, "mb": round(dir_size(tout) / 2**20, 1)} if tout.is_dir() else None
    return {"teacher": teacher, "slots": {s: slot_info(w / s) for s in SLOTS},
            "lora": lora.is_dir(), "loraMB": round(dir_size(lora) / 2**20, 1),
            "merged": merged.is_dir(), "mergedGB": round(dir_size(merged) / 2**30, 1),
            "summary": summary, "params": params}


def clamp(body, key, lo, hi, default, cast=float):
    try:
        v = cast(body.get(key, default))
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a number")
    if not lo <= v <= hi:
        raise ValueError(f"{key} must be between {lo} and {hi}")
    return v


def ws_run(wid, body):
    """Build the quick_finetune command on the server from validated fields (the client never sends a command)."""
    w = ws_dir(wid)
    step = body.get("step", "all")
    if step not in STEPS:
        raise ValueError("unknown step")
    entry = next((m for m in allowed_models() if m["id"] == body.get("model")), None)
    if not entry:
        raise ValueError("Pick a model from the list.")
    prompt = (body.get("prompt") or "").strip()
    if not prompt or len(prompt) > 4000:
        raise ValueError("Write an instruction (max 4000 characters).")
    if not (w / "inputs").is_dir():
        raise ValueError("Upload the inputs zip first.")
    epochs, lr = clamp(body, "epochs", 1, 30, 3), clamp(body, "lr", 1e-6, 1e-2, 1e-4)
    rank, max_len = clamp(body, "rank", 1, 256, 32, int), clamp(body, "max_len", 512, 32768, 8192, int)
    max_new, holdout = clamp(body, "max_new", 64, 8192, 2048, int), clamp(body, "holdout", 1, 50, 10)
    if any(j["running"] for j in (view(x) for x in load())
           if j["name"].startswith("ws:") and not j["name"].endswith(":teacher")):
        raise ValueError("A training job is already running; the GPU is shared, so wait for it to finish.")
    py = ROOT / "venv/bin/python"
    a = [str(py) if py.exists() else "python3", str(Path(__file__).with_name("quick_finetune.py")),
         "--model", entry.get("path") or entry["id"], "--prompt", prompt, "--inputs", str(w / "inputs")]
    if (w / "labels").is_dir():
        a += ["--labels", str(w / "labels")]
    if (w / "test_inputs").is_dir():
        a += ["--test-inputs", str(w / "test_inputs")]
        if (w / "test_labels").is_dir():
            a += ["--test-labels", str(w / "test_labels")]
    else:
        a += ["--holdout", str(holdout)]
    a += ["--epochs", str(epochs), "--lr", str(lr), "--rank", str(rank), "--max-len", str(max_len),
          "--max-new", str(max_new), "--out", str(w / "out"), "--step", step]
    if step == "all":
        shutil.rmtree(w / "out" / "outputs", ignore_errors=True)
        shutil.rmtree(w / "out" / "predictions", ignore_errors=True)
        (w / "params.json").write_text(json.dumps(
            {"model": entry["id"], "vision": bool(entry.get("vision")), "prompt": prompt, "epochs": epochs,
             "lr": lr, "rank": rank, "max_len": max_len}))
    return start(f"ws:{wid}:{step}", shlex.join(a), {})


def readme(params, summary):
    p = params or {}
    base, vision = p.get("model", "<base model>"), bool(p.get("vision"))
    loader = "AutoModelForImageTextToText" if vision else "AutoModelForCausalLM"
    score = f"\nScore on held-out test files: {json.dumps(summary)}\n" if summary else ""
    return f"""# Your fine-tuned LoRA adapter

Base model : {base}
Method     : LoRA (rank {p.get('rank', '?')}), {p.get('epochs', '?')} epoch(s), learning rate {p.get('lr', '?')}
{score}
This folder holds only the adapter. Apply it to the same base model:

## Use it directly (no merge)
```python
from transformers import {loader}, AutoTokenizer
from peft import PeftModel
import torch
base = {loader}.from_pretrained("{base}", dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base, "lora")          # "lora" = the adapter folder from this zip
tok = AutoTokenizer.from_pretrained("{base}")
```

## Merge into one standalone model
```python
model = model.merge_and_unload()
model.save_pretrained("merged"); tok.save_pretrained("merged")
```

## vLLM
`vllm serve {base} --enable-lora --lora-modules mine=lora`   (then request model "mine")

## IMPORTANT: use the same instruction as in training
The model learned to answer this instruction (also saved as prompt.txt). Put it before every input,
as one user message: instruction, a blank line, then the input text.

{p.get('prompt', '')}
"""


class _Stream:
    """Write-only wrapper so zipfile can stream straight to the HTTP socket (no big temp file)."""
    def __init__(self, w):
        self.w = w

    def write(self, b):
        self.w.write(b)
        return len(b)

    def flush(self):
        self.w.flush()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200, raw=False):
        b = obj.encode() if raw else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8" if raw else "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Workspace")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_OPTIONS(self):
        self._send({})

    def _owns(self, j):
        """Public mode: a browser only sees jobs started from its own workspace (header X-Workspace)."""
        return not PUBLIC or j["name"].startswith(f"ws:{self.headers.get('X-Workspace', '-')}:")

    def _download(self, wid, what):
        base = ws_dir(wid) / "out" / "outputs"
        src = ws_dir(wid) / "teacher_out" if what == "teacher" else base / ("lora" if what == "lora" else "merged_16bit")
        if what not in ("lora", "merged", "teacher") or not src.is_dir():
            return self._send({"error": "nothing to download yet"}, 404)
        name = {"lora": "lora-adapter", "merged": "merged-model", "teacher": "teacher-labels"}[what]
        self.send_response(200)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Disposition", f'attachment; filename="{name}.zip"')
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        st = ws_status(wid)
        with zipfile.ZipFile(_Stream(self.wfile), "w", zipfile.ZIP_STORED) as zf:
            for f in sorted(src.rglob("*")):
                if f.is_file() and not (what == "teacher" and "_raw" in f.relative_to(src).parts):
                    zf.write(f, f"{'lora' if what == 'lora' else 'merged-model' if what == 'merged' else 'labels'}/{f.relative_to(src)}")
            if what == "lora":
                zf.writestr("README.md", readme(st["params"], st["summary"]))
                zf.writestr("prompt.txt", (st["params"] or {}).get("prompt", ""))

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        parts = u.path.strip("/").split("/")
        if u.path == "/api/info":
            return self._send({"root": "" if PUBLIC else str(ROOT), "public": PUBLIC})
        if u.path == "/api/teachers":
            return self._send([teacher_public(t) for t in teachers()])
        if u.path == "/api/allowed-models":
            return self._send([{k: m.get(k) for k in ("id", "label", "vision", "note")} for m in allowed_models()])
        if len(parts) >= 4 and parts[:2] == ["api", "ws"]:
            try:
                if parts[3] == "status" and len(parts) == 4:
                    return self._send(ws_status(parts[2]))
                if parts[3] == "download" and len(parts) == 4:
                    return self._download(parts[2], q.get("what", "lora"))
            except ValueError as e:
                return self._send({"error": str(e)}, 400)
        if PUBLIC and (u.path in ("/api/ls", "/api/folds")):
            return self._send({"error": "disabled in public mode"}, 403)
        if u.path == "/api/jobs":
            return self._send([view(j) for j in load() if self._owns(j)])
        if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "log":
            j = next((j for j in load() if j["id"] == parts[2] and self._owns(j)), None)
            if not j:
                return self._send("not found", 404, raw=True)
            f = Path(j["dir"]) / "log.txt"
            if not f.exists():
                return self._send("(this job's log file no longer exists)", raw=True)
            n = int(q.get("bytes", 60000))
            with open(f, "rb") as fh:
                size = f.stat().st_size
                fh.seek(max(0, size - n))
                return self._send(fh.read().decode("utf-8", "replace"), raw=True)
        if u.path in ("/api/prompts", "/api/recipes"):
            kind = u.path.split("/")[-1]
            items = []
            for f in sorted((Path(__file__).parent / kind).glob("*")):
                try:
                    if kind == "prompts" and f.suffix == ".txt":
                        items.append({"name": f.stem.replace("_", " ").capitalize(), "text": f.read_text()})
                    elif kind == "recipes" and f.suffix == ".json":
                        items.append(json.loads(f.read_text()))
                except Exception:
                    pass
            return self._send(items)
        if u.path == "/api/ls":
            d = Path(q.get("path") or Path.home()).expanduser()
            while not d.is_dir() and d != d.parent:
                d = d.parent
            d = d.resolve()
            try:
                ents = sorted(d.iterdir(), key=lambda e: e.name.lower())
                dirs = [e.name for e in ents if e.is_dir() and not e.name.startswith(".")]
                pdfs = sum(1 for e in ents if e.suffix.lower() in (".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"))
                mds = sum(1 for e in ents if e.suffix.lower() == ".md")
            except PermissionError:
                dirs, pdfs, mds = [], 0, 0
            return self._send({"path": str(d), "parent": str(d.parent), "dirs": dirs, "pdfs": pdfs, "mds": mds})
        if u.path == "/api/folds":
            d = Path(q.get("path", ""))
            out = []
            if q.get("path") and d.is_dir():
                for f in sorted(d.glob("fold_*")):
                    if f.is_dir():
                        out.append({"path": str(f), "name": f.name, "hasData": (f / "data").is_dir(),
                                    "trained": (f / q.get("marker", "outputs")).exists(),
                                    "predicted": (f / "predictions").is_dir()})
            return self._send(out)
        self._send("not found", 404, raw=True)

    def do_POST(self):
        u = urlparse(self.path)
        parts = u.path.strip("/").split("/")
        n = int(self.headers.get("Content-Length", 0))
        if len(parts) == 5 and parts[:2] == ["api", "ws"] and parts[3] == "upload":   # raw zip body
            try:
                slot = parts[4]
                if slot not in SLOTS:
                    raise ValueError("unknown slot")
                if n > MAX_UPLOAD:
                    self.rfile.read(0)
                    return self._send({"error": f"Zip is larger than {MAX_UPLOAD // 2**20} MB."}, 413)
                w = ws_dir(parts[2])
                w.mkdir(exist_ok=True)
                return self._send({"ok": True, **extract_zip(self.rfile.read(n), w / slot)})
            except ValueError as e:
                return self._send({"error": str(e)}, 400)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return self._send({"error": "bad request body"}, 400)
        if len(parts) == 4 and parts[:2] == ["api", "ws"] and parts[3] == "run":
            try:
                return self._send(view(ws_run(parts[2], body)))
            except ValueError as e:
                return self._send({"error": str(e)}, 400)
        if len(parts) == 5 and parts[:2] == ["api", "ws"] and parts[3] == "teacher":
            try:
                ws_dir(parts[2])
                if parts[4] == "run":
                    return self._send(view(ws_teacher_run(parts[2], body)))
                if parts[4] == "test":
                    return self._send(teacher_test(body.get("teacher")))
                if parts[4] == "use-labels":
                    return self._send(use_teacher_labels(parts[2]))
            except ValueError as e:
                return self._send({"error": str(e)}, 400)
        if PUBLIC and not (len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "stop"):
            return self._send({"error": "disabled in public mode"}, 403)
        if parts == ["api", "jobs"]:
            return self._send(view(start(body["name"], body["cmd"], body.get("env", {}))))
        if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "stop":
            j = next((j for j in load() if j["id"] == parts[2] and self._owns(j)), None)
            if j and alive(j.get("pid")):
                def _kill(pid=j["pid"]):
                    for sig, wait in ((signal.SIGTERM, 8), (signal.SIGKILL, 0)):
                        try:
                            os.killpg(pid, sig)
                        except ProcessLookupError:
                            return
                        for _ in range(wait * 2):
                            time.sleep(0.5)
                            if not alive(pid):
                                return
                threading.Thread(target=_kill, daemon=True).start()
            return self._send({"ok": True})
        if parts == ["api", "agent-card"]:
            return self._send(fetch_agent_card(body["url"]))
        if parts == ["api", "mkdir"]:
            try:
                Path(body["path"]).mkdir(parents=True, exist_ok=True)
                return self._send({"ok": True})
            except Exception as e:
                return self._send({"error": str(e)}, 400)
        if parts == ["api", "test-connection"]:
            t0 = time.time()
            try:
                req = urllib.request.Request(
                    body["url"].rstrip("/") + "/chat/completions",
                    data=json.dumps({"model": body["model"], "max_tokens": 256, "temperature": 0,
                                     "messages": [{"role": "user", "content": "Say: hello world"}]}).encode(),
                    headers={"Content-Type": "application/json",
                             **({"Authorization": f"Bearer {body['key']}"} if body.get("key") else {})})
                with urllib.request.urlopen(req, timeout=60) as r:
                    msg = json.load(r)["choices"][0]["message"]
                reply = msg.get("content") or msg.get("reasoning_content") or msg.get("reasoning") or ""
                if not reply.strip():
                    reply = "(connected, but the model returned no text; it may be spending the token budget on thinking)"
                return self._send({"ok": True, "reply": reply, "ms": int((time.time() - t0) * 1000)})
            except urllib.error.HTTPError as e:
                return self._send({"ok": False, "error": f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:400]}"})
            except Exception as e:
                return self._send({"ok": False, "error": str(e)})
        if parts == ["api", "test-hf-model"]:
            mid = (body.get("model") or "").strip()
            if not mid:
                return self._send({"ok": False, "error": "Enter a model id first."})
            lp = Path(mid).expanduser()
            if lp.is_dir():
                if not (lp / "config.json").exists():
                    return self._send({"ok": False, "kind": "local", "error": f"Folder has no config.json: {lp}"})
                return self._send(hello(str(lp), f"Local folder found: {lp}"))
            if not re.fullmatch(r"[\w.\-]+/[\w.\-]+", mid):
                return self._send({"ok": False, "error": "Not a Hugging Face id. Expected org/name, e.g. Qwen/Qwen3-VL-8B-Instruct."})
            t0 = time.time()
            try:
                h = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {}
                req = urllib.request.Request(f"https://huggingface.co/api/models/{mid}?blobs=true", headers=h)
                with urllib.request.urlopen(req, timeout=20) as r:
                    m = json.load(r)
                gb = sum((f.get("size") or 0) for f in m.get("siblings", []) if f["rfilename"].endswith((".safetensors", ".bin"))) / 1e9
                tag = m.get("pipeline_tag") or "unknown"
                vision = tag in ("image-text-to-text", "image-to-text", "visual-question-answering") or "vision" in (m.get("tags") or [])
                info = [f"Found on Hugging Face: {mid}", f"Task: {tag}", f"Weights: about {gb:.1f} GB",
                        "Can read images: " + ("yes" if vision else "NO (looks text-only, not suitable for PDF/image fine-tuning)")]
                if m.get("gated"):
                    info.append("Gated: you must accept its licence on Hugging Face and set HF_TOKEN before training.")
                if m.get("gated") and not os.environ.get("HF_TOKEN"):
                    return self._send({"ok": True, "kind": "hf", "vision": vision, "gated": True,
                                       "info": "\n".join(info) + "\nHello test skipped (gated, no HF_TOKEN)."})
                res = hello(mid, "\n".join(info))
                return self._send({**res, "kind": "hf", "vision": vision, "gated": bool(m.get("gated"))})
            except urllib.error.HTTPError as e:
                msg = {401: "Not found, or private/gated: it needs an HF_TOKEN.", 404: "No such model on Hugging Face. Check the spelling."}.get(e.code, f"HTTP {e.code}")
                return self._send({"ok": False, "error": msg})
            except Exception as e:
                return self._send({"ok": False, "error": f"Could not reach Hugging Face: {e}"})
        if parts == ["api", "models"]:
            try:
                req = urllib.request.Request(body["url"].rstrip("/") + "/models",
                                             headers={"Authorization": f"Bearer {body['key']}"} if body.get("key") else {})
                with urllib.request.urlopen(req, timeout=15) as r:
                    data = json.load(r)
                return self._send({"models": sorted(m["id"] for m in data["data"])})
            except Exception as e:
                return self._send({"error": str(e)}, 502)
        self._send("not found", 404, raw=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    print(f"root={ROOT}  listening on http://127.0.0.1:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
