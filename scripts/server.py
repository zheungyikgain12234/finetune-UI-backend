"""Local backend for the Flutter UI (stdlib only). Runs jobs detached, serves logs, scans folds,
and proxies the model-list call (browsers block cross-origin calls to vLLM/OpenAI).
Usage: python3 ovis_ft_ui/scripts/server.py [--port 8765]   (binds to 127.0.0.1 only)"""
import argparse, json, os, signal, subprocess, threading, time, urllib.error, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(os.environ.get("OVIS_FT_ROOT") or Path(__file__).resolve().parents[2])
STORE = ROOT / ".ui_jobs"
STORE.mkdir(exist_ok=True)
INDEX = STORE / "jobs.json"


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
    return {**j, "running": running, "exitCode": ec, "status": status}


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


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200, raw=False):
        b = obj.encode() if raw else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8" if raw else "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_OPTIONS(self):
        self._send({})

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        parts = u.path.strip("/").split("/")
        if u.path == "/api/info":
            return self._send({"root": str(ROOT)})
        if u.path == "/api/jobs":
            return self._send([view(j) for j in load()])
        if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "log":
            j = next((j for j in load() if j["id"] == parts[2]), None)
            if not j:
                return self._send("not found", 404, raw=True)
            f = Path(j["dir"]) / "log.txt"
            n = int(q.get("bytes", 60000))
            with open(f, "rb") as fh:
                size = f.stat().st_size
                fh.seek(max(0, size - n))
                return self._send(fh.read().decode("utf-8", "replace"), raw=True)
        if u.path == "/api/ls":
            d = Path(q.get("path") or Path.home()).expanduser()
            while not d.is_dir() and d != d.parent:
                d = d.parent
            d = d.resolve()
            try:
                ents = sorted(d.iterdir(), key=lambda e: e.name.lower())
                dirs = [e.name for e in ents if e.is_dir() and not e.name.startswith(".")]
                pdfs = sum(1 for e in ents if e.suffix.lower() == ".pdf")
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
                                    "trained": (f / "outputs/merged_16bit").is_dir(),
                                    "predicted": (f / "predictions").is_dir()})
            return self._send(out)
        self._send("not found", 404, raw=True)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        parts = urlparse(self.path).path.strip("/").split("/")
        if parts == ["api", "jobs"]:
            return self._send(view(start(body["name"], body["cmd"], body.get("env", {}))))
        if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "stop":
            j = next((j for j in load() if j["id"] == parts[2]), None)
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
