"""Teacher via an AGENT (BYOA/ADK style, e.g. byoa-md-json): send each text/markdown file to the agent and save the
JSON it returns, one <stem>.json per input, optionally reduced to just the fields you ask for.

Request: POST <url>/run_async  {task_id, goal_id, capability, inputs:{<content-field>: text, focus: ...},
         timeout_secs, hmac_sig, timestamp}   then poll GET <url>/status/<job_id>   (--sync uses POST /run instead)
HMAC   : sha256(secret, "task_id:goal_id:capability:timestamp"), secret from the AGENT_SECRET env var.

What is saved (<stem>.json):
  * --fields empty  -> the FULL agent response:  {"success", "status_code", "response": <agent reply>, "error"}
                       (the agent reply holds status, duration_ms, outputs.structured, ...)
  * --fields given  -> only those parts. Each entry is a path from the top of that document, e.g.
                       response.outputs.structured          -> the structured object itself (single entry = value only)
                       response.outputs.structured.total_sgd, response.duration_ms
                       line_items.description               -> a bare field name is searched inside the document, and the
                       names are also sent to the agent as its `focus`, so it only extracts them.
                       Matching ignores case, spaces and underscores; lists are filtered element by element.
Raw agent responses are kept in <out>/_raw/. Failures and empty selections are listed in flagged.txt."""
import argparse, hashlib, hmac, json, os, re, sys, time, urllib.error, urllib.request, uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

UA = "finetune-studio/1.0"


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def parse_fields(text):
    """'a, b.c\\nd' -> nested tree {'a': {}, 'b': {'c': {}}, 'd': {}}  ({} = keep the whole value)."""
    tree = {}
    for tok in re.split(r"[,\n;]+", text or ""):
        tok = tok.strip().strip("\"'")
        if not tok:
            continue
        node = tree
        for seg in [s.strip() for s in tok.split(".") if s.strip()]:
            node = node.setdefault(seg, {})
    return tree


def project(node, tree):
    if isinstance(node, list):
        return [x for x in (project(i, tree) for i in node) if x not in ({}, None)]
    if not isinstance(node, dict):
        return node
    out = {}
    for k, v in node.items():
        for rk, sub in tree.items():
            if norm(rk) == norm(k):
                out[k] = v if not sub else project(v, sub)
    return out


def _find(node, tree):
    """project at this level; if nothing matches, look inside nested objects (first hit wins)."""
    res = project(node, tree)
    if res not in ({}, []):
        return res
    kids = node.values() if isinstance(node, dict) else node if isinstance(node, list) else []
    for v in kids:
        if isinstance(v, (dict, list)):
            r = _find(v, tree)
            if r not in ({}, []):
                return r
    return {}


def project_any(node, tree):
    """Each requested field is searched independently (root first, then nested), results merged."""
    out = {}
    for k, sub in tree.items():
        r = _find(node, {k: sub})
        if isinstance(r, dict):
            out.update(r)
        elif r:                                  # root is a list of records
            out = r if not out else out
    return out


ROOT_KEYS = {"success", "statuscode", "response", "error", "taskid", "agentid"}


def deep_merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            deep_merge(a[k], v)
        else:
            a[k] = v
    return a


def get_path(node, segs):
    """walk a dotted path (normalised names); lists are mapped element-wise. returns (found, value)."""
    for i, seg in enumerate(segs):
        if isinstance(node, list):
            vals = [get_path(x, segs[i:]) for x in node]
            vals = [v for f, v in vals if f]
            return (bool(vals), vals)
        if not isinstance(node, dict):
            return False, None
        hit = [k for k in node if norm(k) == norm(seg)]
        if not hit:
            return False, None
        node = node[hit[0]]
    return True, node


def select(doc, fields_text):
    """Returns (selected_value, missing_entries, agent_names) for the user's field entries."""
    entries = [t.strip().strip("\"'") for t in re.split(r"[,\n;]+", fields_text or "") if t.strip()]
    out, missing = None, []
    rooted = [e for e in entries if norm(e.split(".")[0]) in ROOT_KEYS]
    bare = [e for e in entries if e not in rooted]
    if len(entries) == 1 and rooted:
        found, val = get_path(doc, [x for x in entries[0].split(".") if x])
        if not found:
            return None, entries, []
        return val, [], []
    out = {}
    for e in rooted:
        found, val = get_path(doc, [x for x in e.split(".") if x])
        if not found:
            missing.append(e)
            continue
        node = cur = {}
        segs = [x for x in e.split(".") if x]
        for sg in segs[:-1]:
            cur = cur.setdefault(sg, {})
        cur[segs[-1]] = val
        deep_merge(out, node)
    if bare:
        tree = parse_fields(", ".join(bare))
        kept = project_any(doc, tree)
        if isinstance(kept, dict):
            deep_merge(out, kept)
        missing += missing_top(kept, tree)
    return out, missing, bare


def missing_top(res, tree):
    have = {norm(k) for k in res} if isinstance(res, dict) else set()
    return [k for k in tree if norm(k) not in have]


def http(method, url, body=None, timeout=60):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(), method=method,
                                 headers={"Content-Type": "application/json", "User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except Exception:
            return e.code, {"error": f"HTTP {e.code}"}


def call_agent(a, secret, text, focus):
    tid, gid, ts = str(uuid.uuid4()), str(uuid.uuid4()), int(time.time())
    sig = hmac.new(secret.encode(), f"{tid}:{gid}:{a.capability}:{ts}".encode(), hashlib.sha256).hexdigest()
    inputs = {a.content_field: text}
    if focus:
        inputs["focus"] = focus
    body = {"task_id": tid, "goal_id": gid, "capability": a.capability, "inputs": inputs,
            "timeout_secs": a.timeout, "hmac_sig": sig, "timestamp": ts}
    base = a.url.rstrip("/")
    if a.sync:
        code, res = http("POST", base + "/run", body, a.timeout + 30)
        if code != 200:
            raise RuntimeError(f"agent HTTP {code}: {json.dumps(res)[:300]}")
        return res
    code, res = http("POST", base + "/run_async", body, 60)
    if code != 202:
        raise RuntimeError(f"agent HTTP {code}: {json.dumps(res)[:300]}")
    job, deadline = res["job_id"], time.time() + a.timeout + 60
    while time.time() < deadline:
        time.sleep(3)
        code, st = http("GET", f"{base}/status/{job}", None, 30)
        if code == 200 and st.get("status") in ("done", "error"):
            if st["status"] == "error":
                raise RuntimeError(f"agent error: {st.get('error')}")
            return st["result"]
    raise RuntimeError("timed out waiting for the agent")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="folder with input text/markdown files (searched recursively)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", required=True, help="agent endpoint, e.g. https://md-json.zackloh.dev")
    ap.add_argument("--capability", default="text-extract")
    ap.add_argument("--content-field", default="md", help="name of the agent input that receives the file text")
    ap.add_argument("--in-ext", default="md", help="input file extension")
    ap.add_argument("--fields", default="", help="comma/newline separated field names to keep")
    ap.add_argument("--instructions", default="", help="extra text for the agent's `focus` input")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=600, help="per-document seconds")
    ap.add_argument("--sync", action="store_true", help="use POST /run instead of /run_async + polling")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    secret = os.environ.get("AGENT_SECRET", "")
    if not secret:
        sys.exit("AGENT_SECRET is not set (enter the agent secret in the UI).")
    entries = [t.strip() for t in re.split(r"[,\n;]+", a.fields) if t.strip()]
    bare_names = [e for e in entries if norm(e.split(".")[0]) not in ROOT_KEYS]
    focus = ""
    if bare_names:
        focus = (f"Extract ONLY these fields and nothing else: {', '.join(bare_names)}. Use exactly these names as the "
                 f"JSON keys (a dotted name such as a.b means key b inside a; repeated rows go in a list).")
    if a.instructions.strip():
        focus = (focus + " " + a.instructions.strip()).strip()

    src, out = Path(a.src).expanduser().resolve(), Path(a.out).expanduser().resolve()
    ext = "." + a.in_ext.lstrip(".")
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() == ext.lower())
    if a.limit:
        files = files[:a.limit]
    (out / "_raw").mkdir(parents=True, exist_ok=True)
    print(f"Found {len(files)} {ext} file(s). Output: {out}", flush=True)
    print(f"Saving: {', '.join(entries) if entries else 'the full agent response'}", flush=True)
    flagged = []

    def work(f):
        dst = out / f.relative_to(src).with_suffix(".json")
        if dst.exists():
            return f, "skipped", None
        dst.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            res = call_agent(a, secret, f.read_text(encoding="utf-8", errors="replace"), focus)
            (out / "_raw" / (f.stem + ".json")).write_text(json.dumps(res, indent=2, ensure_ascii=False))
            if res.get("status") not in (None, "success", "SUCCESS") and res.get("status", "").lower() != "success":
                raise RuntimeError(f"agent status {res.get('status')}: {res.get('error_msg')}")
            outs = res.get("outputs") or {}
            if outs.get("json_parse_failed") or ("structured" in outs and outs["structured"] is None):
                raise RuntimeError("agent returned no valid JSON (json_parse_failed)")
            doc = {"success": True, "status_code": 200, "response": res, "error": None}
            note = f"{time.time()-t0:.0f}s"
            data = doc
            if entries:
                data, miss, _ = select(doc, a.fields)
                if data in (None, {}, []):
                    raise RuntimeError("none of the requested fields were found in the agent output (see _raw/)")
                if miss:
                    note += f", not found: {', '.join(miss)}"
            dst.write_text(json.dumps(data, indent=2, ensure_ascii=False))
            return f, "ok", note
        except Exception as e:
            return f, "flag", f"error: {e}"

    done = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for f, status, note in ex.map(work, files):
            done += 1
            print(f"[{done}/{len(files)}] {status:7} {f.name}  {note or ''}", flush=True)
            if status == "flag":
                flagged.append((str(f), note))
    (out / "flagged.txt").write_text("\n".join(f"{p}\t{r}" for p, r in flagged), encoding="utf-8")
    print(f"\nDone. {len(flagged)} flagged file(s) listed in {out / 'flagged.txt'}", flush=True)


if __name__ == "__main__":
    main()
