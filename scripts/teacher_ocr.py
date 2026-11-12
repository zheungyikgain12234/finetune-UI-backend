"""Teacher OCR over a folder of PDFs -> one <stem>.md each. Mirrors ~/qwen-vl/batch_convert.py:
same prompt (scripts/teacher_prompt.txt, copied verbatim), one page per request, temperature 0, thinking off,
continuation-row merge, <think>/auto-link cleanup, <!-- page N --> markers, flagged.txt for failures/truncation.
Works with any OpenAI-compatible endpoint. API key comes from the TEACHER_API_KEY env var."""
import argparse, base64, json, os, re, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent


def merge_continuation_rows(md):
    """Merge table rows that have exactly one non-empty cell and an empty first cell
    into the row above (joined with <br>), so multi-line items stay one row."""
    out = []
    in_table = False
    for line in md.split("\n"):
        st = line.strip()
        if not (st.startswith("|") and st.endswith("|")):
            in_table = False
            out.append(line)
            continue
        cells = [c.strip() for c in st[1:-1].split("|")]
        is_sep = all(re.fullmatch(r":?-{3,}:?", c) for c in cells if c) and any(cells)
        if not in_table:
            in_table = True
            out.append(line)
            continue
        prev = out[-1].strip() if out else ""
        prev_is_row = prev.startswith("|") and not is_sep
        filled = [i for i, c in enumerate(cells) if c]
        if (not is_sep and len(filled) == 1 and filled[0] > 0 and prev_is_row
                and len(out) >= 3 and out[-2].strip().startswith("|")
                and not re.fullmatch(r"\|[\s:|-]+\|", prev)):
            pc = [c.strip() for c in prev[1:-1].split("|")]
            if len(pc) == len(cells):
                i = filled[0]
                pc[i] = (pc[i] + "<br>" if pc[i] else "") + cells[i]
                out[-1] = "| " + " | ".join(pc) + " |"
                continue
        out.append(line)
    return "\n".join(out)


def render_pages(pdf, dpi):
    import pymupdf
    doc = pymupdf.open(pdf)
    try:
        return [p.get_pixmap(dpi=dpi).tobytes("png") for p in doc]
    finally:
        doc.close()


def ask(a, key, prompt, png):
    body = {"model": a.model, "max_tokens": a.max_tokens, "temperature": 0,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}},
                {"type": "text", "text": prompt}]}]}
    if not a.allow_thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(a.base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {key}"} if key else {})})
    for attempt in range(1, a.retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=a.timeout) as r:
                ch = json.load(r)["choices"][0]
            return ch["message"].get("content") or "", ch.get("finish_reason") == "length"
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}")
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == a.retries:
                raise
            time.sleep(5 * attempt)


def convert(a, key, prompt, pdf, out):
    parts, cut = [], False
    for png in render_pages(pdf, a.dpi):
        md, c = ask(a, key, prompt, png)
        cut = cut or c
        parts.append(md)
    if not parts or any(not x.strip() for x in parts):
        raise RuntimeError("empty or missing page output")
    parts = [merge_continuation_rows(x) for x in parts]
    md = "\n\n---\n\n".join(f"<!-- page {i+1} -->\n{x}" for i, x in enumerate(parts)) if len(parts) > 1 \
        else parts[0]
    md = re.sub(r"<think>.*?</think>", "", md, flags=re.S).strip()
    md = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1", md)   # remove auto-links
    if len(parts) == 1:
        md = f"<!-- page 1 -->\n{md}"
    out.write_text(md, encoding="utf-8")
    return len(parts), cut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="folder with PDFs (searched recursively)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-file", default=str(HERE / "teacher_prompt.txt"))
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--allow-thinking", action="store_true", help="do not send enable_thinking=false")
    a = ap.parse_args()
    key = os.environ.get("TEACHER_API_KEY", "")
    prompt = Path(a.prompt_file).read_text().strip()
    src, dst = Path(a.src).expanduser().resolve(), Path(a.out).expanduser().resolve()
    dst.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(p for p in src.rglob("*") if p.suffix.lower() == ".pdf")
    if a.limit:
        pdfs = pdfs[:a.limit]
    print(f"Found {len(pdfs)} PDF(s). Output: {dst}", flush=True)
    flagged = []

    def work(pdf):
        out = dst / pdf.relative_to(src).with_suffix(".md")
        if out.exists():
            return pdf, "skipped", None
        out.parent.mkdir(parents=True, exist_ok=True)
        t = time.time()
        try:
            pages, cut = convert(a, key, prompt, pdf, out)
            if cut:
                return pdf, "flag", "output cut off (raise Max tokens)"
            return pdf, "ok", f"{pages} page(s), {time.time()-t:.0f}s"
        except Exception as e:
            out.unlink(missing_ok=True)
            return pdf, "flag", f"error: {e}"

    done = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for pdf, status, note in ex.map(work, pdfs):
            done += 1
            print(f"[{done}/{len(pdfs)}] {status:7} {pdf.name}  {note or ''}", flush=True)
            if status == "flag":
                flagged.append((str(pdf), note))
    (dst / "flagged.txt").write_text("\n".join(f"{f}\t{r}" for f, r in flagged), encoding="utf-8")
    print(f"\nDone. {len(flagged)} flagged file(s) listed in {dst / 'flagged.txt'}", flush=True)


if __name__ == "__main__":
    main()
