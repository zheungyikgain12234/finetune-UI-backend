"""General fine-tune driver for the Student tab's Quick mode.
Teach a Hugging Face model to turn INPUT files into LABEL files (e.g. Markdown -> JSON, image -> text, PDF -> Markdown).
Inputs and labels are paired by file name (invoice_1.md  <->  invoice_1.json). Needs: a model id, an inputs folder,
a labels folder, an instruction prompt, and either a hold-out % or separate test folders.
Steps: build the dataset -> fine-tune (gen_train.py) -> run on the test files and score (gen_predict.py).
Run with the venv active."""
import argparse, json, random, re, sys, subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEXT_EXT = {".md", ".txt", ".json", ".csv", ".html", ".xml", ".yaml", ".yml", ".tsv", ".log"}
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--inputs", required=True, help="folder of input files (text, images or PDFs)")
ap.add_argument("--labels", default="", help="folder of label files, same file names (default: same folder as inputs)")
ap.add_argument("--test-inputs", default="", help="optional separate test inputs folder")
ap.add_argument("--test-labels", default="", help="optional separate test labels folder")
ap.add_argument("--holdout", type=float, default=10, help="percent of pairs held out as test when no test folder is given")
ap.add_argument("--prompt", default="", help="instruction shown to the model before each input")
ap.add_argument("--prompt-file", default="")
ap.add_argument("--epochs", type=float, default=3)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--max-len", type=int, default=8192, help="max tokens per example (input + answer)")
ap.add_argument("--max-new", type=int, default=2048, help="max tokens generated at test time")
ap.add_argument("--max-pages", type=int, default=3, help="PDF pages used per example")
ap.add_argument("--out", required=True)
ap.add_argument("--step", default="all", choices=["all", "check", "build", "check-train", "check-infer"])
a = ap.parse_args()

out = Path(a.out).expanduser().resolve()
prompt = Path(a.prompt_file).read_text(encoding="utf-8") if a.prompt_file else a.prompt
if not prompt.strip():
    sys.exit("ERROR: write an instruction first (what should the model do with each input?)")


def flagged(folder):
    f = folder / "flagged.txt"
    return {Path(l.split("\t")[0]).stem for l in f.read_text().splitlines() if l.strip()} if f.exists() else set()


def pairs(inp, lab, kind):
    inp = Path(inp).expanduser().resolve()
    lab = Path(lab).expanduser().resolve() if lab else inp
    for d, n in ((inp, "inputs"), (lab, "labels")):
        if not d.is_dir():
            sys.exit(f"ERROR: {kind} {n} folder not found: {d}")
    skip = flagged(inp) | flagged(lab)
    labels = {}
    for p in sorted(lab.rglob("*")):
        if p.is_file() and p.stem not in skip and p.name != "flagged.txt" and p.suffix.lower() in TEXT_EXT:
            labels.setdefault(p.stem, p)
    res, no_label, same = {}, 0, 0
    for p in sorted(inp.rglob("*")):
        ext = p.suffix.lower()
        if not p.is_file() or p.name == "flagged.txt" or p.stem in skip or ext not in TEXT_EXT | IMG_EXT | {".pdf"}:
            continue
        l = labels.get(p.stem)
        if l is None:
            no_label += 1
        elif l == p:
            same += 1    # an input that is its own label (same folder, same file): not a pair
        elif l.read_text(encoding="utf-8", errors="replace").strip():
            res[p.stem] = (p, l)
    print(f"{kind}: {len(res)} input/label pair(s); {no_label} input(s) without a label (skipped)", flush=True)
    if same:
        print(f"  note: {same} file(s) were skipped because input and label are the same file. "
              f"Inputs and labels need different extensions (e.g. .md and .json) or different folders.", flush=True)
    if not res:
        sys.exit(f"ERROR: no input/label pairs found for {kind}. Files are paired by name, e.g. a.md in the inputs "
                 f"folder and a.json in the labels folder.")
    exts = sorted({p.suffix.lower() for p, _ in res.values()}), sorted({l.suffix.lower() for _, l in res.values()})
    print(f"  input types: {', '.join(exts[0])}  ->  label types: {', '.join(exts[1])}", flush=True)
    return res


data = pairs(a.inputs, a.labels, "train")
test = {}
if a.test_inputs:
    test = pairs(a.test_inputs, a.test_labels or a.test_inputs, "test")
    both = sorted(set(data) & set(test))
    if both:
        sys.exit(f"ERROR: these names are in both train and test (data leak): {both[:10]}")
else:
    names = sorted(data)
    random.Random(3407).shuffle(names)
    n = max(1, round(len(names) * a.holdout / 100)) if len(names) > 1 else 0
    if n == 0:
        sys.exit("ERROR: need at least 2 pairs to hold out a test set.")
    test = {k: data.pop(k) for k in names[:n]}
    print(f"hold-out: {len(test)} test, {len(data)} train", flush=True)

if a.step == "check":
    print(f"\nOK: {len(data)} train and {len(test)} test pair(s) look usable, no overlap.", flush=True)
    sys.exit(0)


def make_row(stem, p, l):
    ans = l.read_text(encoding="utf-8", errors="replace").strip()
    ext = p.suffix.lower()
    row = {"id": stem, "answer": ans, "ext": l.suffix.lower(), "text": "", "images": []}
    if ext in TEXT_EXT:
        row["text"] = p.read_text(encoding="utf-8", errors="replace")
    elif ext in IMG_EXT:
        row["images"] = [str(p)]
    else:   # PDF -> page images
        import pymupdf
        (out / "data" / "images").mkdir(parents=True, exist_ok=True)
        doc = pymupdf.open(p)
        for i, pg in enumerate(doc):
            if i >= a.max_pages:
                break
            img = out / "data" / "images" / f"{stem}_p{i + 1}.png"
            if not img.exists():
                pg.get_pixmap(dpi=150).save(str(img))
            row["images"].append(str(img))
        doc.close()
    return row


def build():
    (out / "data").mkdir(parents=True, exist_ok=True)
    rng = random.Random(3407)
    train = [make_row(k, *v) for k, v in sorted(data.items())]
    rng.shuffle(train)
    val = train[:1] if len(train) >= 20 else []     # tiny validation set only when there is enough data
    train = train[len(val):]
    sets = {"train": train, "val": val, "test": [make_row(k, *v) for k, v in sorted(test.items())]}
    for name, rows in sets.items():
        (out / "data" / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""), encoding="utf-8")
    (out / "data" / "prompt.txt").write_text(prompt, encoding="utf-8")
    print({k: len(v) for k, v in sets.items()}, flush=True)


def run(title, cmd):
    print(f"\n=== {title} ===", flush=True)
    r = subprocess.run([str(c) for c in cmd])
    if r.returncode:
        sys.exit(f"ERROR: step '{title}' failed (exit {r.returncode}); see the log above")


py = sys.executable
train_cmd = [py, HERE / "gen_train.py", "--data", out / "data", "--model", a.model, "--epochs", a.epochs,
             "--lr", a.lr, "--rank", a.rank, "--max-len", a.max_len]

if a.step == "build":
    build()
    print(f"\nOK: dataset written to {out / 'data'}", flush=True)
    sys.exit(0)
if a.step == "check-train":
    build()
    run("Load the model and test one training example", train_cmd + ["--out", out / "outputs", "--check-batch"])
    print("\nOK: the model loads. Check the 'TRAINED-ON TEXT' lines above: they must be the start and end of your label "
          "(not your input).", flush=True)
    sys.exit(0)
if a.step == "check-infer":
    build()
    run(f"Run {a.model} on 1 test example (before fine-tuning)",
        [py, HERE / "gen_predict.py", "--data", out / "data", "--model", a.model, "--out", out / "smoke_predictions",
         "--max-new", min(a.max_new, 1024), "--limit", 1, "--show"])
    print(f"\nOK: inference works. Output in {out / 'smoke_predictions'}", flush=True)
    sys.exit(0)

build()
run(f"1/2 Fine-tune {a.model} for {a.epochs:g} epoch(s)", train_cmd + ["--out", out / "outputs"])
run("2/2 Run the new model on the test examples and score it",
    [py, HERE / "gen_predict.py", "--data", out / "data", "--model", out / "outputs" / "merged_16bit",
     "--out", out / "predictions", "--max-new", a.max_new])
print(f"\nDONE. Model: {out / 'outputs' / 'merged_16bit'}\nPredictions + scores: {out / 'predictions'}", flush=True)
