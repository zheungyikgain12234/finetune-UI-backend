"""Cross-validation folds from teacher output. Generalised from make_folds.py.
For each fold k: N disjoint test pages (symlinked PDFs + ground-truth md); everything else is TRAIN
(built via build_dataset.py into <out>/fold_k/data)."""
import argparse, random, shutil, subprocess, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("pdf_dir")
ap.add_argument("md_dir", help="teacher output folder (<stem>.md per PDF)")
ap.add_argument("--out", required=True)
ap.add_argument("--k", type=int, default=3)
ap.add_argument("--n", type=int, default=50, help="test pages per fold")
ap.add_argument("--seed", type=int, default=3407)
ap.add_argument("--build-dataset", required=True, help="path to build_dataset.py")
a = ap.parse_args()

pdf, md, out = Path(a.pdf_dir), Path(a.md_dir), Path(a.out)
flagged = set()
if (md / "flagged.txt").exists():   # teacher failures / truncated outputs: never use as labels
    flagged = {Path(l.split("\t")[0]).stem for l in (md / "flagged.txt").read_text().splitlines() if l.strip()}
eligible = sorted({p.stem: p for p in pdf.rglob("*.pdf")
                   if (md / f"{p.stem}.md").exists() and p.stem not in flagged}.items())
print(f"{len(flagged)} flagged file(s) excluded", flush=True)
print(f"{len(eligible)} PDFs have teacher output", flush=True)
if len(eligible) < a.k * a.n:
    sys.exit(f"need at least k*n = {a.k * a.n} PDFs with teacher output, have {len(eligible)}")
random.Random(a.seed).shuffle(eligible)

for k in range(1, a.k + 1):
    d = out / f"fold_{k}"
    test = sorted(eligible[(k - 1) * a.n: k * a.n])
    (d / "test_pdfs").mkdir(parents=True, exist_ok=True)
    (d / "ground_truth_md").mkdir(exist_ok=True)
    (d / "test_list.txt").write_text("\n".join(s for s, _ in test) + "\n")
    for stem, src in test:
        t = d / "test_pdfs" / f"{stem}.pdf"
        if not t.exists():
            t.symlink_to(src.resolve())
        shutil.copy(md / f"{stem}.md", d / "ground_truth_md" / f"{stem}.md")
    r = subprocess.run([sys.executable, a.build_dataset, str(pdf), str(md), "--out", str(d / "data"),
                        "--test-list", str(d / "test_list.txt")], capture_output=True, text=True)
    (d / "build_dataset.log").write_text(r.stdout + r.stderr)
    print(f"fold_{k}: {'ok' if r.returncode == 0 else 'build_dataset FAILED, see ' + str(d / 'build_dataset.log')}", flush=True)
    if r.returncode:
        sys.exit(1)
print("DONE", flush=True)
