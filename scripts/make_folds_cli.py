"""Build k cross-validation folds from teacher labels.
Per fold: N disjoint held-out inputs (symlinked into test_inputs/, labels copied into ground_truth/) and a dataset
built by YOUR command template (--build-cmd) from everything else.
Placeholders in --build-cmd: {source_dir} {labels_dir} {out} {test_list}  (test_list = file with held-out stems)."""
import argparse, random, shlex, shutil, subprocess, sys
from pathlib import Path

INPUT_EXT = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}

ap = argparse.ArgumentParser()
ap.add_argument("source_dir", help="input files (PDF/images), searched recursively")
ap.add_argument("labels_dir", help="teacher output folder (<stem>.<label-ext> per input)")
ap.add_argument("--out", required=True)
ap.add_argument("--k", type=int, default=3)
ap.add_argument("--n", type=int, default=50, help="held-out files per fold")
ap.add_argument("--seed", type=int, default=3407)
ap.add_argument("--label-ext", default="md")
ap.add_argument("--build-cmd", default="", help="dataset-build command template (empty = only prepare the splits)")
a = ap.parse_args()

src, lab, out = Path(a.source_dir), Path(a.labels_dir), Path(a.out)
ext = "." + a.label_ext.lstrip(".")
flagged = set()
if (lab / "flagged.txt").exists():   # teacher failures / truncated outputs: never use as labels
    flagged = {Path(l.split("\t")[0]).stem for l in (lab / "flagged.txt").read_text().splitlines() if l.strip()}
eligible = sorted({p.stem: p for p in src.rglob("*") if p.suffix.lower() in INPUT_EXT
                   and (lab / f"{p.stem}{ext}").exists() and p.stem not in flagged}.items())
print(f"{len(flagged)} flagged file(s) excluded", flush=True)
print(f"{len(eligible)} inputs have teacher labels", flush=True)
if len(eligible) < a.k * a.n:
    sys.exit(f"need at least k*n = {a.k * a.n} labelled inputs, have {len(eligible)}")
random.Random(a.seed).shuffle(eligible)

for k in range(1, a.k + 1):
    d = out / f"fold_{k}"
    test = sorted(eligible[(k - 1) * a.n: k * a.n])
    (d / "test_inputs").mkdir(parents=True, exist_ok=True)
    (d / "ground_truth").mkdir(exist_ok=True)
    (d / "test_list.txt").write_text("\n".join(s for s, _ in test) + "\n")
    for stem, f in test:
        t = d / "test_inputs" / f.name
        if not t.exists():
            t.symlink_to(f.resolve())
        shutil.copy(lab / f"{stem}{ext}", d / "ground_truth" / f"{stem}{ext}")
    if a.build_cmd.strip():
        cmd = a.build_cmd.format(**{n: shlex.quote(v) for n, v in {
            "source_dir": str(src), "labels_dir": str(lab), "out": str(d / "data"),
            "test_list": str(d / "test_list.txt")}.items()})
        r = subprocess.run(cmd, shell=True, executable="/bin/bash", capture_output=True, text=True)
        (d / "build_dataset.log").write_text(r.stdout + r.stderr)
        ok = r.returncode == 0
    else:
        ok = True
    print(f"fold_{k}: {'ok' if ok else 'dataset build FAILED, see ' + str(d / 'build_dataset.log')}", flush=True)
    if not ok:
        sys.exit(1)
print("DONE", flush=True)
