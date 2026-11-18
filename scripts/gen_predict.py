"""Run a (fine-tuned or base) model on data/test.jsonl, write one prediction file per example and print a score."""
import argparse, difflib, json, re
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max-new", type=int, default=2048)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--kind", default="auto", choices=["auto", "vision", "text"], help="vision | text | auto-detect")
ap.add_argument("--show", action="store_true", help="print each prediction")
args = ap.parse_args()

root, out = Path(args.data), Path(args.out)
out.mkdir(parents=True, exist_ok=True)
PROMPT = (root / "prompt.txt").read_text(encoding="utf-8")
rows = [json.loads(l) for l in (root / "test.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
if args.limit:
    rows = rows[:args.limit]
if not rows:
    raise SystemExit("ERROR: no test examples")

import torch
from PIL import Image
from model_kind import is_vision
VISION = is_vision(args.model, args.kind)
if VISION:
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForMultimodalLM as Auto
    except ImportError:
        from transformers import AutoModelForImageTextToText as Auto
    processor = AutoProcessor.from_pretrained(args.model)
else:
    from transformers import AutoTokenizer, AutoModelForCausalLM as Auto
    processor = AutoTokenizer.from_pretrained(args.model)
    if any(r["images"] for r in rows):
        raise SystemExit("ERROR: the test data has images but this model is text-only")
model = Auto.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto")
tok = getattr(processor, "tokenizer", processor)
stop = sorted({i for i in (tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id) if isinstance(i, int) and i >= 0})


def generate(r):
    text = f"{PROMPT}\n\n{r['text']}" if r["text"] else PROMPT
    if VISION:
        content = [{"type": "image", "image": Image.open(p).convert("RGB")} for p in r["images"]]
        content.append({"type": "text", "text": text})
    else:
        content = text
    inputs = processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                           tokenize=True, return_dict=True, return_tensors="pt",
                                           enable_thinking=False).to(model.device)
    with torch.no_grad():
        o = model.generate(**inputs, max_new_tokens=args.max_new, do_sample=False, eos_token_id=stop)
    return processor.decode(o[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).split("<think>")[0].strip()


def parse_json(t):
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t.strip())
    try:
        return json.loads(t)
    except Exception:
        return None


def field_acc(pred, gold):
    """Share of gold keys whose value matches exactly (flat comparison of top-level keys)."""
    if not isinstance(gold, dict) or not gold:
        return None
    return sum(isinstance(pred, dict) and pred.get(k) == v for k, v in gold.items()) / len(gold)


scores, valid, is_json = [], 0, 0
for r in rows:
    pred = generate(r)
    (out / f"{r['id']}{r['ext']}").write_text(pred, encoding="utf-8")
    if args.show:
        print(f"\n--- {r['id']} ---\n{pred}\n", flush=True)
    if r["ext"] == ".json":
        is_json += 1
        pj, gj = parse_json(pred), parse_json(r["answer"])
        valid += pj is not None
        s = field_acc(pj, gj)
        scores.append(s if s is not None else float(pj == gj))
    else:
        scores.append(difflib.SequenceMatcher(None, pred, r["answer"]).ratio())
    print(f"{r['id']}: score {scores[-1]:.2f}", flush=True)

kind = "field accuracy (exact value match per key)" if is_json else "text similarity (0-1)"
summary = {"examples": len(rows), "mean_score": sum(scores) / len(scores), "metric": kind}
if is_json:
    summary["valid_json"] = f"{valid}/{is_json}"
(out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("\nSUMMARY", json.dumps(summary), flush=True)
