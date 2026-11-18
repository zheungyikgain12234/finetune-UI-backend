"""Generic LoRA fine-tune of a vision-language model on text-or-image -> text examples (data/{train,val}.jsonl, prompt.txt)."""
import argparse, json
from pathlib import Path
from PIL import Image
from datasets import Dataset
from unsloth import FastVisionModel, FastLanguageModel, is_bf16_supported   # must be imported first
from unsloth.trainer import UnslothVisionDataCollator
from model_kind import is_vision
from trl import SFTTrainer, SFTConfig

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=float, default=3)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--max-len", type=int, default=8192)
ap.add_argument("--kind", default="auto", choices=["auto", "vision", "text"],
                help="vision = vision-language model (images allowed); text = text-only LLM; auto = detect from the model config")
ap.add_argument("--check-batch", action="store_true", help="tokenize one example, print what is trained on, and exit")
args = ap.parse_args()

root = Path(args.data)
PROMPT = (root / "prompt.txt").read_text(encoding="utf-8")


def load_rows(name):
    p = root / f"{name}.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def user_text(text):
    return f"{PROMPT}\n\n{text}" if text else PROMPT


def to_messages(batch):
    out = []
    for text, images, ans in zip(batch["text"], batch["images"], batch["answer"]):
        content = [{"type": "image", "image": Image.open(p).convert("RGB")} for p in images]
        content.append({"type": "text", "text": user_text(text)})
        out.append([{"role": "user", "content": content},
                    {"role": "assistant", "content": [{"type": "text", "text": ans}]}])
    return {"messages": out}


VISION = is_vision(args.model, args.kind)
print(f"model kind: {'vision-language' if VISION else 'text-only'}", flush=True)
if not VISION:
    exec(open(Path(__file__).with_name("gen_train_text.py")).read())
    raise SystemExit
def make_ds(name):
    rows = load_rows(name)
    if not rows:
        return None
    ds = Dataset.from_list([{"text": r["text"], "images": r["images"], "answer": r["answer"]} for r in rows])
    ds.set_transform(to_messages)
    return ds


model, tokenizer = FastVisionModel.from_pretrained(
    args.model, load_in_4bit=False, use_gradient_checkpointing="unsloth", max_seq_length=args.max_len)


def response_marker():
    """Text that precedes the answer in this model's chat template (differs between model families)."""
    try:
        gen = tokenizer.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": "x"}]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        i = gen.rfind("<|im_start|>assistant")
        if i >= 0:
            return gen[i:]
    except Exception as e:
        print("could not detect the response marker:", e)
    return "<|im_start|>assistant\n"


RESPONSE_PART = response_marker()
print("response marker:", repr(RESPONSE_PART))

collator = UnslothVisionDataCollator(
    model, tokenizer, resize="max", max_seq_length=args.max_len,
    train_on_responses_only=True,          # loss only on the answer, not on the instruction/input
    instruction_part="<|im_start|>user\n", response_part=RESPONSE_PART)

train_ds, eval_ds = make_ds("train"), make_ds("val")
if train_ds is None:
    raise SystemExit("ERROR: data/train.jsonl is empty")

if args.check_batch:
    longest = max(load_rows("train"), key=lambda r: len(r["text"]) + len(r["answer"]))
    for label, ex in (("first example", train_ds[0]),):
        batch = collator([ex])
        ids, lab = batch["input_ids"][0], batch["labels"][0]
        tok = getattr(tokenizer, "tokenizer", tokenizer)
        sup = ids[lab != -100]
        print(f"{label}: total tokens {ids.shape[0]} | tokens trained on (answer) {int(sup.shape[0])} | limit {args.max_len}")
        print("\nTRAINED-ON TEXT, start:", repr(tok.decode(sup[:30])))
        print("TRAINED-ON TEXT, end:  ", repr(tok.decode(sup[-15:])))
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    n = len(tok(longest["text"] + longest["answer"])["input_ids"]) if longest["text"] else 0
    if n:
        print(f"\nlongest text example is about {n} tokens (limit {args.max_len})"
              + ("  <-- TOO LONG, raise Max length in Advanced" if n > args.max_len else ""))
    raise SystemExit

model = FastVisionModel.get_peft_model(
    model, finetune_vision_layers=False, finetune_language_layers=True,
    finetune_attention_modules=True, finetune_mlp_modules=True,
    r=args.rank, lora_alpha=args.rank, lora_dropout=0, bias="none", random_state=3407, target_modules="all-linear")
FastVisionModel.for_training(model)

trainer = SFTTrainer(
    model=model, processing_class=tokenizer, data_collator=collator, train_dataset=train_ds, eval_dataset=eval_ds,
    args=SFTConfig(
        per_device_train_batch_size=1, gradient_accumulation_steps=8, num_train_epochs=args.epochs,
        learning_rate=args.lr, warmup_ratio=0.1, lr_scheduler_type="linear", weight_decay=0.01,
        optim="adamw_8bit", seed=3407, bf16=is_bf16_supported(), fp16=not is_bf16_supported(),
        logging_steps=1, eval_strategy="epoch" if eval_ds else "no", save_strategy="no",
        output_dir=args.out, report_to="none", remove_unused_columns=False, dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True}, max_length=args.max_len))

trainer.train()
model.save_pretrained(f"{args.out}/lora")
tokenizer.save_pretrained(f"{args.out}/lora")
model.save_pretrained_merged(f"{args.out}/merged_16bit", tokenizer, save_method="merged_16bit")
print("Saved LoRA to", f"{args.out}/lora", "and merged model to", f"{args.out}/merged_16bit")
