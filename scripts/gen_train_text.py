# Text-only branch of gen_train.py (exec'd from it, so args/PROMPT/load_rows/user_text/FastLanguageModel are in scope).
import torch
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

if any(r["images"] for n in ("train", "val") for r in load_rows(n)):
    raise SystemExit("ERROR: the data has images but this model is text-only. Pick a vision-language model "
                     "(e.g. Qwen3-VL) or remove the images.")

model, tokenizer = FastLanguageModel.from_pretrained(
    args.model, load_in_4bit=False, max_seq_length=args.max_len)
tok = getattr(tokenizer, "tokenizer", tokenizer)
END = "<|im_end|>\n" if tok.convert_tokens_to_ids("<|im_end|>") not in (None, tok.unk_token_id) else (tok.eos_token or "")
PAD = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id


def encode(r):
    """prompt tokens are masked (-100); only the answer is trained on."""
    prompt = tok.apply_chat_template([{"role": "user", "content": user_text(r["text"])}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    a = tok(r["answer"] + END, add_special_tokens=False)["input_ids"]
    return {"input_ids": p + a, "labels": [-100] * len(p) + a, "n_answer": len(a)}


def make_ds(name):
    rows = load_rows(name)
    enc = [encode(r) for r in rows]
    keep = [e for e in enc if len(e["input_ids"]) <= args.max_len]
    if len(keep) < len(enc):
        print(f"{name}: skipped {len(enc) - len(keep)} example(s) longer than {args.max_len} tokens (raise Max length in Advanced)", flush=True)
    return Dataset.from_list(keep) if keep else None


def collate(batch):
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), PAD, dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        k = len(b["input_ids"])
        ids[i, :k], lab[i, :k], att[i, :k] = torch.tensor(b["input_ids"]), torch.tensor(b["labels"]), 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


train_ds, eval_ds = make_ds("train"), make_ds("val")
if train_ds is None:
    raise SystemExit("ERROR: no training examples (empty or all longer than Max length)")

if args.check_batch:
    ex = train_ds[0]
    ids, lab = torch.tensor(ex["input_ids"]), torch.tensor(ex["labels"])
    sup = ids[lab != -100]
    print(f"first example: total tokens {ids.shape[0]} | tokens trained on (answer) {int(sup.shape[0])} | limit {args.max_len}")
    print("\nTRAINED-ON TEXT, start:", repr(tok.decode(sup[:30])))
    print("TRAINED-ON TEXT, end:  ", repr(tok.decode(sup[-15:])))
    longest = max(load_rows("train"), key=lambda r: len(r["text"]) + len(r["answer"]))
    n = len(encode(longest)["input_ids"])
    print(f"\nlongest text example is about {n} tokens (limit {args.max_len})"
          + ("  <-- TOO LONG, raise Max length in Advanced" if n > args.max_len else ""))
    raise SystemExit

model = FastLanguageModel.get_peft_model(
    model, r=args.rank, lora_alpha=args.rank, lora_dropout=0, bias="none", random_state=3407,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth")
FastLanguageModel.for_training(model)

trainer = SFTTrainer(
    model=model, processing_class=tokenizer, data_collator=collate, train_dataset=train_ds, eval_dataset=eval_ds,
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
