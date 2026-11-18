"""Smoke test for the Student tab: load the model and ask it to say hello. Prints one JSON line:
{"ok": bool, "reply"|"error", "load_s", "gen_s", "loader"}. Run with the venv active."""
import argparse, json, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--max-new", type=int, default=32)
a = ap.parse_args()


def out(o):
    print("HELLO_RESULT " + json.dumps(o), flush=True)


try:
    import torch
    from transformers import AutoProcessor, AutoTokenizer
    t0 = time.time()
    model = proc = loader = None
    try:    # same loader gen_predict.py uses (vision-language models)
        try:
            from transformers import AutoModelForMultimodalLM as Auto
        except ImportError:
            from transformers import AutoModelForImageTextToText as Auto
        proc = AutoProcessor.from_pretrained(a.model)
        model = Auto.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto")
        loader = "vision-language"
    except Exception:   # text-only model (e.g. Qwen3-8B)
        from transformers import AutoModelForCausalLM
        proc = AutoTokenizer.from_pretrained(a.model)
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto")
        loader = "text-only"
    load_s = time.time() - t0
    tok = getattr(proc, "tokenizer", proc)
    msgs = [{"role": "user", "content": [{"type": "text", "text": "Say hello in one short sentence."}]
             if loader == "vision-language" else "Say hello in one short sentence."}]
    kw = dict(add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    try:
        inputs = proc.apply_chat_template(msgs, enable_thinking=False, **kw).to(model.device)
    except TypeError:
        inputs = proc.apply_chat_template(msgs, **kw).to(model.device)
    stop = sorted({i for i in (tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id) if isinstance(i, int) and i >= 0})
    t1 = time.time()
    with torch.no_grad():
        o = model.generate(**inputs, max_new_tokens=a.max_new, do_sample=False, eos_token_id=stop)
    reply = tok.decode(o[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).split("</think>")[-1].strip()
    out({"ok": bool(reply), "reply": reply or "(empty reply)", "load_s": round(load_s, 1),
         "gen_s": round(time.time() - t1, 1), "loader": loader})
except Exception as e:
    out({"ok": False, "error": f"{type(e).__name__}: {e}"[:600]})
    sys.exit(1)
