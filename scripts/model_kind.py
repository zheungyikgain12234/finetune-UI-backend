"""Decide whether a model needs the vision-language code path or the plain text-only one."""
from transformers import AutoConfig


def is_vision(model, kind="auto"):
    """kind: 'vision' | 'text' force a path; 'auto' looks for a vision_config in the model's config."""
    if kind in ("vision", "text"):
        return kind == "vision"
    cfg = AutoConfig.from_pretrained(model, trust_remote_code=True).to_dict()
    return "vision_config" in cfg or "visual" in cfg or "vision_tower" in str(cfg.get("architectures", "")).lower()
