import os
from pathlib import Path


def get_model_name(tag: str) -> str:
    """Map a short model tag to the current concrete Anthropic model name."""
    model_names = {
        "haiku": "claude-haiku-4-5-20251001",
        "sonnet": "claude-sonnet-4-6",
        "opus": "claude-opus-4-7",
    }
    if tag not in model_names:
        raise ValueError(
            f"Unknown model tag: {tag!r}. Valid tags: {sorted(model_names)}."
        )
    return model_names[tag]


def get_llm_config(model: str = "sonnet", **overrides):
    # Newer Claude models (Opus 4.7+) reject `temperature` outright — it's
    # deprecated in favour of thinking-mode controls. We default to None and
    # only pass the parameter when the caller explicitly sets it.

    base_config = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    base_cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")

    config = {
        "model": get_model_name(model),
        "max_tokens": 4096,
        "temperature": None,
        "keys_dir": base_config / "ancestors" / "keys",
        "cache_dir": base_cache / "gene" / "llm",
    }
    config.update(overrides)
    return config
