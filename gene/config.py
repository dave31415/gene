import os
from pathlib import Path


def get_llm_config():
    # Newer Claude models (Opus 4.7+) reject `temperature` outright — it's
    # deprecated in favour of thinking-mode controls. We default to None and
    # only pass the parameter when the caller explicitly sets it.

    base_config = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    base_cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")

    config = {
        "model": "claude-opus-4-7",
        "max_tokens": 4096,
        "temperature": None,
        "keys_dir": base_config / "ancestors" / "keys",
        "cache_dir": base_cache / "gene" / "llm",
    }

    return config
