"""Named LLM configs swept by `evals`.

Each entry is a (name → config dict) pair. Starts with the three model
tags; add variants (max_tokens, temperature, ...) by adding new entries.
"""

from typing import Any

from gene.agent.config import get_llm_config


def get_eval_configs() -> dict[str, dict[str, Any]]:
    return {
        "haiku": get_llm_config(model="haiku"),
        "sonnet": get_llm_config(model="sonnet"),
        "opus": get_llm_config(model="opus"),
    }
