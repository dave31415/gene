"""Helpers for the Anthropic wrapper: key resolution, request build, disk cache.

Split out of `llm.py` so `CachedAnthropic` stays a thin orchestration
class. Nothing in here imports the Anthropic SDK; `Cache` is generic.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import diskcache

log = logging.getLogger(__name__)


class LlmConfigError(RuntimeError):
    """Raised when the API key cannot be located."""


def resolve_api_key(keys_dir: Path) -> str:
    """Find the Anthropic API key."""
    key_file = keys_dir / "anthropic"
    if not key_file.exists():
        raise LlmConfigError(f"No Anthropic API key found. Write the key to {key_file}")

    key = key_file.read_text().strip()
    if not key:
        raise LlmConfigError(f"Anthropic API key file is empty: {key_file}")
    return key


def cache_key(request: dict[str, Any]) -> str:
    """Canonical SHA256 of the request. Same input ⇒ same key, every run."""
    payload = json.dumps(request, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_request(config, messages, system=None, tools=None, tool_choice=None):
    request = {"model": config["model"], "messages": messages, "max_tokens": config["max_tokens"]}

    if config["temperature"] is not None:
        request["temperature"] = config["temperature"]
    if system is not None:
        request["system"] = system
    if tools is not None:
        request["tools"] = tools
    if tool_choice is not None:
        request["tool_choice"] = tool_choice

    return request


class Cache:
    """Deterministic disk cache around a single-argument function.

    `call(func, arg)` hashes `arg`, returns the cached result on hit,
    otherwise calls `func(arg)`, stores the result, and returns it.
    Values round-trip through the caller-supplied `serialize` /
    `deserialize` pair.
    """

    def __init__(
        self,
        cache_dir: Path,
        *,
        serialize,
        deserialize,
        use_cache: bool = True,
        verbose: bool = False,
    ):
        self.serialize = serialize
        self.deserialize = deserialize
        self.verbose = verbose
        if use_cache:
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._store: diskcache.Cache | None = diskcache.Cache(str(cache_dir))
        else:
            self._store = None

    def call(self, func, arg: dict):
        if self._store is None:
            return func(arg)

        key = cache_key(arg)
        hit = self._store.get(key)
        if hit is not None:
            if self.verbose:
                log.debug("cache hit: %s", key[:12])
            return self.deserialize(hit)

        result = func(arg)
        self._store[key] = self.serialize(result)
        if self.verbose:
            log.debug("cache miss, stored: %s", key[:12])
        return result
