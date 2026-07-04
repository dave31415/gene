"""Anthropic Messages API wrapper with deterministic disk caching.

Thin orchestration only — key resolution, request build, and cache
mechanics live in `llm_utils`. Second runs of identical prompts are
free.

Patterned on menu_agent's `CachedOpenAI`, ported to Anthropic-native
shapes so we get the native tool_use / tool_result content blocks
downstream.
"""

import json
import logging
import time
from random import random
from typing import Any

from anthropic import Anthropic
from anthropic.types import Message

from gene.agent.config import get_llm_config
from gene.agent.llm_utils import Cache, build_request, resolve_api_key

log = logging.getLogger(__name__)


class CachedAnthropic:
    """Anthropic Messages API with deterministic disk cache.

    Stateless per call: no conversation history is retained between `send()`
    calls — the caller passes the full messages list each time. The stored
    fields (`cache`, `client`) exist only to hold expensive dependencies
    (open disk cache, HTTP connection pool), not to remember prior turns.
    For chat ergonomics, wrap this in a separate `Conversation`-style class.
    """

    def __init__(
        self,
        *,
        config=None,
        cache=None,
        verbose: bool = False,
        use_cache: bool = True,
    ):
        self.config = config if config is not None else get_llm_config()
        self.verbose = verbose

        if cache is None:
            self.cache = Cache(
                self.config["cache_dir"],
                serialize=lambda m: m.model_dump_json(),
                deserialize=Message.model_validate_json,
                use_cache=use_cache,
                verbose=verbose,
            )
        else:
            # if you want to supply the cache
            self.cache = cache

        self.client = Anthropic(api_key=resolve_api_key(self.config["keys_dir"]))

    def _call_api(self, request):
        return self.client.messages.create(**request)

    def send(
        self,
        *,
        messages: list[dict[str, Any]],
        system: str | list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
    ) -> tuple[Message, dict[str, Any]]:
        """Send a Messages API request through the cache. Returns (Message, meta).

        `messages` is a list of alternating user/assistant turns. Each turn has
        `role` ("user" or "assistant") and `content` (a string, or a list of
        content blocks: text, image, tool_use, tool_result). The simplest form
        is `[{"role": "user", "content": "hello"}]`. First turn must be "user".

        `meta` holds `{"cache_hit": bool, "request": dict}` — the request is
        the exact dict sent to the API (or looked up in cache), so callers
        recording observability don't have to reconstruct it. Meta is the
        extension point for future per-call info without another signature break.
        """
        request = build_request(self.config, messages, system, tools, tool_choice)
        msg, cache_hit = self.cache.call(self._call_api, request)
        return msg, {"cache_hit": cache_hit, "request": request}


def print_llm_response(msg: Message) -> None:
    """Print every field of the Message, unfolding nested pydantic objects."""
    print(json.dumps(msg.model_dump(mode="json"), indent=2))


def demo():
    llm = CachedAnthropic(verbose=True)

    rand_num = random()
    t0 = time.perf_counter()

    content = f"Say hello in exactly three words and print this random number {rand_num}."
    messages = [{"role": "user", "content": content}]

    r1, _ = llm.send(messages=messages)
    t1 = time.perf_counter()
    print(f"\n=== first call ({t1 - t0:.2f}s) ===")
    print(r1.content[0].text)

    t0 = time.perf_counter()
    r2, _ = llm.send(messages=messages)
    t1 = time.perf_counter()
    print(f"\n=== second call ({t1 - t0:.2f}s) ===")
    print(r2.content[0].text)

    print("\nFull response object\n----------------\n")
    print_llm_response(r2)


if __name__ == "__main__":
    demo()
