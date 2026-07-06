# Genealogy Agentic AI

A minimal from-scratch agentic-AI harness built on the Anthropic SDK, driven
by a stateless `Turn` value object. Genealogy Q&A over GEDCOM family-tree
data is the driving use case. Deliberately avoids LangChain / LangGraph in
favour of a small, observable, testable core.

See [PROJECT.md](PROJECT.md) for the design writeup and rationale.

## Install

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```
uv sync
```

Put your Anthropic API key at `~/.config/gene/keys/anthropic`.

## Genealogy

Load a GEDCOM file into SQLite:

```
uv run python -m gene.genealogy.load <tag>
```

`<tag>` is the stem of a `.ged` file in `genealogy_data/` (gitignored —
put your own family exports there). The parser goes through
[ged4py](https://github.com/andy-z/ged4py); databases are written to
`gene/genealogy/db/<tag>.sqlite` (also gitignored). Passing an unknown
tag prints the available list.

Chat with the agent scoped to one loaded family:

```
uv run python -m gene.genealogy.chat <tag>
uv run python -m gene.genealogy.chat <tag> --ask "How many people are in this tree?"
uv run python -m gene.genealogy.chat <tag> --log
```

The agent's only tool is `run_query`, a guarded SELECT-only wrapper over
the SQLite DB (statement whitelist, ATTACH/PRAGMA blocked, 100-row cap,
5s wall-clock timeout via `conn.interrupt()`). The live schema is
reflected into the system prompt so the LLM writes SQL against the
actual columns. Turn logs (`--log`) land in
`logs/genealogy-<tag>-<timestamp>.jsonl` and open with `gene.agent.log_view`
(below). Unknown / not-yet-loaded tags exit 2 with a clean error message.

## Scripts

Interactive chat REPL (calculator tool wired in by default):

```
uv run python -m gene.agent.chat_loop --model sonnet
```

Model tags: `haiku`, `sonnet`, `opus`. Add `--verbose` to print a
per-turn stats summary (steps, tools called, tokens, cache hits).

Log each completed Turn as JSONL for later inspection:

```
uv run python -m gene.agent.chat_loop --log                        # auto: logs/chat-{timestamp}.jsonl
uv run python -m gene.agent.chat_loop --log path/to/session.jsonl  # explicit path
```

View a log file (content-focused by default). PATH defaults to the newest
`*.jsonl` in `logs/` when omitted:

```
uv run python -m gene.agent.log_view                               # newest log
uv run python -m gene.agent.log_view PATH                          # In / Out per turn
uv run python -m gene.agent.log_view PATH --trace                  # full per-step trace
uv run python -m gene.agent.log_view PATH --turn N                 # one turn (negatives ok)
uv run python -m gene.agent.log_view PATH --tail K                 # last K turns
```

The log is JSONL, one Turn per line — grep and jq work directly:

```
jq . logs/chat.jsonl | less
head -n 1 logs/chat.jsonl | jq .
```

`CachedAnthropic` demo (two calls, second one served from disk cache):

```
uv run python -m gene.agent.llm
```

## Evals

Both eval runners take a directory of suite modules — each `.py` file is a
suite exporting `CASES` (and, for TurnCases, `build_conversation(llm)` and
optionally `precheck() -> str | None` to skip when data isn't loaded).

Run every suite in a directory against one model:

```
uv run python -m gene.agent.evals gene/agent/eval_cases
uv run python -m gene.agent.evals gene/genealogy/eval_cases --model sonnet
uv run python -m gene.agent.evals gene/agent/eval_cases --suite basic --name simple_math
```

Run the full suite × config matrix and diff against saved baselines:

```
uv run python -m gene.agent.run_evals gene/agent/eval_cases
uv run python -m gene.agent.run_evals gene/agent/eval_cases --save       # overwrite baselines
uv run python -m gene.agent.run_evals gene/agent/eval_cases --no-cache   # bypass cache, record timings
```

Baselines mirror the input dir under `eval_results/<dir>/<suite>/<config>.json`.
Genealogy baselines (`eval_results/gene/genealogy/`) are gitignored since case
names can leak family info.

## Tests

```
uv run pytest
```

## Lint & format

```
uv run ruff check gene/
uv run ruff format gene/
```
