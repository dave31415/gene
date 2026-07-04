# Genealogy Agentic AI

Agentic AI for answering questions from a Gedcom file.

## Install

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```
uv sync
```

Put your Anthropic API key at `~/.config/ancestors/keys/anthropic`.

## Scripts

Interactive chat REPL (calculator tool wired in by default):

```
uv run python -m gene.chat_loop --model sonnet
```

Model tags: `haiku`, `sonnet`, `opus`. Add `--verbose` to print a
per-turn summary (steps, tools called, tokens, cache hits).

`CachedAnthropic` demo (two calls, second one served from disk cache):

```
uv run python -m gene.llm
```

## Evals

Run one suite:

```
uv run python -m gene.evals --suite basic --model sonnet
```

Run the full suite × config matrix and diff against saved baselines:

```
uv run python -m gene.run_evals
uv run python -m gene.run_evals --save        # overwrite baselines
uv run python -m gene.run_evals --no-cache    # bypass cache, record timings
```

Baselines live under `eval_results/<suite>/<config>.json`.

## Tests

```
uv run pytest
```

## Lint & format

```
uv run ruff check gene/
uv run ruff format gene/
```
