Introduction
----------------

This project is a minimal from-scratch agentic-AI harness built directly on
the Anthropic SDK, driven by a stateless `Turn` value object and exercised
against a genealogy Q&A use case. The goal is to discover a small set of
flexible core patterns that generalise to other agentic problems, not to
build a universal framework.

The specific use case — programmatic queries over GEDCOM family-tree data —
is loosely coupled to the core. The `gene.agent` package is domain-agnostic;
`gene.genealogy` uses it as a library.

The deliberate choice to avoid LangChain, LangGraph, and similar frameworks
comes down to control. Those tools coax you into a particular pattern, do
a lot of work invisibly, and — since agentic AI is still very early — are
likely to be superseded. Building the harness by hand keeps the surface
area small enough to understand, easy to instrument, and cheap to change
when the right patterns come into focus.

The application code — the loop, the tool dispatch, the guardrails — is
usually called a *harness*. The LLM outputs tokens; the harness decides
what those tokens mean. Nothing else has authority to run a tool, mutate
state, or decide when the loop terminates. Being able to say exactly what
happens on each turn is the point.

Design principles
------------------

High-level goals:

* **Simplicity** — agentic AI is complex enough already; don't add more.
* **Understandability** — the code should always tell you what it's doing.
* **Flexibility** — reuse a small set of pieces in different ways as harder
  problems come into scope.

These drive lower-level needs:

* **Observability** — must fall out of the design, not be bolted on later.
* **Testability** — unit tests, integration tests, and eval suites that
  detect any drift in behaviour from a code or config change.
* **Rein in statefulness** — most of the code stays functional
  (referentially transparent). The few genuinely stateful pieces are
  isolated and obvious.

Concrete choices that follow:

* Cache every LLM call by request payload; a re-run of an eval is
  deterministic and near-free.
* Make `Turn` a stateless value object holding the ordered `Step` list,
  final message, tokens, and terminal reason.
* Push conversation history, memory, and other stateful concerns to
  higher layers that compose stateless Turns.
* Anthropic-only for now. LiteLLM-style adapters are half a day's work
  and add complexity that doesn't buy anything at this stage. Anthropic's
  three tiers already give the accuracy / speed / cost trade-off the
  eval matrix needs.
* CLI-only. Visualisation or a UI would freeze the design too early.

Architecture
--------------

The fundamental unit is a `Turn`: one user query in, one final answer
out, with any number of intermediate `Step`s in between. Each Step is
one round-trip to the LLM plus the tool calls it triggered, with token
counts, timings, request/response payloads, and a cache-hit flag.

A Turn is a plain value object. Given the same input and a warm LLM
cache, it's referentially transparent — same input, same Turn. This one
choice is what makes everything else cheap:

* **Caching** — `CachedAnthropic` keys every request by its full JSON
  payload; a repeat call returns the exact stored `Message`. Eval re-runs
  hit disk, not the API.
* **Logging** — a chat session log is just a JSONL file with one Turn per
  line. No bespoke tracing subsystem.
* **Evals** — a `TurnCase` can check not only the final text but the
  *shape* of the work: which tools ran, whether SQL had a `WHERE` clause,
  how many steps it took to converge.
* **Testing** — most of the codebase can be exercised with fabricated
  Turn/Step/ToolCall values, no I/O.

`Conversation` wraps Turn with the small amount of state a chat REPL
needs: message history to hand back to the model on the next call. This
is deliberately the *only* stateful layer above Turn, and it is the
correct seam for introducing a smarter Memory later on.

The package split follows the same principle. `gene.agent` is the
library — Turn, Conversation, tool dispatch, evals runner, log viewer.
`gene.genealogy` is a consumer that builds a genealogy-specific
`Conversation` and `run_query` tool on top. The evals runner is
domain-agnostic; it discovers suite modules by directory and duck-typed
convention.

First use case: Genealogy
---------------------------

Family genealogy is a personal hobby and the driving use case. Sites
like geni.com and ancestry.com are fine for browsing, but the raw data
is where the interesting questions live — the kind rules-based code
handles clumsily and an LLM handles well.

GEDCOM is the standard export format: a hierarchical text file of facts
and relations that looks like this:

    0 @I75@ INDI
    1 NAME Waldemar  //
    1 SEX M
    1 BIRT
    2 DATE        1868
    1 DEAT
    2 DATE        1879
    1 FAMC @F3@
    0 @I76@ INDI
    1 NAME Sophie of_Prussia //
    1 TITL Queen of Greece
    1 SEX F

The corpus includes two of my own family trees (all blood relatives
capped by Geni; direct ancestors only in the second) plus a handful of
public trees — English royals, Shakespeare, the Brontës, US Presidents.

Storage: `ged4py` parses the file, Pydantic models normalise it into 3NF,
and the result loads into SQLite. SQLite fits the shape of the problem —
small, write-once / read-many, preinstalled everywhere, no need for a
cloud DB. SQL is also excellent LLM tool bait: models write good SQL,
especially against simple schemas.

Five tables cover the current model:

* individuals
* families
* family_children
* individual_events
* family_events

The whole schema with indices is 45 lines of SQL — it may grow as more
GEDCOM tags come into scope.

Given the schema and a well-written system prompt, the genealogy agent
answers questions like:

> "How many uncles does David Johnston have?"

or, against the royal dataset:

> "Has there ever been a King of England born in Denmark? If so, did
> they have any children who became King or Queen?"

Some questions are better served by tools that walk a tree graph rather
than SQL. Adding those is a matter of registering another tool — the
harness doesn't care.

How it works
--------------

Test running:

Tests live in `tests/` and run with `uv run pytest`. Coverage spans the
parts that don't hit the network: the Turn/Step model, the Conversation
wrapper, the eval-case predicates, the eval-suite loader, the GEDCOM
parser, the SQLite store, and the guarded SQL tool. Paths that would
hit Anthropic or a real DB are deliberately excluded — the eval runner
covers those with real LLM calls, cached where possible so re-runs are
cheap.

The pattern is functional-core / imperative-shell: because Turn is a
plain value object and `CachedAnthropic` can be swapped for a fake, most
of the codebase can be exercised with fabricated messages and no I/O.
Predicate tests, for instance, build fake Turn/Step/ToolCall values
in-memory rather than driving the real agent.

Loading genealogy data:

Drop a GEDCOM file into `genealogy_data/<tag>.ged` (gitignored). Run:

    uv run python -m gene.genealogy.load <tag>

The loader parses via `ged4py`, builds Pydantic models per row, and
writes `gene/genealogy/db/<tag>.sqlite`. The DB path and data path are
both resolved through `gene.genealogy.config.get_db_path(tag)` — a
single seam that eval prechecks, the chat CLI, and future tooling all
share, so "is this dataset installed?" is answered the same way
everywhere. An unknown tag exits with a list of what's actually
available.

Running evals:

Two runners share one contract. `gene.agent.evals` runs a directory of
suites against a single model — quick feedback while iterating on a
case or a prompt. `gene.agent.run_evals` runs every suite × every named
config, diffs the result against a saved baseline, and (with `--save`)
overwrites the baseline. That's what runs before merging a change: does
the pass/fail matrix move? Do the token counts drift? Did any case that
used to take 3 steps now take 5?

A suite is any `.py` file in the given directory that exports `CASES`.
For TurnCase suites it also exports `build_conversation(llm)` (a
factory returning a fully-wired Conversation) and, optionally,
`precheck()` returning a skip reason when data isn't available. The
core runner imports nothing from any use case — it discovers all this
by duck-typed convention. Adding a new domain means adding a new
directory of suite modules, not touching the runner.

The distinction between one-shot `Case` (just a prompt, check the text)
and `TurnCase` (full agent loop, check the *shape* of the work) is
deliberate. "Got the right answer" is often not enough — did the model
filter in SQL or fetch everything and post-filter? Did it burn eight
tool-call rounds when two would do? Those checks are only possible
because Turn is a first-class value with step-level detail.

Observability:

Every LLM round-trip inside a Turn is captured as a Step, and the Turn
itself holds the ordered step list, the terminal reason, the total
tokens, and the final message. Logging a chat session is a matter of
appending each Turn to a JSONL file — nothing bespoke. The
`gene.agent.log_view` CLI reads those logs in either compact mode (one
line per turn with user input and final answer) or `--trace` mode
(every step with tool inputs, outputs, and timings), with ANSI colour
when writing to a TTY.

Observability isn't a separate subsystem — it falls out of the fact
that the fundamental unit is a value object. Anything that can call the
agent can also inspect exactly what happened.

Model config:

`gene.agent.config.get_model_name(tag)` maps short tags — `haiku`,
`sonnet`, `opus` — to concrete Anthropic model names.
`get_llm_config` returns the dict the LLM client wants: model,
max_tokens, key dir, cache dir. This is the one place model names live,
so bumping a model is a one-line change and nothing downstream has to
know it happened.

`get_eval_configs()` in `gene.agent.eval_configs` returns the named
sweep — currently just the three model tags, but the shape supports
tuned variants (e.g. `sonnet-cold` with a different max_tokens or
thinking budget) without touching any runner code.

API key config:

The Anthropic API key sits at `~/.config/gene/keys/anthropic` —
XDG-friendly, no dotenv, no environment variables shipped in the repo.
The keys directory resolves through `XDG_CONFIG_HOME` if set, otherwise
`~/.config`. The layout leaves room for additional providers later.

Security / safety:

The single tool exposed to the genealogy agent is `run_query`, a
guarded wrapper around SQLite. Three defences stack:

1. **Statement whitelist** — the first non-comment token must be
   `SELECT` or `WITH`. Everything else is refused.
2. **Keyword blacklist** — `ATTACH`, `PRAGMA`, and `LOAD_EXTENSION` are
   refused even inside an otherwise-valid SELECT, since those are the
   paths a model-authored query could use to escape the DB.
3. **Wall-clock timeout** — a background thread calls
   `conn.interrupt()` after five seconds, so a pathological Cartesian
   product can't stall the loop.

Results are capped at 100 rows to keep the LLM's context bounded on
mistakes. These aren't a substitute for treating the DB as untrusted,
but they raise the cost of a bad query from "corrupts state" to
"returns an error message".

The disk cache is content-addressed by the exact request payload, so
cache poisoning would require write access to the local cache dir —
not a threat the code needs to defend against.

Caching and timings:

The disk cache (via the `diskcache` package) is what makes eval runs
cheap. Every LLM request is keyed by its full JSON payload; a repeat
call returns the exact stored `Message` with `cache_hit=True` in the
metadata. `run_evals --no-cache` bypasses the cache and records a
timing sample per case per config into `<config>.timings.jsonl`, then
rebuilds `timings_summary.json` files from the accumulated history — so
latency numbers reflect real API round-trips, aggregated over many runs
to smooth out variance.

Log files are gitignored; eval baselines are committed except for
`eval_results/gene/genealogy/`, since even case names in that tree leak
family info.

Dependencies
--------------

Kept intentionally small:

* **anthropic** — official SDK for Claude models. Handles HTTP, message
  formatting, and the tool-use protocol (model emits tool-call blocks;
  the harness runs them and hands results back).
* **pydantic** — used for GEDCOM row models and eval-case value objects.
  Already a transitive dependency of `anthropic`.
* **diskcache** — a SQLite-backed key-value store used to cache LLM
  responses locally.
* **ged4py** — reads GEDCOM files.
* **sqlite3** (stdlib) — storage for both the family data and the cache.

Future additions
------------------

The design leaves room for two extensions that will likely come once a
use case demands them:

* **Smarter memory.** Right now `Conversation` just accumulates raw
  messages and hands them back. Long conversations will eventually
  pressure the context window. The Conversation seam is the right place
  to insert compaction — summaries of older turns, a table-of-contents
  index, or an explicit memory-lookup tool that lets the model retrieve
  older detail on demand. The Turn stays stateless throughout.
* **Layered thought / multi-agent.** Turns compose. A supervisor Turn
  can call sub-agent Turns with different system prompts; a state
  machine can model modes (plan / execute / verify / repair) as
  transitions between Turn configurations. The stateless-Turn contract
  is what keeps this from turning into a graph of tangled state.

Neither ships until a real problem justifies it.

TODO
------

* Host public GEDCOMs somewhere fetchable so the genealogy evals can
  run without a manual data install.
* No Docker or container packaging yet; local `uv sync` is the current
  build story. Revisit if the project moves to shared infrastructure.
