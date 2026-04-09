# Plan 66: Additional checkpointer types in YAML (redis, postgres)

## Context

Today `CheckpointerDef`
(`sherma/langgraph/declarative/schema.py:383`) only supports
`type: Literal["memory"]`, so `DeclarativeAgent.get_graph`
(`sherma/langgraph/declarative/agent.py:194-197`) can only wire up a
`MemorySaver`. Production agents need durable state persistence via
`langgraph-checkpoint-redis` (`AsyncRedisSaver`) and
`langgraph-checkpoint-postgres` (`AsyncPostgresSaver`). Both libraries
expose `from_conn_string(url)` async context managers plus direct
constructors that accept a pre-built Redis/psycopg client.

The `DeclarativeAgent.http_async_client` (`agent.py:138`) already carries
per-tenant auth as a Bearer token and is threaded through to
`populate_registries` / chat-model construction via the
`_extract_bearer_token` helper (`loader.py:188-196`). The same client
must also reach the checkpointer factory so that multi-tenant deploys
can authenticate their Redis/Postgres connections from the same auth
context, instead of hard-coding credentials into the YAML.

## Steps

1. **Extend `CheckpointerDef` in `sherma/langgraph/declarative/schema.py`**
   - Replace the single `CheckpointerDef` class with a pydantic v2
     discriminated union on `type`:
     - `MemoryCheckpointerDef` — `type: Literal["memory"] = "memory"`.
     - `RedisCheckpointerDef` — `type: Literal["redis"]`, `url: str`,
       optional `ttl_minutes: int | None = None`.
     - `PostgresCheckpointerDef` — `type: Literal["postgres"]`,
       `url: str`.
   - Expose the union as
     `CheckpointerDef = Annotated[Union[...], Field(discriminator="type")]`
     so existing references in `DeclarativeConfig.checkpointer` keep
     working.
   - Add a `model_validator` on the redis/postgres variants that rejects
     empty or obviously invalid URLs.

2. **Add `build_checkpointer` in `sherma/langgraph/declarative/loader.py`**
   - New async factory:
     ```python
     async def build_checkpointer(
         definition: CheckpointerDef | None,
         *,
         http_async_client: Any | None = None,
     ) -> tuple[BaseCheckpointSaver | None, AsyncExitStack | None]:
     ```
   - Returns `(None, None)` when the YAML has no `checkpointer:` block
     (caller falls back to `self.checkpointer`, preserving today's
     behaviour).
   - For `memory`: returns `(MemorySaver(), None)`.
   - For `redis` / `postgres`:
     - Lazy-import `AsyncRedisSaver` /
       `AsyncPostgresSaver` inside the branch. Raise a clear
       `DeclarativeConfigError` if the optional package is not
       installed, pointing at the `sherma[redis]` /
       `sherma[postgres]` extras.
     - Call `_resolve_checkpointer_url(url, http_async_client)` (new
       private helper) that reuses `_extract_bearer_token` and
       substitutes a `${SHERMA_AUTH_TOKEN}` placeholder in `url` with
       the Bearer token when present. URLs without the placeholder are
       used verbatim, so existing fully-qualified
       `redis://user:pass@host` / `postgres://...` URLs keep working.
     - Open the saver's `from_conn_string(resolved_url)` context manager
       through an `AsyncExitStack` (so the caller owns lifecycle).
     - Call `await saver.asetup()` once on construction (creates Redis
       indices / Postgres tables — idempotent).
     - Return `(saver, exit_stack)`.

3. **Wire `build_checkpointer` into `DeclarativeAgent`
   (`sherma/langgraph/declarative/agent.py`)**
   - Store an `_checkpointer_exit_stack: AsyncExitStack | None = None`
     private field on the agent.
   - In `get_graph`, replace the current step 3 block with:
     ```python
     saver, stack = await build_checkpointer(
         config.checkpointer, http_async_client=self.http_async_client
     )
     if saver is not None:
         checkpointer = saver
         self._checkpointer_exit_stack = stack
     else:
         checkpointer = self.checkpointer
     ```
   - Add `async def aclose(self) -> None` on `DeclarativeAgent` that
     closes the exit stack (if any) and resets
     `_compiled_graph` / `_checkpointer_exit_stack`. Document it as the
     cleanup entry point for redis/postgres savers.
   - Make `DeclarativeAgent` usable as an async context manager
     (`__aenter__` / `__aexit__`) that forwards to `get_graph` /
     `aclose` for ergonomic teardown.

4. **Optional dependencies in `pyproject.toml`**
   - Add two new extras under `[project.optional-dependencies]`:
     - `redis = ["langgraph-checkpoint-redis>=0.0.3"]`
     - `postgres = ["langgraph-checkpoint-postgres>=2.0.0", "psycopg[binary]>=3.2.0"]`
   - Do **not** add them to the base `dependencies` list — keep the
     default install small. The builder lazy-imports them and raises a
     clear error if missing.

5. **Unit tests** — add to `tests/langgraph/declarative/`:
   - `test_schema.py` — CheckpointerDef discriminated-union parsing:
     memory (default + explicit), redis (with/without `ttl_minutes`,
     URL validation), postgres, and schema errors for invalid `type`
     or missing `url`.
   - `test_loader.py` — `build_checkpointer` covering:
     - `None` definition returns `(None, None)`.
     - Memory returns a `MemorySaver`.
     - `${SHERMA_AUTH_TOKEN}` substitution when `http_async_client`
       carries a Bearer header, and passthrough when it does not.
     - Redis/postgres builders monkeypatched to capture the resolved
       URL and assert `asetup` is awaited exactly once.
     - `DeclarativeConfigError` raised when the optional package import
       fails (simulate via `sys.modules` injection).
   - `test_agent.py` — extend the existing
     `test_declarative_agent_yaml_checkpointer` with a redis variant
     (using the monkeypatched builder) and verify `aclose` closes the
     exit stack.

6. **Integration tests** — add `tests/integration/test_checkpointers.py`
   marked `integration`:
   - Skip when `SHERMA_TEST_REDIS_URL` / `SHERMA_TEST_POSTGRES_URL` is
     not set (mirroring the pattern in
     `tests/integration/test_weather_agent.py`).
   - Build a minimal declarative agent, invoke it twice across two
     `get_graph` calls sharing the same `thread_id`, and assert the
     second run sees persisted messages from the first.

7. **Docs** — update:
   - `docs/declarative-agents.md` — expand the "Checkpointer" section
     with YAML examples for redis and postgres, the
     `${SHERMA_AUTH_TOKEN}` substitution rule, the optional extras, and
     a note about calling `await agent.aclose()` (or using it as an
     async context manager) to release redis/postgres connections.
   - `docs/api-reference.md` — update the `CheckpointerDef` section to
     document the three variants and the new `aclose` method.
   - Mirror both changes into
     `skills/sherma/references/declarative-agents.md` and
     `skills/sherma/references/api-reference.md`.

8. **Skill SKILL.md** — update `skills/sherma/SKILL.md`:
   - YAML quick reference: show `checkpointer.type` values
     `memory | redis | postgres` with the `url` field and token
     placeholder.
   - Gotchas: note that redis/postgres require the matching `sherma`
     extra and that `aclose` must be called (or the agent used as an
     async context manager) to release connections.
   - API surface listing: add `aclose` / async context manager support
     to `DeclarativeAgent`.

9. **Run**
   `uv run ruff check .`, `uv run ruff format --check .`,
   `uv run pyright`, `uv run pytest -m "not integration"`.

10. **Commit and push** to
    `claude/feat-additional-checkpointer-types-in-yaml-redis-p-DIFoO`.

## Plan Revisions

_None yet._
