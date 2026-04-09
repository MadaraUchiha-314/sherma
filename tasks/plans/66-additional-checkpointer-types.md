# Plan 66: Additional checkpointer types in YAML (redis, postgres)

## Context

Today `CheckpointerDef`
(`sherma/langgraph/declarative/schema.py:383`) only supports
`type: Literal["memory"]`, so `DeclarativeAgent.get_graph`
(`sherma/langgraph/declarative/agent.py:194-197`) can only wire up a
`MemorySaver`. Production agents need durable state persistence via
`langgraph-checkpoint-redis` (`AsyncRedisSaver`) and
`langgraph-checkpoint-postgres` (`AsyncPostgresSaver`). Both libraries
expose `from_conn_string(url)` async context managers.

Credentials (Redis passwords, Postgres passwords) must stay out of the
checked-in YAML. The plan provides two complementary, generic
mechanisms that mirror patterns already used elsewhere in sherma:

- **Env-var interpolation** on string fields in `CheckpointerDef` —
  `${VAR}` / `${VAR:-default}` placeholders are expanded from
  `os.environ` at load time. This is a general-purpose substitution
  helper, not a checkpointer-specific hack, and can be reused by other
  declarative fields later.
- **A new `on_checkpointer_create` lifecycle hook** that mirrors
  `on_chat_model_create`
  (`sherma/langgraph/declarative/loader.py:305-329`). Hooks can rewrite
  the `kwargs` passed to the saver **or** return a ready-built
  `BaseCheckpointSaver` instance, giving user code the freedom to
  compute credentials at runtime (fetch from Vault, AWS Secrets
  Manager, per-tenant stores, etc.) without the framework prescribing
  where they come from.

The previous revision of this plan tried to smuggle a Bearer token off
`http_async_client.headers["authorization"]` and substitute it into the
URL via a `${SHERMA_AUTH_TOKEN}` placeholder. That coupled checkpointer
auth to LLM auth, invented a framework-specific convention, and
required sniffing HTTP request headers. The env-var + hook approach is
generic, explicit, and consistent with how LLM construction is already
customised.

## Steps

1. **Generic env-var substitution helper**
   - Add `sherma/langgraph/declarative/env.py` with
     `expand_env_vars(value: str) -> str` that resolves
     `${VAR}` and `${VAR:-default}` against `os.environ`. Raise
     `DeclarativeConfigError` for `${VAR}` when `VAR` is unset and no
     default is provided. Non-string inputs are returned unchanged so
     the helper is safe to call on arbitrary `Any` values.
   - Unit test in `tests/langgraph/declarative/test_env.py` covering:
     unchanged strings without placeholders, single and multiple
     substitutions, default values, missing var error, and
     non-string passthrough.

2. **Extend `CheckpointerDef` in `sherma/langgraph/declarative/schema.py`**
   - Replace the single class with a pydantic v2 discriminated union on
     `type`:
     - `MemoryCheckpointerDef` — `type: Literal["memory"] = "memory"`.
     - `RedisCheckpointerDef` — `type: Literal["redis"]`, `url: str`,
       optional `ttl_minutes: int | None = None`.
     - `PostgresCheckpointerDef` — `type: Literal["postgres"]`,
       `url: str`.
   - Expose the union as
     `CheckpointerDef = Annotated[Union[...], Field(discriminator="type")]`
     so `DeclarativeConfig.checkpointer` keeps working unchanged.
   - Add a `field_validator` on each `url` field that runs
     `expand_env_vars` during validation, so parsing the YAML is when
     the substitution happens. YAML authors can write
     `url: "redis://default:${REDIS_PASSWORD}@redis.example.com:6379"`
     and the value is already expanded by the time the builder sees it.

3. **`on_checkpointer_create` hook**
   - Add `CheckpointerCreateContext` in `sherma/hooks/types.py`:
     ```python
     @dataclass
     class CheckpointerCreateContext:
         definition: CheckpointerDef | None
         checkpointer: Any | None = None  # BaseCheckpointSaver | Callable[[], ...] | None
     ```
     (Mirrors `ChatModelCreateContext` — hooks can return a ready
     instance, a zero-arg factory, or leave `checkpointer=None` to fall
     back to the default builder.)
   - Add `ON_CHECKPOINTER_CREATE = "on_checkpointer_create"` to
     `HookType`, and add the corresponding async method (returning
     `CheckpointerCreateContext | None`) to both `HookExecutor`
     (protocol) and `BaseHookExecutor` (no-op default) in
     `sherma/hooks/executor.py`.
   - Unit test in `tests/hooks/test_manager.py` (or the appropriate
     existing file) exercising the new hook point through `HookManager`.

4. **`build_checkpointer` factory in
   `sherma/langgraph/declarative/loader.py`**
   - New async function:
     ```python
     async def build_checkpointer(
         definition: CheckpointerDef | None,
         *,
         hook_manager: HookManager | None = None,
     ) -> tuple[BaseCheckpointSaver | None, AsyncExitStack | None]:
     ```
   - When `definition` is `None`, return `(None, None)` so the caller
     falls back to `self.checkpointer` (today's behaviour).
   - For `memory`: returns `(MemorySaver(), None)`.
   - For `redis` / `postgres`:
     - Lazy-import `AsyncRedisSaver` /
       `AsyncPostgresSaver` inside the branch. Raise
       `DeclarativeConfigError` with an install hint pointing at the
       `sherma[redis]` / `sherma[postgres]` extras if the import fails.
     - Run the new `on_checkpointer_create` hook first (if a hook
       manager is provided). If a hook returns a checkpointer instance
       or factory, use it directly (wrap factories via
       `AsyncExitStack` only if they return a context manager) and
       **skip** the default `from_conn_string` path.
     - Otherwise, open
       `AsyncRedisSaver.from_conn_string(definition.url)` /
       `AsyncPostgresSaver.from_conn_string(definition.url)` through an
       `AsyncExitStack` and call `await saver.asetup()` once.
     - Return `(saver, exit_stack)`.

5. **Wire `build_checkpointer` into `DeclarativeAgent`
   (`sherma/langgraph/declarative/agent.py`)**
   - Add a private `_checkpointer_exit_stack: AsyncExitStack | None = None`
     field on `DeclarativeAgent`.
   - Replace the current step 3 block (lines 194-197) with:
     ```python
     saver, stack = await build_checkpointer(
         config.checkpointer, hook_manager=self.hook_manager
     )
     if saver is not None:
         checkpointer = saver
         self._checkpointer_exit_stack = stack
     else:
         checkpointer = self.checkpointer
     ```
   - Add `async def aclose(self) -> None` that closes the exit stack
     (if any) and resets `_compiled_graph` /
     `_checkpointer_exit_stack`. Document it as the cleanup entry
     point for redis/postgres-backed agents.
   - Implement `__aenter__` / `__aexit__` on `DeclarativeAgent` that
     forward to `get_graph` / `aclose` for ergonomic teardown.
   - **No** `http_async_client` is passed to `build_checkpointer`. The
     builder has no dependency on HTTP auth.

6. **Optional dependencies in `pyproject.toml`**
   - Add two new extras under `[project.optional-dependencies]`:
     - `redis = ["langgraph-checkpoint-redis>=0.0.3"]`
     - `postgres = ["langgraph-checkpoint-postgres>=2.0.0", "psycopg[binary]>=3.2.0"]`
   - Do **not** add them to the base `dependencies` list — keep the
     default install small. The builder lazy-imports them.

7. **Unit tests** — add / extend in `tests/langgraph/declarative/`:
   - `test_schema.py` — CheckpointerDef discriminated-union parsing:
     memory (default + explicit), redis (with/without `ttl_minutes`),
     postgres, invalid `type`, missing `url`, and `${VAR}` expansion at
     parse time (including the missing-var error path via
     `monkeypatch.delenv`).
   - `test_loader.py` — `build_checkpointer` coverage:
     - `None` definition returns `(None, None)`.
     - Memory returns a `MemorySaver`.
     - Redis/postgres builders monkeypatched to capture the URL and
       assert `asetup` is awaited exactly once.
     - `DeclarativeConfigError` raised when the optional package import
       fails (simulate via `sys.modules` injection).
     - `on_checkpointer_create` hook returning an instance short-circuits
       the `from_conn_string` path.
     - `on_checkpointer_create` hook leaving `checkpointer=None` falls
       through to the default path.
   - `test_agent.py` — extend `test_declarative_agent_yaml_checkpointer`
     with a redis variant using the monkeypatched builder and verify
     `aclose` closes the exit stack; add a test exercising
     `DeclarativeAgent` as an async context manager.

8. **Integration tests** — new `tests/integration/test_checkpointers.py`
   marked `integration`:
   - Skip when `SHERMA_TEST_REDIS_URL` / `SHERMA_TEST_POSTGRES_URL` is
     not set, mirroring the pattern in
     `tests/integration/test_weather_agent.py`.
   - Build a minimal declarative agent, invoke it twice across two
     `get_graph` calls sharing the same `thread_id`, and assert the
     second run sees persisted messages from the first.

9. **Docs** — update:
   - `docs/declarative-agents.md` — expand the "Checkpointer" section
     with YAML examples for redis and postgres, document the
     `${VAR}` / `${VAR:-default}` env-var substitution rule (with a
     "never hard-code credentials" note), the new
     `on_checkpointer_create` hook (with a small example that fetches
     a password from Vault), the `sherma[redis]` / `sherma[postgres]`
     extras, and the `aclose` / async-context-manager teardown
     pattern.
   - `docs/hooks.md` (or the equivalent hooks reference) — add
     `on_checkpointer_create` to the hook catalogue alongside
     `on_chat_model_create`.
   - `docs/api-reference.md` — update the `CheckpointerDef` section to
     document the three variants, the env-var interpolation, and the
     new `aclose` method on `DeclarativeAgent`.
   - Mirror all of the above into `skills/sherma/references/`.

10. **Skill SKILL.md** — update `skills/sherma/SKILL.md`:
    - YAML quick reference: show `checkpointer.type` values
      `memory | redis | postgres` with `url` and the `${VAR}` env-var
      placeholder syntax.
    - Gotchas: redis/postgres require the matching `sherma` extra;
      `aclose` (or async context manager) must be called to release
      connections; `${VAR}` is resolved at load time, not at graph
      invoke time.
    - API surface listing: add `aclose` / async context manager support
      to `DeclarativeAgent` and add `on_checkpointer_create` to the
      hooks list.

11. **Run**
    `uv run ruff check .`, `uv run ruff format --check .`,
    `uv run pyright`, `uv run pytest -m "not integration"`.

12. **Commit and push** to
    `claude/feat-additional-checkpointer-types-in-yaml-redis-p-DIFoO`.

## Plan Revisions

### Revision 1 — drop the Bearer-token hack

**Dropped.** The previous revision threaded
`DeclarativeAgent.http_async_client` into `build_checkpointer` and
substituted a `${SHERMA_AUTH_TOKEN}` placeholder in connection URLs
with the token pulled off `http_async_client.headers["authorization"]`
(via `_extract_bearer_token`). This was flagged as:

- framework-specific (invents a `SHERMA_AUTH_TOKEN` convention);
- a hack (sniffing HTTP request headers to recover credentials); and
- a needless coupling between LLM auth and checkpointer auth.

**Replaced with two orthogonal, generic mechanisms:**

1. **Env-var interpolation** (`${VAR}` / `${VAR:-default}`) on string
   fields in `CheckpointerDef`, implemented via a reusable
   `expand_env_vars` helper (new `env.py` module). This is a
   general-purpose substitution that can later be applied to other
   declarative fields with zero churn.
2. **`on_checkpointer_create` lifecycle hook** (mirroring
   `on_chat_model_create`). Hooks can compute credentials however they
   like (Vault, AWS Secrets Manager, per-tenant stores, pulling from a
   request context) and either mutate the `definition` or return a
   pre-built `BaseCheckpointSaver`.

`http_async_client` is no longer passed to `build_checkpointer`; its
purpose is LLM provider auth and it stays scoped to that.
