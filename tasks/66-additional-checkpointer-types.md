# Task 66: Additional checkpointer types in YAML (redis, postgres)

Extend the declarative YAML config so `CheckpointerDef` supports
`redis` (via `langgraph-checkpoint-redis`'s `AsyncRedisSaver`) and
`postgres` (via `langgraph-checkpoint-postgres`'s `AsyncPostgresSaver`)
in addition to the current memory-only option.

Credentials must never live in checked-in YAML. Two complementary,
generic mechanisms are provided for injecting them:

1. **`${VAR}` / `${VAR:-default}` environment-variable interpolation**
   on string fields of `CheckpointerDef`, implemented via a new reusable
   `expand_env_vars` helper.
2. **A new `on_checkpointer_create` lifecycle hook** that mirrors
   `on_chat_model_create`. Hooks can rewrite the parsed definition or
   return a pre-built `BaseCheckpointSaver` instance to integrate with
   Vault / AWS Secrets Manager / per-tenant stores without the framework
   prescribing where credentials come from.

Redis and Postgres savers open async connection pools, so
`DeclarativeAgent` is now an async context manager and exposes
`aclose()` to release them cleanly.

The optional `sherma[redis]` and `sherma[postgres]` extras lazy-install
the underlying libraries; the builder imports them on demand and
raises `DeclarativeConfigError` with an install hint if the extra is
missing.

GitHub issue: MadaraUchiha-314/sherma#66

## Chat Iterations

### Iteration 1: Initial plan with Bearer-token hack — dropped

The first plan draft threaded `DeclarativeAgent.http_async_client` into
`build_checkpointer` and substituted a `${SHERMA_AUTH_TOKEN}` placeholder
in connection URLs with a token pulled off
`http_async_client.headers["authorization"]`. User feedback:

> SHERMA_AUTH_TOKEN seems like a custom solution and reading from
> authorization header seems like a hack. build a generic way to read
> from env variables or substitue these values programmatically at
> runtime through hooks.

The plan was revised to use generic `${VAR}` / `${VAR:-default}`
env-var interpolation plus the new `on_checkpointer_create` hook.
`http_async_client` is no longer passed to the checkpointer builder —
its purpose is LLM provider auth and it stays scoped to that.
