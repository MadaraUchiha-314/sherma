# Task 64: Expose Registries in NodeExecuteContext

**GitHub Issue:** https://github.com/MadaraUchiha-314/sherma/issues/64

## Description

`NodeExecuteContext` — the hook context that fires for `custom` nodes — currently exposes only `node_context`, `node_name`, `state`, and `result`. Custom-node hook authors therefore cannot reach chat models, tools, prompts, skills, or sub-agents at execution time without smuggling them in via closures captured at agent-initialization time.

Expose the `RegistryBundle` on `NodeExecuteContext` (and, for consistency, on `NodeContext` so it flows to all node builders) so that `node_execute` hooks can use the full registry surface — invoking LLMs for token counting / summarization, resolving tools dynamically, rendering prompts, or loading skills — while keeping custom nodes self-contained and declarative.

## Requirements

1. Add `registries: RegistryBundle | None` field to `NodeExecuteContext`.
2. Add `registries: RegistryBundle | None` field to `NodeContext` so the bundle is available to every node builder (populated by the agent at graph-construction time).
3. `build_custom_node` populates `NodeExecuteContext.registries` from `NodeContext.registries` before firing the `node_execute` hook.
4. `RegistryBundle` is marked non-serializable in `sherma/hooks/serialization.py` so `RemoteHookExecutor` strips it on the wire (it contains live chat-model instances).
5. Docs (`docs/hooks.md`, `docs/declarative-agents.md`) and skill references (`skills/sherma/references/`) are updated together with a short example showing a custom-node hook that invokes a chat model via `ctx.registries`.
6. Tests cover: registries flow through to the hook, default `None` when not supplied, a hook that uses `llm_registry`/`chat_models` to run a chat model, and that `registries` is stripped by the JSON-RPC serializer.

## Chat Iterations

_(none yet)_
