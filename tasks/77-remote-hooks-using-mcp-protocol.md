# Task 77: Remote Hooks Using MCP Protocol

**GitHub Issue:** https://github.com/MadaraUchiha-314/sherma/issues/77

## Description

Design and implement MCP-based remote hooks for sherma. Sherma already supports remote hooks via JSON-RPC 2.0 over HTTP. This task adds MCP as an additional remote hook transport, allowing hook servers to be implemented as MCP servers with one tool per hook type.

## Requirements

1. Add `MCPHookExecutor` that connects to an MCP server and invokes hook tools
2. Add `MCPHookServer` SDK helper to wrap `HookHandler` as an MCP server
3. Support stdio and SSE/HTTP transports via YAML configuration
4. Discover available hooks via MCP `list_tools` (only call hooks the server exposes)
5. Reuse existing serialization layer
6. Keep JSON-RPC remote hooks — MCP is an additional option, not a replacement
7. Update docs and skill references

## Chat Iterations

### 2026-05-28: Answers to open questions

Decisions made on issue #77:
1. Use `hooks/` prefix for MCP tool names (e.g. `hooks/before_llm_call`).
2. Hooks are always request/response. No MCP notifications.
3. No auto-detection of tools — tools must use the `hooks/` prefix to be considered hook tools.

The implementation must include unit tests, integration tests, and a working example.

### 2026-05-28: MCP tool name separator

MCP tool names cannot contain `/` (the spec allows only `[A-Za-z0-9_.-]+`). The `hooks/` prefix from the decision above became `hooks.` (dot separator) to stay MCP-compliant while preserving the same namespacing intent.

## Implementation Summary

- `sherma/hooks/mcp_executor.py` — `MCPHookExecutor` + `MCPHookTransport` (client side; stdio / streamable-http / sse).
- `sherma/hooks/mcp_server.py` — `MCPHookServer` wrapping a `HookHandler` as a `FastMCP` server, exposing only overridden hooks as `hooks.<name>` tools.
- `sherma/langgraph/declarative/schema.py` — `MCPHookDef`; `HookDef` now accepts `import_path` | `url` | `mcp`.
- `sherma/langgraph/declarative/loader.py` — `populate_hooks` is now async and connects MCP hook servers, returning the executors for cleanup.
- `sherma/langgraph/declarative/agent.py` — connects MCP hooks during `get_graph`, adds `aclose()` for teardown.
- Example: `examples/mcp_hook_server/stdio_server.py`.
- Tests: `tests/hooks/test_mcp_executor.py`, `tests/hooks/test_mcp_server.py`, `tests/integration/test_mcp_hook_server.py`, plus schema/loader test updates.
- Docs + skill: `docs/hooks.md`, `docs/api-reference.md`, `docs/declarative-agents.md` (mirrored to `skills/sherma/references/`), and `skills/sherma/SKILL.md`.
