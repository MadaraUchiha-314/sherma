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

(None yet)
