# Task 74: First-class MCP support and env-var interpolation in YAML

GitHub Issue: https://github.com/MadaraUchiha-314/sherma/issues/74

## Problem

Two related gaps surfaced while comparing sherma's `agent.yaml` to Claude
Managed Agents' `agent.yaml` (see issue #74):

1. **No first-class MCP support in YAML.** Sherma already depends on
   `mcp` and `langchain-mcp-adapters`, but a declarative agent cannot
   point at an MCP server and pull its tools into the tool registry
   without writing Python glue code. Claude's manifest exposes
   `mcp_servers:` and a `mcp_toolset` binding directly in YAML.

2. **No environment-variable interpolation in YAML.** Production
   manifests inevitably reference URLs, tokens, and per-environment
   values that should not be checked in. Claude's manifest supports
   `${VAR}`-style substitution (e.g., `url: "${FACTSET_MCP_URL}"`).
   Sherma users currently have to template the YAML themselves.

## Goal

Land both improvements in a single PR so users can write MCP-backed
agents with environment-driven configuration in pure YAML.

### Feature 1 — `mcp_servers:` in YAML

Add a top-level `mcp_servers:` section. Each entry declares an MCP
server (HTTP/SSE/stdio); on config load sherma connects, lists tools,
and registers each tool in the tool registry. The tools become usable
from `call_llm` nodes via the existing tool-binding mechanisms
(`tools:` list, `use_tools_from_registry: true`).

### Feature 2 — `${VAR}` interpolation in YAML

Substitute `${VAR}` and `${VAR:-default}` in any string value before
the YAML is parsed into Pydantic models. Missing required variables
raise `DeclarativeConfigError` with a list of unresolved names.

## Chat Iterations

_None yet._
