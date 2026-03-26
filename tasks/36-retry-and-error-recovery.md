# Task 36: Retry and Error Recovery Process

**GitHub Issue:** https://github.com/MadaraUchiha-314/sherma/issues/36

## Description

Add declarative retry and error recovery support to sherma's YAML agent framework. Currently, there is no built-in way to specify error handling, recovery, or retries for nodes — particularly important for LLM calls and IO operations.

## Requirements

1. Declarative `on_error` attribute on nodes for error handling, retries, and recovery
2. Retry policies with configurable max attempts, backoff strategies, and delays
3. Fallback routing to error-handler nodes when retries are exhausted
4. **Critical:** Must not catch LangGraph's `GraphBubbleUp` / `GraphInterrupt` exceptions — these control interrupt flow and must propagate untouched
5. Error context available in state for downstream nodes (via CEL)
6. Integration with existing `on_node_error` hook system

## Chat Iterations

_(none yet)_
