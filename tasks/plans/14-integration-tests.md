# Plan: Integration Tests for sherma

## Context

The sherma project has 7 working examples that demonstrate its core capabilities (programmatic agents, declarative agents, hooks, multi-agent orchestration, skills, A2A server, remote hooks). These examples are currently only runnable manually with real API keys. The goal is to port them into automated integration tests with mocked LLM calls, so the full agent lifecycle is continuously verified in CI without requiring external services.

## Mocking Strategy

**LLM mocking**: Build a `FakeChatModel` extending `BaseChatModel` that returns scripted `AIMessage` responses in sequence. Inject via `agent._registries = RegistryBundle(chat_models={"llm-id": fake_model})` (pattern from existing test at `tests/langgraph/declarative/test_agent.py:144`). For multi-agent tests where sub-agents are created internally, `monkeypatch` the `_construct_chat_model` function in `sherma/langgraph/declarative/loader.py`.

**Tool mocking**: Create a fake `get_weather` tool in `tests/integration/fake_tools.py` that returns deterministic data without HTTP calls. Reference via `import_path` in test YAML.

## File Plan

### New Files

| File | Purpose |
|------|---------|
| `tests/integration/fake_tools.py` | Fake `get_weather` tool (no HTTP) |
| `tests/integration/conftest.py` | `FakeChatModel`, helper fixtures |
| `tests/integration/test_weather_agent.py` | Programmatic `LangGraphAgent` with ReAct graph |
| `tests/integration/test_declarative_weather_agent.py` | Declarative agent + interrupt/resume flow |
| `tests/integration/test_declarative_hooks_agent.py` | Hook executors fire correctly, prompt guardrail modifies prompt |
| `tests/integration/test_multi_agent.py` | Supervisor delegates to sub-agent via `use_sub_agents_as_tools` |
| `tests/integration/test_declarative_skill_agent.py` | Skill discovery, loading, and execution loop |
| `tests/integration/test_remote_hook_server.py` | JSON-RPC hook server via `httpx.ASGITransport` |
| `tests/integration/test_a2a_server.py` | `ShermaAgentExecutor` processes A2A messages |

## Implementation Order

1. `fake_tools.py` (dependency for all tests)
2. `conftest.py` (FakeChatModel + helpers)
3. `test_weather_agent.py` (simplest, validates FakeChatModel works)
4. `test_declarative_weather_agent.py` (validates declarative + interrupt)
5. `test_declarative_hooks_agent.py` (hooks verification)
6. `test_remote_hook_server.py` (isolated, no agent needed)
7. `test_a2a_server.py` (executor test)
8. `test_multi_agent.py` (complex, depends on monkeypatch pattern)
9. `test_declarative_skill_agent.py` (most complex response scripting)

## Verification

```bash
uv run pytest tests/integration/ -m integration -v
uv run pytest -m "not integration"
uv run ruff check .
uv run pyright
```
