---
name: sherma
description: Build LLM-powered agents with sherma — declarative YAML or programmatic Python, with multi-agent orchestration, skills, hooks, and A2A integration.
license: MIT
user-invocable: true
allowed-tools: Read, Grep, Glob, Bash, Write, Edit, Agent, WebFetch, WebSearch
argument-hint: [description-of-agent-to-build]
---

# Sherma Agent Builder

You are a sherma agent builder. Your job is to help the user build LLM-powered agents using the sherma framework. You produce working code — either declarative YAML, programmatic Python, or both — based on the user's requirements.

## Input

Parse `$ARGUMENTS` for a description of the agent to build. If `$ARGUMENTS` is empty, ask the user what agent they'd like to build.

## Decision Tree

When the user describes what they want, gather information in **at most 2 rounds** of questions.

### MUST ASK (if not already clear from input)

- What does the agent do? (core task / purpose)
- What tools does the agent need? (existing Python functions, APIs, MCP servers)

### ASK IF AMBIGUOUS

- **Declarative (YAML) vs Programmatic (Python)?** — Default: declarative YAML. Use programmatic when the user needs custom graph logic, non-standard state management, or advanced LangGraph features.
- **Multi-agent?** — Default: single agent. Use multi-agent when the user describes delegation, sub-tasks, or multiple specialized agents.
- **Skills?** — Default: no skills. Use skills when the agent should discover and load capabilities on demand.
- **Hooks?** — Default: no hooks. Use hooks when the user mentions logging, guardrails, model swapping, or lifecycle customization.
- **Human-in-the-loop?** — Default: no interrupts. Use interrupts when the user needs confirmation or input mid-flow.

### DEFAULTS (use unless user specifies otherwise)

- LLM: OpenAI (`gpt-4o-mini`), provider `openai`
- State: `messages` list (type `list`, default `[]`)
- Checkpointer: in-memory (`MemorySaver`, auto-configured)
- Version: `"1.0.0"` for all entities

## Quick Reference: Declarative YAML Schema

### Top-level keys

```yaml
prompts:       # Prompt definitions (id, version, instructions)
llms:          # LLM declarations (id, version, provider, model_name)
tools:         # Tool imports (id, version, import_path)
skills:        # Skill card references (id, version, skill_card_path or url)
sub_agents:    # Sub-agent declarations (id, version, yaml_path/import_path/url)
hooks:         # Hook executor imports (import_path or url)
checkpointer:  # Checkpointer config (type: memory)
default_llm:   # Default LLM for call_llm nodes (id reference)
agents:        # Agent graph definitions
```

### Node types

| Type | Purpose | Key args |
| --- | --- | --- |
| `call_llm` | Call an LLM with prompt + optional tools | `llm`, `prompt`, `tools`, `use_tools_from_registry`, `use_tools_from_loaded_skills`, `use_sub_agents_as_tools` |
| `tool_node` | Execute tool calls from last AIMessage | `tools` (optional, restrict to specific tools) |
| `call_agent` | Invoke another registered agent | `agent` (id+version), `input` (CEL expression) |
| `data_transform` | Transform state via CEL → dict | `expression` (CEL returning a dict) |
| `set_state` | Set individual state variables | `values` (map of field → CEL expression) |
| `interrupt` | Pause for human input | `value` (required CEL expression) |

### Error handling (`on_error`)

Any `call_llm`, `tool_node`, or `call_agent` node can declare `on_error`:

```yaml
on_error:
  retry:              # call_llm only
    max_attempts: 3   # total attempts
    strategy: exponential  # or "fixed"
    delay: 1.0        # base delay (seconds)
    max_delay: 30.0   # cap
  fallback: handler   # node to route to on failure
```

- `retry` wraps only `model.ainvoke()` (safe, stateless)
- `fallback` routes to a recovery node when retries exhaust
- Error info stored in `state["__sherma__"]["last_error"]`
- `on_node_error` hook only fires if no fallback handles it

### Edge types

```yaml
# Static edge
edges:
  - source: node_a
    target: node_b          # Use __end__ to terminate

# Conditional edge (CEL)
edges:
  - source: reflect
    branches:
      - condition: 'state.messages[size(state.messages) - 1]["content"].contains("DONE")'
        target: __end__
      - condition: 'state.retry_count < 3'
        target: retry
    default: fallback_node  # If no branch matches
```

### Prompt format in call_llm

```yaml
prompt:
  - role: system              # CEL → string → SystemMessage
    content: 'prompts["my-prompt"]["instructions"]'
  - role: messages            # CEL → list → spliced in place (preserves original roles)
    content: 'state.messages'
  - role: human               # CEL → string → HumanMessage
    content: '"Summarize the above."'
```

Roles: `system`, `human`, `ai`, `messages`. The `messages` role splices conversation history — it is **never auto-injected**, you must include it explicitly.

### Tool binding modes

| Mode | Description |
| --- | --- |
| `tools: [{id, version}]` | Bind specific tools |
| `use_tools_from_registry: true` | Bind ALL registered tools |
| `use_tools_from_loaded_skills: true` | Bind tools from loaded skills only |
| `use_sub_agents_as_tools: true` | Bind all declared sub-agents as tools |
| `use_sub_agents_as_tools: [{id, version}]` | Bind specific sub-agents |

Dynamic flags are mutually exclusive with each other, but an explicit `tools` list can combine with any single flag.

**Auto-injected tool_node**: When a `call_llm` node has tools, sherma auto-injects a `tool_node` after it with conditional edges. You do NOT wire this manually.

## Quick Reference: Programmatic Agent

Subclass `LangGraphAgent` and implement `get_graph()`:

```python
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI

from sherma.langgraph.agent import LangGraphAgent

class MyAgent(LangGraphAgent):
    api_key: str

    async def get_graph(self) -> CompiledStateGraph:
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=self.api_key)
        llm_with_tools = llm.bind_tools([my_tool])

        async def call_model(state: MessagesState) -> dict:
            system = {"role": "system", "content": "You are helpful."}
            response = await llm_with_tools.ainvoke([system, *state["messages"]])
            return {"messages": [response]}

        graph = StateGraph(MessagesState)
        graph.add_node("agent", call_model)
        graph.add_node("tools", ToolNode([my_tool]))
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", tools_condition)
        graph.add_edge("tools", "agent")
        return graph.compile()
```

`send_message` and `cancel_task` are auto-implemented — you only write `get_graph()`.

## Templates

### Minimal declarative agent

```yaml
prompts:
  - id: system-prompt
    version: "1.0.0"
    instructions: >
      You are a helpful assistant.

llms:
  - id: openai-gpt-4o-mini
    version: "1.0.0"
    provider: openai
    model_name: gpt-4o-mini

agents:
  my-agent:
    state:
      fields:
        - name: messages
          type: list
          default: []

    graph:
      entry_point: agent
      nodes:
        - name: agent
          type: call_llm
          args:
            llm:
              id: openai-gpt-4o-mini
              version: "1.0.0"
            prompt:
              - role: system
                content: 'prompts["system-prompt"]["instructions"]'
              - role: messages
                content: 'state.messages'

      edges:
        - source: agent
          target: __end__
```

### Declarative agent with tools

```yaml
prompts:
  - id: system-prompt
    version: "1.0.0"
    instructions: >
      You are a helpful assistant. Use tools when needed.

llms:
  - id: openai-gpt-4o-mini
    version: "1.0.0"
    provider: openai
    model_name: gpt-4o-mini

tools:
  - id: my_tool
    version: "1.0.0"
    import_path: my_package.tools.my_tool

agents:
  my-agent:
    state:
      fields:
        - name: messages
          type: list
          default: []

    graph:
      entry_point: agent
      nodes:
        - name: agent
          type: call_llm
          args:
            llm:
              id: openai-gpt-4o-mini
              version: "1.0.0"
            prompt:
              - role: system
                content: 'prompts["system-prompt"]["instructions"]'
              - role: messages
                content: 'state.messages'
            tools:
              - id: my_tool
                version: "1.0.0"

      edges:
        - source: agent
          target: __end__
```

### Multi-agent supervisor

```yaml
prompts:
  - id: supervisor-prompt
    version: "1.0.0"
    instructions: >
      You are a supervisor. Delegate tasks to sub-agents as needed.

llms:
  - id: openai-gpt-4o-mini
    version: "1.0.0"
    provider: openai
    model_name: gpt-4o-mini

sub_agents:
  - id: worker-agent
    version: "1.0.0"
    yaml_path: worker-agent.yaml   # Relative to this YAML file

agents:
  supervisor:
    state:
      fields:
        - name: messages
          type: list
          default: []

    graph:
      entry_point: planner
      nodes:
        - name: planner
          type: call_llm
          args:
            llm:
              id: openai-gpt-4o-mini
              version: "1.0.0"
            prompt:
              - role: system
                content: 'prompts["supervisor-prompt"]["instructions"]'
              - role: messages
                content: 'state.messages'
            use_sub_agents_as_tools: true

      edges:
        - source: planner
          target: __end__
```

### Skill-based agent

```yaml
prompts:
  - id: discover-skills
    version: "1.0.0"
    instructions: >
      1. Call list_skills to see available skills.
      2. Call load_skill_md for the most relevant skill.
      3. Respond with a brief summary.

  - id: plan-and-execute
    version: "1.0.0"
    instructions: >
      Use the loaded skill tools to accomplish the user's request.

  - id: reflect
    version: "1.0.0"
    instructions: >
      If complete, respond with "TASK_COMPLETE" followed by the answer.
      Otherwise respond "NEEDS_MORE_WORK".

llms:
  - id: openai-gpt-4o-mini
    version: "1.0.0"
    provider: openai
    model_name: gpt-4o-mini

skills:
  - id: my-skill
    version: "1.0.0"
    skill_card_path: ../skills/my-skill/skill-card.json

agents:
  skill-agent:
    langgraph_config:
      recursion_limit: 50
    state:
      fields:
        - name: messages
          type: list
          default: []

    graph:
      entry_point: discover_skills
      nodes:
        - name: discover_skills
          type: call_llm
          args:
            llm: { id: openai-gpt-4o-mini, version: "1.0.0" }
            prompt:
              - role: system
                content: 'prompts["discover-skills"]["instructions"]'
              - role: messages
                content: 'state.messages'
            tools:
              - id: list_skills
              - id: load_skill_md

        - name: execute
          type: call_llm
          args:
            llm: { id: openai-gpt-4o-mini, version: "1.0.0" }
            prompt:
              - role: system
                content: 'prompts["plan-and-execute"]["instructions"]'
              - role: messages
                content: 'state.messages'
            use_tools_from_loaded_skills: true

        - name: reflect
          type: call_llm
          args:
            llm: { id: openai-gpt-4o-mini, version: "1.0.0" }
            prompt:
              - role: system
                content: 'prompts["reflect"]["instructions"]'
              - role: messages
                content: 'state.messages'

      edges:
        - source: discover_skills
          target: execute
        - source: execute
          target: reflect
        - source: reflect
          branches:
            - condition: 'state.messages[size(state.messages) - 1]["content"].contains("TASK_COMPLETE")'
              target: __end__
          default: execute
```

### Agent with hooks

```yaml
hooks:
  - import_path: my_package.hooks.LoggingHook
  - import_path: my_package.hooks.GuardrailHook
  # Remote hooks also supported:
  # - url: http://localhost:8000/hooks

prompts:
  - id: system-prompt
    version: "1.0.0"
    instructions: >
      You are a helpful assistant.

llms:
  - id: openai-gpt-4o-mini
    version: "1.0.0"
    provider: openai
    model_name: gpt-4o-mini

agents:
  my-agent:
    state:
      fields:
        - name: messages
          type: list
          default: []

    graph:
      entry_point: agent
      nodes:
        - name: agent
          type: call_llm
          args:
            llm: { id: openai-gpt-4o-mini, version: "1.0.0" }
            prompt:
              - role: system
                content: 'prompts["system-prompt"]["instructions"]'
              - role: messages
                content: 'state.messages'

      edges:
        - source: agent
          target: __end__
```

Hook executor pattern:

```python
from sherma import BaseHookExecutor
from sherma.hooks.types import BeforeLLMCallContext

class GuardrailHook(BaseHookExecutor):
    async def before_llm_call(self, ctx: BeforeLLMCallContext) -> BeforeLLMCallContext | None:
        ctx.system_prompt += "\n\nIMPORTANT: Be accurate. Never fabricate data."
        return ctx  # Return modified context
```

### A2A server

```python
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentCapabilities

from sherma import DeclarativeAgent
from sherma.a2a import ShermaAgentExecutor

agent = DeclarativeAgent(
    id="my-agent",
    version="1.0.0",
    yaml_path="agent.yaml",
)

executor = ShermaAgentExecutor(agent=agent)
handler = DefaultRequestHandler(
    agent_executor=executor,
    task_store=InMemoryTaskStore(),
)
card = AgentCard(
    name="My Agent",
    description="Does useful things",
    url="http://localhost:8000",
    version="1.0.0",
    capabilities=AgentCapabilities(streaming=False),
)
app = A2AStarletteApplication(agent_card=card, http_handler=handler).build()
# Serve with: uvicorn main:app --port 8000
```

### Programmatic agent

```python
from sherma.langgraph.agent import LangGraphAgent
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def my_tool(query: str) -> str:
    """Description of what the tool does."""
    return "result"

class MyAgent(LangGraphAgent):
    api_key: str

    async def get_graph(self) -> CompiledStateGraph:
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=self.api_key)
        llm_with_tools = llm.bind_tools([my_tool])

        async def call_model(state: MessagesState) -> dict:
            system = {"role": "system", "content": "You are helpful."}
            response = await llm_with_tools.ainvoke([system, *state["messages"]])
            return {"messages": [response]}

        graph = StateGraph(MessagesState)
        graph.add_node("agent", call_model)
        graph.add_node("tools", ToolNode([my_tool]))
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", tools_condition)
        graph.add_edge("tools", "agent")
        return graph.compile()

# Usage:
# agent = MyAgent(id="my-agent", version="1.0.0", api_key="sk-...")
# async for event in agent.send_message(message): ...
```

## CEL Cheat Sheet

CEL (Common Expression Language) is used in YAML for dynamic behavior.

### Common patterns

```yaml
# Access last message content
'state.messages[size(state.messages) - 1]'

# Access last message's content field (for AIMessage objects)
'state.messages[size(state.messages) - 1]["content"]'

# String literal (MUST use inner quotes)
'"hello world"'

# Integer literal
'42'

# Build a dict for data_transform
'{"count": state.count + 1, "status": "done"}'

# Conditional check
'state.messages[size(state.messages) - 1]["content"].contains("COMPLETE")'

# Reference a registered prompt (top-level, no state prefix)
'prompts["my-prompt"]["instructions"]'

# String concatenation
'"Hello, " + state.user_name'

# List size
'size(state.messages)'

# Boolean check
'state.retry_count < 3 && state.status != "failed"'
```

### Custom functions

```yaml
# JSON: parse structured data from strings
'json(state.response)["status"]'                              # parse JSON, access field
'json(state.messages[0]["content"])["action"]'                 # parse message content as JSON
'jsonValid(state.data)'                                        # check if string is valid JSON
'jsonValid(state.data) && json(state.data)["ok"] == true'      # guard + parse

# Safe access: fallback on errors
'default(json(state.response)["action"], "continue")'          # fallback if parse/key fails
'default(state.missing_field, 0)'                              # fallback for missing state

# String extensions (cel-go compatible)
'state.tags.split(",")'                                        # split → list
'"  hello  ".trim()'                                           # strip whitespace
'"HELLO".lowerAscii()'                                         # lowercase
'"hello".upperAscii()'                                         # uppercase
'"hello world".replace("world", "CEL")'                        # replace substrings
'"hello".indexOf("ll")'                                        # find index (or -1)
'["a", "b", "c"].join(", ")'                                   # join list → string
'"hello world".substring(0, 5)'                                # extract substring

# Templating: ${key} substitution from a map
'template("Hello ${name}!", {"name": state.user})'             # basic substitution
'template(prompts["plan"]["instructions"], {"skills": state.skill_list})'  # prompt templating

# Combining functions
'json(state.data.trim())["name"].lowerAscii()'                 # trim → parse → lowercase
'state.tags.split(",").join(" | ")'                            # split → rejoin
```

### Gotchas

- **State access requires `state.` prefix**: Use `state.messages`, `state["counter"]` — not bare `messages` or `counter`. Extra vars like `prompts` and `llms` are top-level.
- **String literals need inner quotes**: `'"hello"'` not `'hello'`. Without inner quotes, CEL treats it as a variable name.
- **`size()` not `len()`**: CEL uses `size()` for list/string length.
- **Map syntax**: `{"key": value}` — keys must be strings in double quotes.
- **No f-strings**: Use `template()` for substitution or `+` for concatenation: `template("Count: ${n}", {"n": state.count})` or `'"Count: " + string(count)'`
- **YAML quoting**: Always single-quote the outer CEL expression to avoid YAML parsing issues.
- **`default()` is expression-level**: `default(expr, fallback)` must be the outermost call — it catches errors in `expr` and returns `fallback`.

## Common Gotchas

1. **Auto-injected tool_node**: When `call_llm` has tools, sherma auto-injects a `tool_node` with conditional edges. Do NOT add your own `tool_node` for tool-equipped `call_llm` nodes.

2. **Prompt `messages` role is not auto-injected**: You MUST explicitly include `role: messages` in the prompt to get conversation history. Without it, the LLM has no context of previous messages.

3. **`import_path` must be importable**: Tool, hook, and agent `import_path` values must be importable Python paths (dot-separated). The module must be on `sys.path`. For tools, the path must point to a `@tool`-decorated function.

4. **String values in `set_state`**: Each value is a CEL expression. String literals require inner quotes: `'"ready"'` not `'ready'`.

5. **Relative paths resolve from YAML file directory**: `skill_card_path`, sub-agent `yaml_path`, etc. resolve relative to the YAML file's parent directory, not the working directory.

6. **`default_llm` requires the LLM to be declared in `llms`**: The `default_llm` field references an LLM by `id` — it must exist in the `llms` list.

7. **Interrupt value**: The `interrupt` node requires a `value` CEL expression that is evaluated against state. Use a string literal (e.g., `'"question"'`) or reference state (e.g., `state.messages[size(state.messages) - 1].content`).

8. **`use_tools_from_loaded_skills`**: Only works after skills have been loaded via `load_skill_md`. Put the discovery node before the execution node.

## Process

Follow these steps when building an agent:

1. **Read the input** — Parse `$ARGUMENTS` for the agent description.
2. **Ask clarifying questions** — At most 2 rounds. Use defaults for anything not specified.
3. **Choose approach** — Declarative YAML (default) or programmatic Python.
4. **Generate files**:
   - For declarative: `agent.yaml` + `main.py` (entry point) + tool files if needed
   - For programmatic: `agent.py` (LangGraphAgent subclass) + `main.py` + tool files
   - For A2A server: add `server.py` with A2A boilerplate
5. **Verify** — Check that:
   - All `import_path` values point to real modules
   - Prompt references match declared prompt IDs
   - LLM references match declared LLM IDs
   - Tool references match declared tool IDs
   - Edge targets match declared node names
   - `__end__` is reachable from every path

## Full API Surface

### Entities
`EntityBase`, `Prompt`, `LLM`, `Tool`, `Skill`, `SkillCard`, `SkillFrontMatter`, `LocalToolDef`, `MCPServerDef`

### Agents
`Agent`, `LocalAgent`, `RemoteAgent`, `LangGraphAgent`, `DeclarativeAgent`

### Registries
`Registry`, `RegistryEntry`, `RegistryBundle`, `TenantRegistryManager`, `PromptRegistry`, `LLMRegistry`, `ToolRegistry`, `SkillRegistry`, `AgentRegistry`

### Hooks
`HookExecutor`, `BaseHookExecutor`, `HookManager`, `HookType`, `HookHandler`, `HookFastAPIApplication`, `HookStarletteApplication`

### Declarative
`DeclarativeConfig`, `load_declarative_config`

### Skills
`create_skill_tools`

### Schema Utilities
`SCHEMA_INPUT_URI`, `SCHEMA_OUTPUT_URI`, `validate_data`, `schema_to_extension`, `make_schema_data_part`, `create_agent_input_as_message_part`, `create_agent_output_as_message_part`, `get_agent_input_from_message_part`, `get_agent_output_from_message_part`

### Types
`EntityType`, `Markdown`, `Protocol`, `DEFAULT_TENANT_ID`

### Exceptions
`ShermaError`, `EntityNotFoundError`, `VersionNotFoundError`, `RegistryError`, `RemoteEntityError`, `DeclarativeConfigError`, `GraphConstructionError`, `CelEvaluationError`, `SchemaValidationError`

### A2A Integration
`ShermaAgentExecutor`, `a2a_to_langgraph`, `langgraph_to_a2a`, `combine_ai_messages`

For detailed signatures, see `references/api-reference.md`.
