# Declarative Agents

Declarative agents let you define an entire LangGraph agent in a single YAML file -- the graph topology, prompts, LLMs, tools, skills, and routing logic. Dynamic behavior is expressed with [CEL (Common Expression Language)](https://cel.dev/) expressions evaluated against the agent's state at runtime.

## YAML Structure

A declarative agent YAML has these top-level sections:

```yaml
prompts:      # Prompt definitions
llms:         # LLM declarations
tools:        # Tool imports
skills:       # Skill card references
sub_agents:   # Sub-agent declarations (for multi-agent orchestration)
hooks:        # Hook executor imports
agents:       # Agent graph definitions
```

All entity registrations and the graph definition live in one file, giving you a complete snapshot of the agent.

## Entity Declarations

### Prompts

```yaml
prompts:
  - id: my-prompt
    version: "1.0.0"
    instructions: >
      You are a helpful assistant.
      Be concise and accurate.
```

Prompts are accessible in CEL expressions via `prompts["my-prompt"]["instructions"]`.

### LLMs

```yaml
llms:
  - id: openai-gpt-4o-mini
    version: "1.0.0"
    provider: openai        # Currently supports "openai"
    model_name: gpt-4o-mini
```

The LLM provider reads API keys from environment variables (e.g., `OPENAI_API_KEY`).

### Tools

Tools reference Python callables by import path:

```yaml
tools:
  - id: get_weather
    version: "1.0.0"
    import_path: my_package.tools.get_weather
```

The import path should point to a LangChain/LangGraph `@tool`-decorated function.

### Skills

```yaml
skills:
  - id: weather
    version: "1.0.0"
    skill_card_path: path/to/skill-card.json  # Local path
    # url: https://example.com/skill-card.json  # Or remote URL
```

See [Skills](skills.md) for details on skill cards and progressive disclosure.

### Hooks

```yaml
hooks:
  - import_path: my_package.hooks.LoggingHook
  - import_path: my_package.hooks.GuardrailHook
```

Hooks can also be passed programmatically via the `DeclarativeAgent` constructor. See [Hooks](hooks.md).

### Sub-Agents

Declare other agents as sub-agents to enable multi-agent orchestration. Sub-agents are automatically wrapped as LangGraph tools that the supervisor LLM can invoke. See [Multi-Agent](multi-agent.md) for the full guide.

```yaml
sub_agents:
  - id: weather-agent
    version: "1.0.0"
    yaml_path: agents/weather-agent.yaml        # From a YAML file

  - id: search-agent
    version: "1.0.0"
    import_path: my_agents.search_agent          # From a Python module

  - id: remote-agent
    version: "1.0.0"
    url: https://remote-agent.example.com        # Remote A2A agent

  - id: pre-registered-agent
    version: "1.0.0"
    # No source -- expects the agent to already be in the registry
```

## Agent Definition

Each agent is defined under the `agents` key:

```yaml
agents:
  my-agent:
    state:
      fields:
        - name: messages
          type: list
          default: []
        - name: counter
          type: int
          default: 0

    graph:
      entry_point: first_node

      nodes:
        - name: first_node
          type: call_llm
          args: ...

      edges:
        - source: first_node
          target: __end__
```

### State Schema

The `state` section defines the agent's state shape. Supported types: `str`, `int`, `float`, `bool`, `list`, `dict`.

If a field named `messages` is present (type `list`), sherma uses LangGraph's `MessagesState` as the base class, which provides the standard message accumulation behavior.

State fields are available in all CEL expressions.

## Node Types

### `call_llm`

Calls an LLM with a prompt and optional tool bindings.

```yaml
- name: agent
  type: call_llm
  args:
    llm:
      id: openai-gpt-4o-mini
      version: "1.0.0"
    prompt: 'prompts["my-prompt"]["instructions"]'  # CEL expression
    tools:                          # Optional: bind specific tools
      - id: get_weather
        version: "1.0.0"
    # use_tools_from_registry: true       # Or: bind ALL registered tools
    # use_tools_from_loaded_skills: true   # Or: bind tools from loaded skills
    # use_sub_agents_as_tools: true        # Or: bind sub-agents as tools
```

**Tool binding modes** (mutually exclusive):

| Mode | Description |
| --- | --- |
| `tools` (explicit list) | Bind only the listed tools |
| `use_tools_from_registry: true` | Bind all tools in the registry |
| `use_tools_from_loaded_skills: true` | Bind only tools loaded via skill discovery |
| `use_sub_agents_as_tools: true` | Bind sub-agents declared in `sub_agents` as tools |

**Auto-injected tool_node**: When a `call_llm` node has tools, sherma automatically injects a `tool_node` after it with the correct conditional edges. If the LLM responds with tool calls, execution routes to the tool node; otherwise it continues to the next edge. You don't need to wire this manually.

### `tool_node`

Executes tool calls from the last `AIMessage`. Usually auto-injected, but can be declared explicitly:

```yaml
- name: tools
  type: tool_node
  args:
    tools:                  # Optional: restrict to specific tools
      - id: get_weather
        version: "1.0.0"
```

If no `tools` list is provided, the node resolves all tools from the registry.

### `call_agent`

Invokes another registered agent:

```yaml
- name: delegate
  type: call_agent
  args:
    agent:
      id: sub-agent
      version: "1.0.0"
    input: 'messages[size(messages) - 1]'  # CEL expression for input
```

The agent can be local or remote. The input is evaluated as a CEL expression against state, sent as an A2A message, and the response is added to `messages`.

### `data_transform`

Transforms state using a CEL expression that returns a dict:

```yaml
- name: update_stats
  type: data_transform
  args:
    expression: '{"query_count": query_count + 1, "status": "done"}'
```

The returned dict is merged into the state. Only include the keys you want to update.

### `set_state`

Sets individual state variables via CEL expressions:

```yaml
- name: init
  type: set_state
  args:
    values:
      counter: "0"
      status: '"ready"'     # Note: string literals need inner quotes
```

Each value is a CEL expression. String literals must be double-quoted inside the YAML string.

### `interrupt`

Pauses graph execution to request human input:

```yaml
- name: ask_user
  type: interrupt
  args:
    value: '"What would you like to do next?"'  # CEL expression
```

The interrupt value is sent to the client as an A2A `TaskStatusUpdateEvent` with state `input_required`. When the user responds, execution resumes from this node.

## Edges

### Static Edges

```yaml
edges:
  - source: node_a
    target: node_b
```

Use `__end__` as the target to terminate the graph.

### Conditional Edges

Use CEL expressions for dynamic routing:

```yaml
edges:
  - source: reflect
    branches:
      - condition: 'messages[size(messages) - 1].contains("TASK_COMPLETE")'
        target: __end__
      - condition: 'retry_count < 3'
        target: retry
    default: summarize    # Fallback if no branch matches
```

Branches are evaluated in order. The first matching condition determines the target. If no branch matches and no `default` is set, the graph ends.

## CEL Expressions

[CEL](https://cel.dev/) is used throughout the YAML for dynamic behavior. Expressions have access to:

- **State variables**: All fields in the state schema (`messages`, `counter`, etc.)
- **Prompts**: `prompts["prompt-id"]["instructions"]`
- **LLMs**: `llms["llm-id"]["model_name"]`

CEL supports standard operations: arithmetic, string manipulation, list operations (`size()`, indexing), map construction, comparisons, and boolean logic.

### Examples

```yaml
# Access last message content
'messages[size(messages) - 1]'

# Build a dict for state transformation
'{"count": count + 1, "status": "done"}'

# Conditional check
'messages[size(messages) - 1].contains("COMPLETE")'

# Reference a registered prompt
'prompts["my-prompt"]["instructions"]'

# String literal (note inner quotes)
'"hello world"'

# Integer literal
'42'
```

## Loading a Declarative Agent

### From a YAML file

```python
from sherma import DeclarativeAgent

agent = DeclarativeAgent(
    id="my-agent",          # Must match an agent key in the YAML
    version="1.0.0",
    yaml_path="agent.yaml",
)
```

### From a YAML string

```python
agent = DeclarativeAgent(
    id="my-agent",
    version="1.0.0",
    yaml_content=yaml_string,
)
```

### From a parsed config

```python
from sherma import DeclarativeConfig, load_declarative_config

config = load_declarative_config(yaml_path="agent.yaml")
agent = DeclarativeAgent(
    id="my-agent",
    version="1.0.0",
    config=config,
)
```

### With hooks

```python
from my_hooks import LoggingHook, GuardrailHook

agent = DeclarativeAgent(
    id="my-agent",
    version="1.0.0",
    yaml_path="agent.yaml",
    hooks=[LoggingHook(), GuardrailHook()],
)
```

## Complete Example

A skill-aware agent that discovers skills, executes tasks, and reflects on results:

```yaml
prompts:
  - id: discover-skills
    version: "1.0.0"
    instructions: >
      You have access to a catalog of skills. Given the user's request:
      1. Call list_skills to see available skills.
      2. Call load_skill_md for the most relevant skill.
      3. Respond with a brief text summary.

  - id: plan-and-execute
    version: "1.0.0"
    instructions: >
      Based on the loaded skills, plan and execute the user's request.

  - id: reflect
    version: "1.0.0"
    instructions: >
      Review the results. If complete, respond with "TASK_COMPLETE"
      followed by the answer. Otherwise respond "NEEDS_MORE_WORK".

llms:
  - id: openai-gpt-4o-mini
    version: "1.0.0"
    provider: openai
    model_name: gpt-4o-mini

skills:
  - id: weather
    version: "1.0.0"
    skill_card_path: skills/weather/skill-card.json

agents:
  skill-agent:
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
            prompt: 'prompts["discover-skills"]["instructions"]'
            tools:
              - id: list_skills
              - id: load_skill_md

        - name: execute
          type: call_llm
          args:
            llm: { id: openai-gpt-4o-mini, version: "1.0.0" }
            prompt: 'prompts["plan-and-execute"]["instructions"]'
            use_tools_from_loaded_skills: true

        - name: reflect
          type: call_llm
          args:
            llm: { id: openai-gpt-4o-mini, version: "1.0.0" }
            prompt: 'prompts["reflect"]["instructions"]'

      edges:
        - source: discover_skills
          target: execute

        - source: execute
          target: reflect

        - source: reflect
          branches:
            - condition: 'messages[size(messages) - 1].contains("TASK_COMPLETE")'
              target: __end__
          default: execute
```
