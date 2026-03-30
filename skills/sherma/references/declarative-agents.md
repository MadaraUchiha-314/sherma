# Declarative Agents

Declarative agents let you define an entire LangGraph agent in a single YAML file -- the graph topology, prompts, LLMs, tools, skills, and routing logic. Dynamic behavior is expressed with [CEL (Common Expression Language)](https://cel.dev/) expressions evaluated at runtime. State fields are accessed via the `state` prefix (e.g., `state.messages`, `state["counter"]`).

## YAML Structure

A declarative agent YAML has these top-level sections:

```yaml
manifest_version: 1   # Required: schema version (currently 1)
prompts:      # Prompt definitions
llms:         # LLM declarations
tools:        # Tool imports
skills:       # Skill card references
sub_agents:   # Sub-agent declarations (for multi-agent orchestration)
hooks:        # Hook executor imports
checkpointer: # Checkpointer configuration (for state persistence)
default_llm:  # Default LLM for call_llm nodes (optional)
agents:       # Agent graph definitions
```

### Manifest Version

Every declarative agent YAML **must** include a `manifest_version` field as a top-level integer. This tracks the version of the declarative agent schema that the YAML uses, allowing a single `DeclarativeAgent` runtime to handle `agent.yaml` files with varying manifest versions simultaneously.

The current manifest version is **1**. Increment it when breaking changes are made to the schema.

```yaml
manifest_version: 1
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
    skill_card_path: ../skills/weather/skill-card.json  # Relative to YAML file
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
    yaml_path: weather-agent.yaml               # Relative to this YAML file

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

### Checkpointer

The checkpointer enables state persistence across graph invocations, which is required for features like `interrupt` nodes (human-in-the-loop). By default, `DeclarativeAgent` uses an in-memory checkpointer (`MemorySaver`), so you don't need to configure anything for basic usage.

To explicitly declare a checkpointer in YAML:

```yaml
checkpointer:
  type: memory    # In-memory checkpointer (currently the only supported type)
```

You can also pass a checkpointer programmatically via the constructor:

```python
from langgraph.checkpoint.memory import MemorySaver

agent = DeclarativeAgent(
    id="my-agent",
    version="1.0.0",
    yaml_path="agent.yaml",
    checkpointer=MemorySaver(),
)
```

When a checkpointer is active, all graph invocations require a `thread_id` in the config to identify the conversation thread. The `send_message` method handles this automatically using `context_id`, `task_id`, or a generated UUID.

### Default LLM

When multiple `call_llm` nodes use the same LLM, you can set a top-level `default_llm` instead of repeating the `llm` field on every node:

```yaml
manifest_version: 1

default_llm:
  id: openai-gpt-4o-mini

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
            # No llm field -- inherits from default_llm
            prompt:
              - role: system
                content: '"You are helpful."'
              - role: messages
                content: 'state.messages'
      edges: []
```

A step-level `llm` always takes precedence over `default_llm`. If neither is set, graph construction raises an error.

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

State fields are accessed in CEL expressions via the `state` prefix: `state.messages`, `state["counter"]`, etc.

## Node Types

### `call_llm`

Calls an LLM with a prompt and optional tool bindings. The `llm` field can be omitted when a top-level `default_llm` is configured (see [Default LLM](#default-llm)).

```yaml
- name: agent
  type: call_llm
  args:
    llm:                            # Optional when default_llm is set
      id: openai-gpt-4o-mini
      version: "1.0.0"
    prompt:
      - role: system
        content: 'prompts["my-prompt"]["instructions"]'
      - role: messages
        content: 'state.messages'
    tools:                          # Optional: bind specific tools
      - id: get_weather
        version: "1.0.0"
    # use_tools_from_registry: true       # Or: bind ALL registered tools
    # use_tools_from_loaded_skills: true   # Or: bind tools from loaded skills
    # use_sub_agents_as_tools: true        # Or: bind all sub-agents as tools
    # use_sub_agents_as_tools:              # Or: bind specific sub-agents
    #   - id: weather-agent
    #     version: "1.0.0"
```

#### Prompt Format

The `prompt` field is an array of message items. Each item has a `role` and a `content` (a CEL expression):

| Role | Behavior |
| --- | --- |
| `system` | CEL evaluates to a string, wrapped as a `SystemMessage` |
| `human` | CEL evaluates to a string, wrapped as a `HumanMessage` |
| `ai` | CEL evaluates to a string, wrapped as an `AIMessage` |
| `messages` | CEL evaluates to a list of messages, **spliced in place** preserving their original roles |

The `messages` role is how you inject conversation history into the prompt. State messages are **never** auto-injected -- you must explicitly include them with `role: messages`. This gives you full control over where conversation history appears relative to system instructions and other messages.

```yaml
# Typical pattern: system prompt, then conversation history
prompt:
  - role: system
    content: 'prompts["my-prompt"]["instructions"]'
  - role: messages
    content: 'state.messages'

# Advanced: inject history in the middle, add a trailing instruction
prompt:
  - role: system
    content: 'prompts["sys"]["instructions"]'
  - role: messages
    content: 'state.messages'
  - role: human
    content: '"Now summarize the above conversation"'

# Few-shot examples via explicit roles
prompt:
  - role: system
    content: '"You classify sentiment as positive or negative."'
  - role: human
    content: '"I love this product!"'
  - role: ai
    content: '"positive"'
  - role: messages
    content: 'state.messages'
```

**Tool binding modes:**

| Mode | Description |
| --- | --- |
| `tools` (explicit list) | Bind the listed tools (can be combined with any flag below) |
| `use_tools_from_registry: true` | Bind all tools in the registry |
| `use_tools_from_loaded_skills: true` | Bind only tools loaded via skill discovery |
| `use_sub_agents_as_tools: true` / `all` | Bind **all** sub-agents declared in `sub_agents` as tools |
| `use_sub_agents_as_tools: [list]` | Bind **specific** sub-agents by `id`/`version` |

`use_sub_agents_as_tools` accepts three forms: `true` (or `all`) to bind all declared sub-agents, a list of `RegistryRef` objects (`id` + `version`) to bind a specific subset, or `false` (default) to disable. Each ref's `id` must match a declared `sub_agents` entry.

The dynamic flags (`use_tools_from_registry`, `use_tools_from_loaded_skills`, `use_sub_agents_as_tools`) are mutually exclusive with each other. However, an explicit `tools` list can be combined with any single dynamic flag -- the tools are merged additively and deduplicated by name.

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
    input: 'state.messages[size(state.messages) - 1]'  # CEL expression for input
```

The agent can be local or remote. The input is evaluated as a CEL expression against state, sent as an A2A message, and the response is added to `messages`.

### `data_transform`

Transforms state using a CEL expression that returns a dict:

```yaml
- name: update_stats
  type: data_transform
  args:
    expression: '{"query_count": state.query_count + 1, "status": "done"}'
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

Pauses graph execution to request human input. The `value` argument is a **required** CEL expression that is evaluated against the current state to produce the interrupt value:

```yaml
- name: ask_user
  type: interrupt
  args:
    value: '"What is your name?"'
```

The CEL expression can reference state, enabling structured metadata:

```yaml
- name: ask_approval
  type: interrupt
  args:
    value: '{"type": "approval", "draft": state.messages[size(state.messages) - 1].content, "actions": ["approve", "reject"]}'
```

When the user responds, execution resumes from this node. The user's response is wrapped as a `HumanMessage` and appended to state.

### `load_skills`

Programmatically loads skills by evaluating a CEL expression to get a list of skill IDs. For each skill, it loads the SKILL.md, registers tools, and synthesizes `AIMessage(tool_calls)` + `ToolMessage` pairs into `state.messages` — making the result indistinguishable from progressive disclosure to downstream nodes.

```yaml
- name: load_selected_skills
  type: load_skills
  args:
    skill_ids: 'json(state.messages[size(state.messages) - 1].content)["skills"]'
```

The `skill_ids` CEL expression must evaluate to a list of objects with `id` (required) and `version` (optional, defaults to `"*"`) keys:

```python
# Example CEL result:
[{"id": "weather", "version": "1.0.0"}, {"id": "calendar"}]
```

Loaded tools are tracked in `__sherma__.loaded_tools_from_skills` and can be used by downstream `call_llm` nodes with `use_tools_from_loaded_skills: true`. If a skill fails to load, it is skipped with a warning and remaining skills continue loading.

**When to use `load_skills` vs progressive disclosure:**
- Use `load_skills` when the agent needs skills loaded before the planning node runs (e.g., a structured-output LLM selects skills upfront).
- Use progressive disclosure (`list_skills` / `load_skill_md` tools) when the LLM should discover and load skills dynamically during conversation.

### `custom`

A node whose logic is defined entirely by hooks. The `custom` node type has no built-in behavior — it fires `node_enter` → `node_execute` → `node_exit`, and the `node_execute` hook (unique to custom nodes) provides the execution logic.

This is the escape hatch for procedural logic that doesn't fit declarative node types (complex token counting, custom API calls with auth, stateful computations), while keeping the YAML purely declarative.

```yaml
- name: summarize_if_needed
  type: custom
  args:
    metadata:              # Optional: arbitrary data accessible to hooks
      description: "Summarize long conversations"
```

The corresponding hook:

```python
from sherma.hooks import BaseHookExecutor, NodeExecuteContext

class SummarizationHook(BaseHookExecutor):
    async def node_execute(self, ctx: NodeExecuteContext) -> NodeExecuteContext | None:
        if ctx.node_name == "summarize_if_needed":
            messages = ctx.state["messages"]
            ctx.result = {
                "summary_messages": await do_summarization(messages),
                "summarized_until": len(messages),
            }
            return ctx
        return None
```

The returned `result` dict is merged into state (same semantics as `data_transform`). Hook metadata is accessible via `ctx.node_context.node_def.args.metadata`.

## Error Handling (`on_error`)

Nodes can declare an `on_error` block for retry and fallback routing:

```yaml
- name: agent
  type: call_llm
  args:
    llm: { id: gpt-4o }
    prompt:
      - role: system
        content: '"You are helpful."'
      - role: messages
        content: state.messages
  on_error:
    retry:
      max_attempts: 3       # total attempts (1 initial + 2 retries)
      strategy: exponential  # "fixed" | "exponential"
      delay: 1.0             # base delay in seconds
      max_delay: 30.0        # cap for exponential backoff
    fallback: error_handler  # node to route to on failure
```

### Support Matrix

| Node type | `retry` | `fallback` |
|-----------|:-------:|:----------:|
| `call_llm` | Yes | Yes |
| `tool_node` | No | Yes |
| `call_agent` | No | Yes |
| `data_transform` | No | No |
| `set_state` | No | No |
| `interrupt` | No | No |
| `load_skills` | No | No |
| `custom` | No | Yes |

- **`retry`** is only supported on `call_llm` because the retry wraps only the `model.ainvoke()` call (stateless, safe to retry). Other node types may have side effects.
- **`fallback`** is supported on IO-bound nodes (`call_llm`, `tool_node`, `call_agent`, `custom`). When retries are exhausted (or on first failure for nodes without retry), execution routes to the fallback node instead of crashing.

### Error State

When an error triggers fallback routing, error details are stored in `state["__sherma__"]["last_error"]`:

```python
{
    "node": "agent",           # node that failed
    "type": "RateLimitError",  # exception class name
    "message": "Rate limit exceeded",
    "attempt": 3,              # which attempt failed
}
```

This is accessible in CEL for downstream error handler nodes.

### Interaction with `on_node_error` Hook

The `on_node_error` hook runs only when declarative `on_error` does not handle the error:

1. Exception occurs
2. Retry `model.ainvoke()` (if `call_llm` with `retry`)
3. Retries exhausted - store error in `__sherma__`
4. If `fallback` configured - route to fallback node (hook **not** called)
5. If no fallback - call `on_node_error` hook - re-raise if not consumed

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
      - condition: 'state.messages[size(state.messages) - 1].contains("TASK_COMPLETE")'
        target: __end__
      - condition: 'state.retry_count < 3'
        target: retry
    default: summarize    # Fallback if no branch matches
```

Branches are evaluated in order. The first matching condition determines the target. If no branch matches and no `default` is set, the graph ends.

## CEL Expressions

[CEL](https://cel.dev/) is used throughout the YAML for dynamic behavior. Expressions have access to:

- **State variables**: Accessed via `state.field` or `state["field"]` (e.g., `state.messages`, `state["counter"]`)
- **Prompts**: `prompts["prompt-id"]["instructions"]`
- **LLMs**: `llms["llm-id"]["model_name"]`
- **Skills**: `skills["skill-id"]["name"]`, `skills["skill-id"]["description"]` (when skills are declared)

State fields are always accessed through the `state` prefix. Extra variables like `prompts`, `llms`, and `skills` remain at the top level.

CEL supports standard operations: arithmetic, string manipulation, list operations (`size()`, indexing, `filter`, `exists`, `map`), map construction, comparisons, and boolean logic.

CEL can also handle Pydantic models, dataclasses, and any object with `__dict__` -- these are automatically converted to CEL maps, so you can access their fields with standard map syntax (e.g., `obj.field` or `obj["field"]`).

### List Macros (built-in)

CEL provides built-in macros for filtering, searching, and transforming lists. These support variable binding and predicate expressions natively:

| Macro | Syntax | Description |
| --- | --- | --- |
| `filter` | `list.filter(x, predicate)` | Returns elements matching the predicate |
| `exists` | `list.exists(x, predicate)` | Returns `true` if any element matches |
| `all` | `list.all(x, predicate)` | Returns `true` if all elements match |
| `exists_one` | `list.exists_one(x, predicate)` | Returns `true` if exactly one element matches |
| `map` | `list.map(x, expr)` | Transforms each element |

```yaml
# Filter messages by type
'state.messages.filter(m, m["type"] == "human")'

# Check if any message matches a condition
'state.messages.exists(m, m["type"] == "ai" && m["content"].contains("COMPLETE"))'

# Check if all items satisfy a predicate
'state.items.all(x, x > 0)'

# Count matching elements
'size(state.messages.filter(m, m["additional_kwargs"]["type"] == "approval_decision")) > 0'

# Extract a field from each element
'state.messages.map(m, m["type"])'
```

### Custom Functions

In addition to standard CEL built-ins, sherma provides custom functions inspired by [agentgateway's CEL extensions](https://agentgateway.dev/docs/standalone/latest/reference/cel/).

#### JSON Functions

| Function | Description | Example |
| --- | --- | --- |
| `json(string)` | Parse a JSON string into a CEL map or list | `json(state.content)["status"]` |
| `jsonValid(string)` | Check whether a string is valid JSON | `jsonValid(state.data)` |

```yaml
# Parse JSON from message content and access a field
'json(state.messages[size(state.messages) - 1]["content"])["action"]'

# Guard: only route if content is valid JSON with the right field
'jsonValid(state.response) && json(state.response)["status"] == "complete"'

# Parse a JSON array
'json(state.items_json)[0]'
```

#### Safe Access

| Function | Description | Example |
| --- | --- | --- |
| `default(expr, fallback)` | Return *fallback* if *expr* errors | `default(json(state.data)["key"], "none")` |

`default()` catches evaluation errors in the first argument (missing keys, invalid JSON, etc.) and returns the fallback value instead. This is especially useful with `json()` for resilient routing:

```yaml
# Extract action from JSON response, fall back to "continue"
'default(json(state.response)["action"], "continue")'

# Safe nested access
'default(json(state.body)["result"]["confidence"], 0.0)'

# Safe state access
'default(state.retry_count, 0)'
```

#### String Extensions

Aligned with the [cel-go strings extension](https://pkg.go.dev/github.com/google/cel-go/ext#Strings):

| Function | Description | Example |
| --- | --- | --- |
| `split(string, separator)` | Split string into a list | `"a,b,c".split(",")` |
| `trim(string)` | Strip leading/trailing whitespace | `state.input.trim()` |
| `lowerAscii(string)` | Convert to lowercase | `state.name.lowerAscii()` |
| `upperAscii(string)` | Convert to uppercase | `state.name.upperAscii()` |
| `replace(string, old, new)` | Replace all occurrences | `state.text.replace("old", "new")` |
| `indexOf(string, substr)` | Index of first occurrence (-1 if not found) | `state.text.indexOf("needle")` |
| `join(list, separator)` | Join list elements into a string | `state.items.join(", ")` |
| `substring(string, start, end)` | Extract substring (start inclusive, end exclusive) | `state.text.substring(0, 10)` |

```yaml
# Split tags and rejoin with different separator
'state.tags.split(",").join(" | ")'

# Trim and lowercase for normalization
'state.input.trim().lowerAscii()'

# Combine JSON parsing with string functions
'json(state.data.trim())["name"].lowerAscii()'
```

#### Templating

| Function | Description | Example |
| --- | --- | --- |
| `template(string, map)` | Substitute `${key}` placeholders from a map | `template("Hello ${name}!", {"name": "world"})` |

Unresolved placeholders (keys not in the map) are left as-is. Non-string values are coerced to strings.

```yaml
# Inject state into a prompt template
'template(prompts["plan-prompt"]["instructions"], {"skill_instructions": state.skill_instructions})'

# Multiple placeholders
'template("Hello ${name}, your role is ${role}.", {"name": state.user, "role": state.assigned_role})'

# Non-string values are coerced
'template("Count: ${n}, Active: ${flag}", {"n": state.count, "flag": state.active})'
```

#### List Utilities

| Function | Description | Example |
| --- | --- | --- |
| `last(list)` | Return the last element of a list (error if empty) | `last(state.items)` |

Combine `last()` with the built-in `filter()` macro to implement a **findLast** pattern:

```yaml
# Find the last human message
'last(state.messages.filter(m, m["type"] == "human"))'

# Find the last approval decision, with safe fallback
'default(last(state.messages.filter(m, m["additional_kwargs"]["type"] == "approval_decision"))["content"], "")'

# Route based on whether the last matching message exists
'default(last(state.messages.filter(m, m["type"] == "ai"))["content"], "") != ""'
```

All custom functions can be called both as functions (`json(x)`) and as methods (`x.json()`).

### Examples

```yaml
# Access last message content
'state.messages[size(state.messages) - 1]'

# Build a dict for state transformation
'{"count": state.count + 1, "status": "done"}'

# Conditional check
'state.messages[size(state.messages) - 1].contains("COMPLETE")'

# Reference a registered prompt (top-level, no state prefix)
'prompts["my-prompt"]["instructions"]'

# String literal (note inner quotes)
'"hello world"'

# Integer literal
'42'

# Filter and check for matching messages (routing pattern)
'state.messages.exists(m, m["additional_kwargs"]["type"] == "approval_decision")'

# Find last matching message content with fallback
'default(last(state.messages.filter(m, m["additional_kwargs"]["type"] == "approval_decision"))["content"], "")'
```

### Message Metadata Access

LangChain message objects are automatically converted to CEL maps, exposing all public fields including `content`, `type`, `additional_kwargs`, `tool_calls`, and more.

#### Message type

Use `type` to distinguish message roles (`"ai"`, `"human"`, `"system"`, `"tool"`):

```yaml
# Route based on whether the last message is from the AI
'state.messages[size(state.messages) - 1]["type"] == "ai"'
```

#### `additional_kwargs`

Messages carry arbitrary metadata in `additional_kwargs`. Access nested values with standard map syntax:

```yaml
# Check a custom metadata tag on the last message
'state.messages[size(state.messages) - 1]["additional_kwargs"]["type"] == "approval_decision"'

# Access nested metadata (e.g., A2A metadata)
'state.messages[0]["additional_kwargs"]["a2a_metadata"]["taskId"]'

# Route based on metadata
edges:
  - source: get_approval
    branches:
      - condition: >
          state.messages[size(state.messages) - 1]["additional_kwargs"]["type"] == "approval_decision"
        target: handle_approval
    default: continue
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
    base_path=Path("path/to/yaml/dir"),  # Required for relative file paths
)
```

When using `yaml_content`, relative file paths in the YAML (like `skill_card_path` or sub-agent `yaml_path`) cannot be resolved without a `base_path`. If your YAML references only absolute paths or Python import paths, `base_path` is not needed.

### From a parsed config

```python
from sherma import DeclarativeConfig, load_declarative_config

config = load_declarative_config(yaml_path="agent.yaml")
agent = DeclarativeAgent(
    id="my-agent",
    version="1.0.0",
    config=config,
    base_path=Path("path/to/yaml/dir"),  # Required for relative file paths
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

## Path Resolution

All file paths in a YAML config (`skill_card_path`, sub-agent `yaml_path`) are resolved against a **`base_path`**:

- **`yaml_path` provided**: `base_path` is automatically derived from the YAML file's parent directory. No manual setup needed.
- **`yaml_content` or `config` provided**: Set `base_path` explicitly if the YAML contains relative file paths.
- **Absolute paths**: Always work regardless of `base_path`.
- **Relative paths without `base_path`**: Raise a `DeclarativeConfigError`.

This ensures agents work correctly from any working directory, not just the project root.

```yaml
# These paths are resolved relative to the YAML file's directory:
skills:
  - id: weather
    version: "1.0.0"
    skill_card_path: ../skills/weather/skill-card.json  # Relative to YAML dir

sub_agents:
  - id: weather-agent
    version: "1.0.0"
    yaml_path: weather_agent.yaml  # Relative to YAML dir
```

**What is NOT affected by `base_path`:**
- `import_path` (tools, agents, hooks) -- uses Python's `importlib` and `sys.path`
- Skill card `base_uri` -- resolved relative to the skill card file's own location
- Remote URLs -- used as-is

## Complete Example

A skill-aware agent that discovers skills, executes tasks, and reflects on results. Note the use of `default_llm` to avoid repeating the LLM reference on every node:

```yaml
manifest_version: 1

prompts:
  - id: discover-skills
    version: "1.0.0"
    instructions: >
      You have access to a catalog of skills. Here are the available skills:
      ${available_skills}

      Given the user's request:
      1. Call load_skill_md for the most relevant skill from the catalog above.
      2. Respond with a brief text summary.

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

default_llm:
  id: openai-gpt-4o-mini

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
            prompt:
              - role: system
                content: 'template(prompts["discover-skills"]["instructions"], {"available_skills": string(skills)})'
              - role: messages
                content: 'state.messages'
            tools:
              - id: load_skill_md
              - id: unload_skill

        - name: execute
          type: call_llm
          args:
            prompt:
              - role: system
                content: 'prompts["plan-and-execute"]["instructions"]'
              - role: messages
                content: 'state.messages'
            use_tools_from_loaded_skills: true

        - name: reflect
          type: call_llm
          args:
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
            - condition: 'state.messages[size(state.messages) - 1].contains("TASK_COMPLETE")'
              target: __end__
          default: execute
```

## Complete Example: Human-in-the-Loop Approval

An approval workflow that uses message metadata for routing. A hook tags human responses with `additional_kwargs["decision"]`, and CEL edges inspect that metadata to approve or loop back for revision:

```yaml
manifest_version: 1

prompts:
  - id: draft-prompt
    version: "1.0.0"
    instructions: >
      Draft a response to the user's request. Be thorough.

  - id: revise-prompt
    version: "1.0.0"
    instructions: >
      The reviewer asked for changes. Revise your draft accordingly.

llms:
  - id: openai-gpt-4o-mini
    version: "1.0.0"
    provider: openai
    model_name: gpt-4o-mini

default_llm:
  id: openai-gpt-4o-mini

hooks:
  - import_path: my_package.hooks.ApprovalTaggingHook

agents:
  approval-agent:
    state:
      fields:
        - name: messages
          type: list
          default: []

    graph:
      entry_point: draft
      nodes:
        - name: draft
          type: call_llm
          args:
            prompt:
              - role: system
                content: 'prompts["draft-prompt"]["instructions"]'
              - role: messages
                content: 'state.messages'

        # Pause for human review
        - name: get_approval
          type: interrupt
          args:
            value: >
              {"type": "approval", "draft": state.messages[size(state.messages) - 1]["content"]}

        - name: revise
          type: call_llm
          args:
            prompt:
              - role: system
                content: 'prompts["revise-prompt"]["instructions"]'
              - role: messages
                content: 'state.messages'

      edges:
        - source: draft
          target: get_approval

        # Route using additional_kwargs metadata set by a hook
        - source: get_approval
          branches:
            - condition: >
                state.messages[size(state.messages) - 1]["additional_kwargs"]["decision"] == "approve"
              target: __end__
          default: revise

        # After revision, go back for another review
        - source: revise
          target: get_approval
```

The `ApprovalTaggingHook` sets `additional_kwargs["decision"]` on the human message during `node_exit` of the interrupt node. You can also route on the message `type` field — for example, `state.messages[0]["type"] == "human"` returns `true` for `HumanMessage` objects.
