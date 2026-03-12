# Figma AI Prompt: Visual Graph Builder for Sherma Declarative Agents

## 1. Product Overview

Design a **visual graph builder** UI for creating declarative AI agents. Users build agent workflows by dragging nodes onto a canvas, connecting them with edges, and configuring each node's properties in a detail panel. The output is a YAML configuration file that defines the complete agent.

**Core interactions:**
- **Drag & drop** node types from a palette onto an infinite canvas
- **Connect nodes** by drawing edges between them (static or conditional)
- **Click a node** to open a configuration panel with a form for that node's properties
- **View YAML** for the selected node or the entire agent system
- **Manage resources** (prompts, LLMs, tools, skills, sub-agents, hooks, checkpointer) in a sidebar panel

The graph is a directed acyclic/cyclic workflow. Special node names `__start__` and `__end__` represent the graph entry and exit points.

---

## 2. Node Types

There are **6 node types** users can place on the canvas. Each has a distinct purpose and configuration schema.

### 2.1 `call_llm`

**Purpose:** Calls a large language model with a prompt and optional tool bindings.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | Unique node identifier |
| `llm` | RegistryRef (`{id, version}`) | yes | Reference to a declared LLM |
| `prompt` | CEL string | yes | CEL expression evaluated against state (e.g. `prompts["my-prompt"]["instructions"]`) |
| `tools` | list of RegistryRef | no | Explicit list of tools to bind |
| `use_tools_from_registry` | boolean | no | Bind all registered tools |
| `use_tools_from_loaded_skills` | boolean | no | Bind tools from loaded skills |
| `use_sub_agents_as_tools` | boolean | no | Bind sub-agents as callable tools |
| `response_format` | ResponseFormat | no | Structured output schema (`{name, description, schema}`) |

**Notes:**
- The three `use_*` boolean flags are mutually exclusive with each other, but can be combined with an explicit `tools` list.
- When tools are bound, the system auto-injects a `tool_node` after this node with conditional routing (if LLM returns tool calls → tool_node, otherwise → next edge).

**Example YAML:**
```yaml
- name: agent
  type: call_llm
  args:
    llm:
      id: openai-gpt-4o-mini
      version: "1.0.0"
    prompt: 'prompts["weather-system-prompt"]["instructions"]'
    tools:
      - id: get_weather
        version: "1.0.0"
```

### 2.2 `tool_node`

**Purpose:** Executes tool calls from the last AI message. Usually auto-injected after `call_llm`, but can be placed explicitly.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | Unique node identifier |
| `tools` | list of RegistryRef | no | Restrict execution to specific tools (if omitted, uses all registry tools) |

**Example YAML:**
```yaml
- name: tools
  type: tool_node
  args:
    tools:
      - id: get_weather
        version: "1.0.0"
```

### 2.3 `call_agent`

**Purpose:** Invokes another registered agent (local or remote).

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | Unique node identifier |
| `agent` | RegistryRef (`{id, version}`) | yes | Reference to a sub-agent |
| `input` | CEL string | yes | CEL expression for the input to send (e.g. `messages[size(messages) - 1]`) |

**Example YAML:**
```yaml
- name: delegate
  type: call_agent
  args:
    agent:
      id: sub-agent
      version: "1.0.0"
    input: 'messages[size(messages) - 1]'
```

### 2.4 `data_transform`

**Purpose:** Transforms state using a CEL expression that returns a dictionary. The returned keys are merged into the agent state.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | Unique node identifier |
| `expression` | CEL string | yes | CEL expression returning a dict (e.g. `{"count": count + 1, "status": "done"}`) |

**Example YAML:**
```yaml
- name: update_stats
  type: data_transform
  args:
    expression: '{"query_count": query_count + 1, "last_status": "done"}'
```

### 2.5 `set_state`

**Purpose:** Sets individual state variables via CEL expressions.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | Unique node identifier |
| `values` | dict of `field_name → CEL string` | yes | Each value is a CEL expression. String literals need inner quotes (e.g. `'"ready"'`). |

**Example YAML:**
```yaml
- name: init
  type: set_state
  args:
    values:
      query_count: "0"
      last_status: '"ready"'
```

### 2.6 `interrupt`

**Purpose:** Pauses graph execution for human-in-the-loop input. The last AI message is sent to the user; when they respond, execution resumes.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | Unique node identifier |
| `value` | CEL string | no | Fallback value if no AI message exists (rarely used) |

**Example YAML:**
```yaml
- name: ask_user
  type: interrupt
  args: {}
```

---

## 3. Edge Types

Edges connect nodes in the graph. There are **2 types**.

### 3.1 Static Edge

A simple directed connection from one node to another.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `source` | string | yes | Source node name |
| `target` | string | yes | Target node name (use `__end__` to terminate) |

**Visual:** A solid arrow from source to target.

**Example YAML:**
```yaml
- source: init
  target: agent
```

### 3.2 Conditional Edge

Routes to different targets based on CEL conditions evaluated in order.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `source` | string | yes | Source node name |
| `branches` | list of `{condition: CEL string, target: string}` | yes | Ordered list of condition/target pairs |
| `default` | string | no | Fallback target if no branch matches (if omitted, graph ends) |

**Visual:** Multiple arrows from source, each labeled with its condition. One arrow labeled "default" for the fallback.

**Built-in condition:** `has_tool_calls` — used by auto-injected edges after `call_llm` nodes with tools.

**Example YAML:**
```yaml
- source: reflect
  branches:
    - condition: 'messages[size(messages) - 1].contains("TASK_COMPLETE")'
      target: __end__
    - condition: 'retry_count < 3'
      target: retry
  default: summarize
```

---

## 4. Resource Types (Sidebar Panel)

Resources are declared at the top level of the YAML and referenced by nodes. Design a sidebar or panel with **7 tabs/sections** for managing these.

### 4.1 Prompts

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | yes | Unique identifier |
| `version` | string | no | Semver, defaults to `"*"` |
| `instructions` | string (multiline) | yes | The prompt text |

### 4.2 LLMs

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | yes | Unique identifier |
| `version` | string | no | Semver, defaults to `"*"` |
| `provider` | string | no | Provider name, defaults to `"openai"` |
| `model_name` | string | yes | Model identifier (e.g. `gpt-4o-mini`) |

### 4.3 Tools

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | yes | Unique identifier |
| `version` | string | no | Semver, defaults to `"*"` |
| `import_path` | string | no | Python import path (e.g. `my_package.tools.get_weather`) |
| `url` | string | no | Remote tool URL |
| `protocol` | string | no | Protocol for remote tools |

### 4.4 Skills

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | yes | Unique identifier |
| `version` | string | no | Semver, defaults to `"*"` |
| `url` | string | no | Remote skill card URL |
| `skill_card_path` | string | no | Local path to skill card JSON |

### 4.5 Sub-Agents

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | yes | Unique identifier |
| `version` | string | no | Semver, defaults to `"*"` |
| `url` | string | no | Remote A2A agent URL |
| `import_path` | string | no | Python import path |
| `yaml_path` | string | no | Path to another declarative YAML |

### 4.6 Hooks

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `import_path` | string | yes | Python import path to hook class |

### 4.7 Checkpointer

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | enum: `"memory"` | yes | Currently only `"memory"` is supported |

---

## 5. Agent-Level Configuration

Each agent has top-level configuration in addition to its graph.

### 5.1 State Schema

Defines the agent's state shape. Displayed as an editable table/list.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | Field name |
| `type` | enum: `str`, `int`, `float`, `bool`, `list`, `dict` | no | Defaults to `str` |
| `default` | any | no | Default value |

**Special behavior:** If a field named `messages` with type `list` exists, the system uses LangGraph's `MessagesState` base class for message accumulation.

### 5.2 Graph Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `entry_point` | string | yes | Name of the first node to execute |
| `recursion_limit` | int | no | Max graph recursion depth |
| `max_concurrency` | int | no | Max parallel node execution |
| `tags` | list of strings | no | Tags for tracing/filtering |
| `metadata` | dict | no | Arbitrary key-value metadata |

### 5.3 Input/Output Schema

Optional JSON Schema objects (`input_schema`, `output_schema`) that define the agent's external interface.

---

## 6. UI Views to Design

### 6.1 Canvas View (main area)

- Infinite zoomable/pannable canvas
- Nodes rendered as distinct cards/shapes (visually differentiated by type)
- Edges rendered as arrows (solid for static, branching for conditional)
- Drag nodes from a **node palette** onto the canvas
- `__start__` and `__end__` are special pseudo-nodes always present
- Entry point edge from `__start__` to the `entry_point` node

### 6.2 Node Detail Panel (right side)

- Opens when a node is clicked
- Shows a form matching the node type's schema (see Section 2)
- RegistryRef fields should be dropdowns populated from declared resources
- CEL expression fields should be code-editor-style text inputs
- Boolean flags as toggle switches

### 6.3 Per-Node YAML View

- Toggle on the detail panel to show the YAML snippet for the selected node
- Read-only, syntax-highlighted YAML

### 6.4 Full System YAML View

- A toggleable full-screen or split-pane view showing the complete agent YAML
- Includes all resources, agents, state, graph, nodes, and edges
- Read-only, syntax-highlighted

### 6.5 Resource Sidebar (left side)

- Tabbed or accordion sections for each of the 7 resource types (Section 4)
- Each section lists declared resources with add/edit/delete actions
- Clicking a resource opens an edit form
- Resources are referenced by nodes via their `id`

### 6.6 Agent Config Panel

- Accessible from a top-bar or settings icon
- State schema editor (add/remove/edit fields)
- Graph config fields (entry_point dropdown, recursion_limit, etc.)
- Support for multiple agents in a single YAML (tabs or dropdown to switch)

---

## 7. Example YAML Snippets

These complete examples show what the UI produces. Use them to understand the full data model.

### 7.1 Simple Weather Agent

```yaml
prompts:
  - id: weather-system-prompt
    version: "1.0.0"
    instructions: >
      You are a helpful weather assistant.
      Use the get_weather tool to look up current weather
      for any city the user asks about.
      Provide a concise, friendly response.

llms:
  - id: openai-gpt-4o-mini
    version: "1.0.0"
    provider: openai
    model_name: gpt-4o-mini

tools:
  - id: get_weather
    version: "1.0.0"
    import_path: examples.tools.get_weather

agents:
  weather-agent:
    state:
      fields:
        - name: messages
          type: list
          default: []
        - name: query_count
          type: int
          default: 0
        - name: last_status
          type: str
          default: ""

    graph:
      entry_point: init

      nodes:
        - name: init
          type: set_state
          args:
            values:
              query_count: "0"
              last_status: '"ready"'

        - name: agent
          type: call_llm
          args:
            llm:
              id: openai-gpt-4o-mini
              version: "1.0.0"
            prompt: 'prompts["weather-system-prompt"]["instructions"]'
            tools:
              - id: get_weather
                version: "1.0.0"

        - name: update_stats
          type: data_transform
          args:
            expression: '{"query_count": query_count + 1, "last_status": "done"}'

        - name: ask_next_place
          type: interrupt
          args: {}

      edges:
        - source: init
          target: agent
        - source: agent
          target: update_stats
        - source: update_stats
          target: ask_next_place
        - source: ask_next_place
          target: agent
```

**Graph visualization:** `__start__` → `init` → `agent` → (auto-injected tool_node loop) → `update_stats` → `ask_next_place` → `agent` (loop)

### 7.2 Multi-Agent Supervisor

```yaml
prompts:
  - id: supervisor-prompt
    version: "1.0.0"
    instructions: >
      You are a travel planning assistant.
      You have access to a weather agent that can look up the current weather for any city.
      When the user asks about travel plans, use the weather agent to check weather conditions
      at the destination and include that information in your recommendations.
      Provide helpful, concise travel advice.

llms:
  - id: openai-gpt-4o-mini
    version: "1.0.0"
    provider: openai
    model_name: gpt-4o-mini

sub_agents:
  - id: weather-agent
    version: "1.0.0"
    yaml_path: weather_agent.yaml

agents:
  travel-planner:
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
            prompt: 'prompts["supervisor-prompt"]["instructions"]'
            use_sub_agents_as_tools: true

      edges:
        - source: planner
          target: __end__
```

**Graph visualization:** `__start__` → `planner` → (auto-injected tool_node for sub-agent calls) → `__end__`

---

## 8. CEL Expression Reference

CEL (Common Expression Language) is used throughout for dynamic behavior. The UI should treat CEL fields as code inputs (monospace font, syntax awareness).

**Available in CEL context:**
- All state fields (e.g. `messages`, `counter`, `query_count`)
- `prompts["<id>"]["instructions"]` — access prompt text
- `llms["<id>"]["model_name"]` — access LLM info
- Standard operations: arithmetic, string ops, `size()`, list indexing, map construction, comparisons, boolean logic

**Common patterns:**
- `prompts["my-prompt"]["instructions"]` — reference a prompt
- `messages[size(messages) - 1]` — last message
- `'{"key": value + 1}'` — dict for state update
- `'"literal string"'` — string literal (inner quotes)
- `'messages[size(messages) - 1].contains("DONE")'` — conditional check

---

## 9. Design Guidelines

- Each of the 6 node types should be **visually distinct** (different color, icon, or shape)
- Conditional edges should show branch labels (the CEL condition or a summary)
- The canvas should feel like a modern graph/flow editor (think: Figma's own FigJam, or tools like Retool Workflows, n8n, LangFlow)
- Resource references (RegistryRef fields) should use dropdowns populated from the sidebar resources
- The YAML views are read-only previews of what the visual builder produces
- Support for **multiple agents** in a single file (tabs or agent switcher)
- `__start__` and `__end__` are always-visible pseudo-nodes on the canvas
