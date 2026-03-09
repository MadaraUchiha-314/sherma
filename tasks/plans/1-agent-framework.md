Sherma is a framework for creating LLM based agents.

## Primitives

The following are the primitives that we want in our agent framework. Every part of the agent framework uses these primities as the building blocks to build an agent.

### Registry
- Every entity used in building an agent is referenced using a registry. Every entity must be registered in the registry and must be retrieved using the registry.
- The entities that we want in the registry are:
    - Prompts
    - LLMs
    - Tools
    - Skills
    - Agents
- Every entity in the registry has an `id` and a `version` and must be referred using that. 
- Versioning is semver. Follows minor, major, patch.
    - So entities can be retried by version strings like `1.*` and the registry fetches the latest minor for major version `1` and so on.
- Every entity in the registry can be `local` or `remote`
- When registering an entity, the following modes should be available:
    - register an instance
    - register a factory which returns the instance
    - register an entity using a URI

- RegistryEntry
    - id: str
    - version: str
    - remote: bool
        - whether the entity is remote or local
    - instance
    - factory
    - url: Optional
    - protocol: Optional

- Registry (the base class) must have the following interface:
    - get(id, version) -> RegistryEntry
        - If instance or factory is not None, returns that
        - If remote, uses `fetch` internally to get the entity and return that
        - Else, throws
    - add(id, version, RegistryEntry)
    - update(id, version, RegistryEntry)
    - fetch(id, version) -> RegistryEntry
        - this is applicable only for remote entities
        - abstract method. will be implemented by the specific registry for an entity. ToolRegistry will implement it for Tool. Skill Registry will implement for Skill etc
    - refresh(id, version) -> None
        - this is applicable only for remote entities
        - abstract method. will be implemented by the specific registry for an entity. ToolRegistry will implement it for Tool. Skill Registry will implement for Skill etc
    - NOTE: All these interfaces are async
        - Even the factory which returns the instance can be an async function

- Entity specific registry and entries should be created for each of the entities: Prompts, LLMs, Tools, Skills, Agents
    - Each entity specific registry will inherit from the Registry base class created above
    - Each entity specific registry will have attributes specific to that entry. This is mainly to help fetch the remote entity if the entity is remote.
        - The agent registry will have an optional attribute for agent_card_url
            - A2A is the only protocol for remote agent we support
        - The skill registry will have an optional attribute for skill_card_url
        - The tool registry will have optional attributes for mcp_url and transport
            - MCP is the only protocol remote tool we support
        - The prompt registry will have an optional attribute for prompt_url
            - There's no custom protocol for a prompt. A GET request to the url should return the prompt as text
        - The LLM registry will just have a url which points to an OpenAI compatible API URL
    - NOTE: Instead of having bespoke attributes for each url, you can keep it as a single attribute `url` and add another attribute called `protocol`, so that it's easier to consolidate. Maybe add these as optional properties to `RegistryEntry`
        - For skills and prompts, keep the protocol as `custom` for now
    - ToolRegistry.get will return a Tool
    - PromptRegistry.get will return a Prompt
    - AgentRegistry.get will return an Agent
    - and so on...

### Prompt

Prompt entity will have the folllowing attributes (in addition to id, version):
    - instructions: str

### LLM

LLM entity will have the folllowing attributes (in addition to id, version):
    - model_name

### Tool

Tool entity will have the folllowing attributes (in addition to id, version):
    - function: Callable

- Whether a tool is local or remote is indicated by the `remote` attribute in the registry entry.

#### LangGraph Tool

- LangGraph tool should be an instance of Tool where the function is a `BaseTool` from LangGraph

### Skill

- Skill entity will have the folllowing attributes (in addition to id, version):
    - font_matter
        - has various attributes like: name, description, 
    - body
    - scripts: list[Tool]
    - references: list[Markdown]
    - assets: list[Any]

- NOTE: Markdown is just a type alias to `str`

- Agent Skills docs: https://agentskills.io/what-are-skills
    - Explore thoroughly

### Agent & Sub-Agent

- Agent entity will have the folllowing attributes (in addition to id, version):
    - agent_card: A2A Agent Card (Optional, use get_card to fetch if not present)
        - Agent Card becomes the interface that other entities in the system understand the capabilities of the agent
        - For e.g. an LLM when it plans to solve a task may choose to use an agent based on it's name, descrioption, skills etc

- Both Local and Remote Agents inherit from Agent (base class)

#### Local Agent

- The interface for a local agent is: https://github.com/a2aproject/a2a-python/blob/main/src/a2a/client/client.py
    - Use only the following methods from it:
        - send_message
        - cancel_task
        - get_card
            - this should be present in the Agent (base-class) as both agents will need it
        - Maybe all the methods should be present in the base-class, but need a way to tell the consumers of the framework that other methods are not available.

#### Remote Agent

- The interface for Remote Agent is same as what the A2A client SDK provides
    - https://github.com/a2aproject/a2a-python/blob/main/src/a2a/client/client.py
- For any interaction with the remote agent use the A2A client SDK's functions
- The idea here is that it doesn't matter if the remote agent was created using this framework or not, any agent using the framework should be able to communicate with an agent supporting A2A protocol

## A2A Protocol

- A2A protocol requires everyone to implement `AgentExecutor`
- The framework provides an out of the box implementation so that everyone doesn't have to create one
- The opinions that the framework takes:
    - For every message, the framework create a task (A2A task) if there's no existing task
    - Use `new_task` from A2A sdk to create a new task
    - Use `TaskUpdater` to manage the lifecycle of the task before/after invoking the agent
    - Use the Agent (base class) as the interface to invoke the agent

### A2A Extensions

Coming soon

## Single Agent System

Coming Soon

## Multi Agent System

Coming Soon

## LangGraph Agent

- When user of this framework want to JUST write an agent using LangGraph, they should not be able be worrying about the A2A protocol and the interfaces etc
- Create a LangGraphAgent which extends Agent (base class)
- LangGraphAgent automatically implements the `send_message`, `cancel_task` methods so that consumers of the framework don't have to do so
- LangGraphAgent provides an abstract method `async get_graph() -> CompiledStateGraph` and use that to implement `send_message` and `cancel_task`

- LangGraphAgent also takes LangFuse as the opinionated tracing and observability solution and provides a config based integration into any custom LangFuse Server.

- Since A2A and LangGraph have 2 different models for "messages", create utilities to convert back and forth between LangGraph message and A2A message. Make sure that there is no information loss in the conversion

## Declarative Agent

Coming Soon

## Misc coding principles
- All strings should be part of some Enum. Source of truth for these Enums should be libraries imported like A2A and MCP.
- httpx AsyncClient
    - To make this framework enterprise and production ready, any time an implementation internally does network calls to fetch any resource, it should accept a union of http.AsyncClient or a factory which returns an http.AsyncClient
    - The httpx async client allows consumers of the framework to fully customize the request headers that go to any outbound network calls to fetch agent card or MCP or LangFuse
    - This should be optional and only for performance optimization if the consuming application needs it
    - If not provided, the framework should instantiate one.
    - So that the framework doesn't instantiate many instances of the http async client, the framework should keep a single instance of http client per request context using ContextVar
- Logging
    - Follow whatever is the correct pattern for consuming applications to configure the logger and the framework code should just use the logger