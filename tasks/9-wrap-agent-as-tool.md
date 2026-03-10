- In order to effectively orchestract a multi-agent system, a supervisor agent needs to plan across: tools, skills, sub-agents available to it
- Since LLMs plan by outputting tool_calls, we need to wrap agents as tools

- Create a wrapper function which takes an agent and wraps it as a LangGraph tool

- Provide an config in Declarative Agent to automatically wrap sub-agents as tools.

- Provide an option in Declarative Agent to declare sub-agents
    - Sub-Agents can be declared as full-fledged entites like agents are or can be just referenced using id, version and expecting that they would have been already registered
    