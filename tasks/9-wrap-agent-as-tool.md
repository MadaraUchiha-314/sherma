- In order to effectively orchestract a multi-agent system, a supervisor agent needs to plan across: tools, skills, sub-agents available to it
- Since LLMs plan by outputting tool_calls, we need to wrap agents as tools

- Create a wrapper function which takes an agent and wraps it as a LangGraph tool

- Provide an config in Declarative Agent to automatically wrap sub-agents as tools.

- Provide an option in Declarative Agent to declare sub-agents
    - Sub-Agents can be declared as full-fledged entites like agents are or can be just referenced using id, version and expecting that they would have been already registered
    - Since agents can also declare an input schema as an extension, when we wrap the agent as a tool, we need to convey that input schema.
    - Structured Tool in langgraph has that feature to provide a custom input schema
    - So we need to combine message which is the default input for an agent and input schema
    - may be we need to create a tool that takes 2 args: one for the input schema and one for message
    - `my_agent_as_tool(request: Message, agent_input: AgentInputSchema)`
        - something like the above
        - and then we need to combine request and add agent_input as a part.
