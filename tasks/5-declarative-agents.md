- We want to be able to create Agents is a declarative way
- We want to be able to created langgraph based agents using YAML and Common Expression Language (CEL)
    - https://cel.dev/
    - NOTE: ALL EXPRESSION EVALUATIONS AND VARIABLE RETRIEVAL AND BINDING SHOULD HAPPEN THROUGH CEL
    - Use a python library for CEL
- We want to use all the primitives we have defined in our framework: Prompts, Tools, Skills, Agents etc
- Declarative Agent is a capability extending LangGraph agent

- Data Binding is done before the execution of the agent.
- Data includes state + configuration

- State
    - A schema for state has to be specified and the state should conform to that schema at all times
    - state acts as the data-binding for this declarative agent

- We provide the following types of nodes:
    - call_llm
        - Call an LLM with a prompt
        - LLM to call can be specified as a registry entry
        - Prompt to use can be specified as a function of "state" using CEL
            - for e.g. if there's a key called "messages" in state, then this node can specify the prompt as:
                - state["messages] + prompt[("id", "version")].instructions
        - tool_binding
            - If tools are specified for this node, then the LLM is bound to the tools before invocation
        - If tools are bound, then the next node must always be ToolNode
            - If not the message history will go into an inconsistent state
            - When an LLM creates an AIMessage with tool_calls, the next message in state.messages should always be a ToolMessage
                - Else downstream calls to LLM services fail or LangGraph validations fail
            - This means that when a call_llm node is specified with tool, then the next node always has to be a tool_node
    - tool_node
        - This node should be explicitly specified by the user only when they want to explicitly call a tool
        - This node must be the next node of call_llm if tools are present
        - since ToolNode in LangGraph expects an AIMessage with tool_call, when the user explicitly wants to call a tool, manually inject this AIMessage before the ToolNode is invoked
    - call_agent
        - Call an agent
        - input to an agent can be crafted using state and CEL
        - any agent registered in the registry can be called
    - data_transform
        - Any data in state can be transformed using CEL
        - The resulting state should conform to the state schema
        - Input is state and output is state
    - set_state
        - Any variable can be set into the state
        - Input is state and output is state
- We provide the ability for users to specify edges:
    - Edges can be static, i.e. from one specified node to other
    - Dynamic edges with if/else if/else can be specified using CEL
    - conditional if/else statements can only use the variables in state


- The graph (nodes, edges), the input/output schema, the entity registration should all happen in one single YAML file
    - So that one gets a full snapshot of the agent by looking at the YAML

- The YAML will look like:

```yaml
agents:
    my-agent:
        graph:
            nodes:
                - plan
                    type: call_llm
                    args:
                        tools:
                            // Everything is referenced through registry
                            - id: ''
                            - version: ''
                - call_my_sub_agent
                    type: call_agent
llms:
    - id, version
    - instructions: ''
tools:
    - id, version
    - url
    - protocol
prompts:
    - 
skills:
    -
```
    