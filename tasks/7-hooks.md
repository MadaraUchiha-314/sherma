- We want to give programmatic control to developers who are writing LangGraph and Declarative Agents
- Design a comprehensive hooks system where developers can register hooks for various lifecycles:
    - before llm call
        - allows complete control of the input that goes to the LLM
    - after llm call
        - allows complete control of the response that is received from the LLM
    - before tool call
        - allows complete control of the tool call that happens
    - after tool call
        - allows complete control of the response of a tool call
    - before agent call
        - allows complete control of the agent invocation
    - after agent call
        - allows complete control of the agent's response
    - before/after skill load
    - node enter and exit
    - before/after interrupt

- create a "type" for each hook
- multiple hooks can be registered for each type and they will be executed in the order that they were registered
- all hooks should be async

- Create the concept of a hook executor which can be registered
- A hook executor will have all the hook methods in a single class/interface/protocol (choose whatever is correct)

- If a hook returns None, that means that it's a no-op
- All hooks return None by default