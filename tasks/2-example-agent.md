# Examples

- Add a secrets.json and secrets.example.json and read the OpenAI API Key from that.

## Simple LangGraph Agent
- Create example agent using LangGraph + OpenAI as the LLM provider (use ChatOpenAI from LangGraph).
    - The agent can implement a simple ReAct loop with planning, tool execution (using ToolNode), reflection
    - Add a simple LangGraph tool which makes an API call to weather api to get the weather.
    - The details about the API are in https://open-meteo.com/
- The example script should take a query as an arg and answer that query using the agent.
