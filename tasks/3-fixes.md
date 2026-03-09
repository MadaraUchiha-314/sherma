- The methods in sherma/entities/agent/base.py need to conform to A2A client
    - https://github.com/a2aproject/a2a-python/blob/main/src/a2a/client/client.py
- Similarly for sherma/a2a/executor.py
    - The ShermaAgentExecutor needs to inherit from: https://github.com/a2aproject/a2a-python/blob/main/src/a2a/server/agent_execution/agent_executor.py
- All the types need to correspond to A2A type definitions

- LangGraph Agent
    - LangGraph agent converts taskId received as args to thread_id that LangGraph requires