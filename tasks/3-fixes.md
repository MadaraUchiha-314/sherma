- The methods in sherma/entities/agent/base.py need to conform to A2A client
    - https://github.com/a2aproject/a2a-python/blob/main/src/a2a/client/client.py
- Similarly for sherma/a2a/executor.py
    - The ShermaAgentExecutor needs to inherit from: https://github.com/a2aproject/a2a-python/blob/main/src/a2a/server/agent_execution/agent_executor.py
- All the types need to correspond to A2A type definitions

- LangGraph Agent
    - LangGraph agent converts taskId received as args to thread_id that LangGraph requires

- Always create the Task at the executor layer:

```py
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    Message,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
)
from a2a.utils.message import new_agent_text_message
from a2a.utils.task import new_task

# Extract the current task from the context
# This is the current task being processed
task: Task | None = context.current_task
# If no task is found, create a new task
# This is the first message from the user
if task is None:
    # Check that we have a message to create a task from
    if context.message is None:
        raise ValueError("Cannot create task: no message in context")
    # Create a new task for the first message from the user
    task = new_task(context.message)
# Initialize the task updater
task_updater = TaskUpdater(event_queue, task.id, task.context_id)
# Set the current task in the context
context.current_task = task
```

- Add the `task_id` and `context_id` from the task that's created while sending the message to the agent using `send_message`
    - Message from A2A has the attributes for context_id and task_id
- Use the methods of task_updater to update the task whenver you receive the response from the agent
- If the response from the agent is a `Task` then extract the artifacts and extract the task state and message from task.status
    - use task_updater to update both the artifacts and status

- Update LangGraph agent to support interrupts
    - When LangGraph's graph is in an interrupted state, then the task should go to the status of `input-required`
    - In that case, send a task status update event with state as `input-required`
