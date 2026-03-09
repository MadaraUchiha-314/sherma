from sherma.entities.agent.base import Agent


class LocalAgent(Agent):
    """A local agent that users extend with custom logic.

    Subclass and implement:
    - ``send_message(request, *, context, request_metadata, extensions)``
      → async iterator of ``UpdateEvent | Message | Task``
    - ``cancel_task(request, *, context, extensions)`` → ``Task``
    """

    ...
