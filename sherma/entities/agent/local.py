from sherma.entities.agent.base import Agent


class LocalAgent(Agent):
    """A local agent that users extend with custom logic.

    Subclass and implement ``send_message`` and ``cancel_task``.
    """

    ...
