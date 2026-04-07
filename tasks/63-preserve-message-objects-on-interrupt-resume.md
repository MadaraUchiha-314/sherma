# Task 63: Preserve message objects on interrupt resume

## Problem

When an `interrupt` node resumes, `nodes.py:964` wraps the response as
`HumanMessage(content=str(response))`, discarding structured message objects
and their `additional_kwargs` metadata. This blocks human-in-the-loop flows
that route on structured resume payloads (approval decisions, action tags,
sender info) without needing an `after_interrupt` hook workaround.

## Requested behavior

On interrupt resume:

- If `response` is a `BaseMessage`, pass it through as `{"messages": [response]}`.
- If `response` is a `list[BaseMessage]`, pass it through as `{"messages": response}`.
- Otherwise, fall back to `HumanMessage(content=str(response))`.

## Acceptance

- `sherma/langgraph/declarative/nodes.py` preserves `BaseMessage` resume values.
- `docs/declarative-agents.md` and `skills/sherma/references/declarative-agents.md`
  describe the new resume-value handling.

## References

- Issue: MadaraUchiha-314/sherma#63
- `sherma/langgraph/declarative/nodes.py:964`
