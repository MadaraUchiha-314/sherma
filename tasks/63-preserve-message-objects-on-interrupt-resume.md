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

## Chat Iterations

- **Add a runnable example.** Added
  `examples/approval_agent/main_structured_resume.py` — a variant of the
  approval-agent client that drives the graph directly and passes a
  `HumanMessage` with `additional_kwargs={"decision": ...}` straight into
  `Command(resume=...)`. The same `agent.yaml` is reused; no
  `ApprovalTaggingHook` is needed because the interrupt node now
  preserves the message verbatim. Updated the YAML header comment, the
  docs, and the skill reference to point at the new variant.

- **Add tests for the new dispatch.** The first pass shipped without
  coverage for the three-way dispatch. Added:
  - Unit tests in `tests/langgraph/declarative/test_agent.py`:
    `test_declarative_agent_interrupt_preserves_base_message`,
    `test_declarative_agent_interrupt_preserves_list_of_base_messages`,
    `test_declarative_agent_interrupt_non_message_falls_back_to_str`,
    `test_declarative_agent_structured_resume_routes_without_hook`.
  - Integration test at `tests/integration/test_approval_agent.py`,
    mirroring `examples/approval_agent/main_structured_resume.py` by
    loading the real `agent.yaml` with a `FakeChatModel`. Covers both
    the direct-approve path and the revise → approve loop.
