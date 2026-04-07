# Plan 63: Preserve message objects on interrupt resume

## Goal

Stop `str()`-ifying `BaseMessage` resume values in the `interrupt` node so
structured human-in-the-loop payloads survive the resume.

## Steps

1. **Edit `sherma/langgraph/declarative/nodes.py`**
   - Import `BaseMessage` from `langchain_core.messages`.
   - Replace the single-line result assignment after the `after_interrupt`
     hook with a type-dispatch:
     - `list[BaseMessage]` → `{"messages": response}`
     - `BaseMessage` → `{"messages": [response]}`
     - else → `{"messages": [HumanMessage(content=str(response))]}`

2. **Update docs** (`docs/declarative-agents.md`) under the `interrupt`
   section: describe the three-way dispatch and the `additional_kwargs`
   use case.

3. **Update skill reference**
   (`skills/sherma/references/declarative-agents.md`) with the same change
   to keep docs and skill in sync per CLAUDE.md.

## Out of scope

- New hooks or schema changes.
- Changes to `before_interrupt` / `after_interrupt` hook contracts.

## Plan Revisions

- **Add a runnable example to existing examples.** Extended scope to add
  `examples/approval_agent/main_structured_resume.py`, which exercises
  the new behavior end-to-end without an `ApprovalTaggingHook` (the
  client passes a structured `HumanMessage` directly into
  `Command(resume=...)`). Also updated the YAML header comment, docs,
  and skill reference to mention the new variant.
