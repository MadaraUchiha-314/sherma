# Task 61: File-based prompt loading in YAML

GitHub Issue: https://github.com/MadaraUchiha-314/sherma/issues/61

## Problem

`PromptDef.instructions` accepts only inline strings. Long prompts must be
inlined as YAML block scalars, making `agent.yaml` unwieldy. Production
systems typically have many prompts exceeding 100 lines each; keeping them
as separate files follows version control, reuse, and readability best
practices.

## Goal

Add an `instructions_path` field to `PromptDef` that loads prompt
instructions from a file relative to the YAML's `base_path`, mirroring the
existing `skill_card_path` mechanism on `SkillDef`. Exactly one of
`instructions` or `instructions_path` must be provided.

## Chat Iterations

_None yet._
