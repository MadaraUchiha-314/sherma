# Task 30: Skill card should explicitly state its extensions

GitHub Issue: https://github.com/MadaraUchiha-314/sherma/issues/30

## Problem

The `examples/skills/weather/skill-card.json` uses `local_tools` but doesn't
explicitly declare which skill extensions it relies on in the `extensions` field.

## Goal

Add an `extensions` field to the weather skill card that explicitly declares the
extensions it uses (i.e., `local_tools`). Update documentation and the skill
examples in docs to match.

## Chat Iterations

1. Initial implementation used `extensions` as `dict[str, Any]` with `{"local_tools": true}`.
2. User requested changing to an array with `uri`-identified entries, modeled after
   [A2A AgentExtension](https://a2a-protocol.org/latest/specification/#444-agentextension).
   Changed `extensions` from `dict[str, Any]` to `list[SkillExtension]` where
   `SkillExtension` has `uri` (required), `description`, `required`, and `params`.
