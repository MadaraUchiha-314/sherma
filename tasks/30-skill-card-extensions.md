# Task 30: Skill card should explicitly state its extensions

GitHub Issue: https://github.com/MadaraUchiha-314/sherma/issues/30

## Problem

The `examples/skills/weather/skill-card.json` uses `local_tools` but doesn't
explicitly declare which skill extensions it relies on in the `extensions` field.

## Goal

Add an `extensions` field to the weather skill card that explicitly declares the
extensions it uses (i.e., `local_tools`). Update documentation and the skill
examples in docs to match.
