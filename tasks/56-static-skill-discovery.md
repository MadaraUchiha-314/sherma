# Task 56: Skill Discovery Adds an Extra LLM Loop Because Front-Matter Is Not Discovered Statically

GitHub Issue: https://github.com/MadaraUchiha-314/sherma/issues/56

## Problem

In the `declarative_skill_agent` example, the `discover_skills` node currently:

1. Calls `list_skills` tool (LLM invokes it) to get skill metadata (id, version, name, description)
2. Calls `load_skill_md` to load the relevant skill
3. Responds with a summary

Step 1 wastes an LLM loop — the agent has to call `list_skills` just to see what's available. This metadata is already known at build time from the skill cards registered in the YAML config.

## Solution

1. Add a new node (e.g., `list_skills` of type `list_skills`) that runs **before** `discover_skills` and statically calls the `list_skills` tool, injecting results into state messages.
2. Remove `list_skills` from the `discover_skills` node's tool list.
3. Update the `discover_skills` prompt to reference the already-available skill catalog instead of instructing the LLM to call `list_skills`.
4. Use sherma's template variable injection (CEL + `skills` extra var) to make skill metadata available in prompts at build time.

## Chat Iterations

_(none yet)_
