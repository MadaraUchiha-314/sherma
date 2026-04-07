# Task 62: Conditional entry point routing

GitHub Issue: https://github.com/MadaraUchiha-314/sherma/issues/62

## Problem

The declarative graph currently forces an unconditional edge from `START` to
a single `entry_point` node (`agent.py:254`). Agents often need to branch at
entry based on message type, sender, or metadata — for example, to
distinguish a fresh user request from a passthrough message in a
human-in-the-loop flow. Today users must insert a no-op `set_state` node
purely to attach conditional edges to it.

## Goal

Allow conditional (or plain) routing directly from `__start__` by permitting
edges with `source: __start__` in the graph `edges` list. When any edge
declares `__start__` as its source, `entry_point` becomes optional and
`START` is wired through those edges instead of an unconditional jump.

## Chat Iterations

_None yet._
