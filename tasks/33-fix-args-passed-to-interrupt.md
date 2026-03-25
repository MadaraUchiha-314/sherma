# Task: Fix args passed to interrupt (#33)

GitHub issue: https://github.com/MadaraUchiha-314/sherma/issues/33

## Description

Interrupt value always prefers last AIMessage — the interrupt node tries `_find_last_ai_message(state)` first and only falls back to the CEL `args.value` when no AIMessage exists. The interrupt node should always take a CEL expression; the automatic parsing of the last AI message should be removed.

## Chat Iterations

_(none yet)_
