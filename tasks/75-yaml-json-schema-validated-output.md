# Task 75: JSON-Schema-validated agent output in declarative YAML

GitHub Issue: https://github.com/MadaraUchiha-314/sherma/issues/74 (follow-up)

## Problem

`AgentDef.input_schema` and `AgentDef.output_schema` already accept
JSON Schema dicts in YAML, but those values are **not wired through**
to the runtime agent. As a result, declarative agents can't take
advantage of the existing A2A-executor input/output validation, and
the comparison comment on issue #74 correctly noted that sherma has
"no JSON-Schema-validated output channel" — Claude's
`output_schema:` (e.g. earnings-reviewer's `transcript-reader`
sub-agent) has no equivalent.

## Goal

Connect the YAML's `input_schema` / `output_schema` (JSON Schema dicts)
to the running agent, so:

* The agent's outgoing `DataPart`s tagged `agent_output: true` are
  validated against the declared output schema.
* The agent's incoming `DataPart`s tagged `agent_input: true` are
  validated against the declared input schema.
* The agent card auto-publishes the schemas as A2A extensions
  (`urn:sherma:schema:input` / `urn:sherma:schema:output`), exactly
  like the existing Pydantic-typed schema flow.

Validation uses the `jsonschema` library directly (already a
transitive dependency).

## Chat Iterations

_None yet._
