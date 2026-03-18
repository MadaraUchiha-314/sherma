# Plan: Add manifest version to declarative agent schema

## Issue
GitHub Issue #25: Add a manifest version to the declarative agent schema.

## Requirements
- Add a `manifest_version` field to `DeclarativeConfig` (mandatory, integer)
- Initial version set to 1
- Validate that `manifest_version` is present and valid during config loading
- Add `manifest_version: 1` to all example agent YAML files
- Add tests for the new field

## Changes

### 1. Schema (`sherma/langgraph/declarative/schema.py`)
- Add `manifest_version: int` field to `DeclarativeConfig` (required, no default)

### 2. Loader (`sherma/langgraph/declarative/loader.py`)
- No changes needed — Pydantic validation will enforce the required field

### 3. Example agent YAML files
Add `manifest_version: 1` to:
- `examples/declarative_weather_agent/agent.yaml`
- `examples/declarative_skill_agent/agent.yaml`
- `examples/declarative_hooks_agent/agent.yaml`
- `examples/multi_agent/supervisor_agent.yaml`
- `examples/multi_agent/weather_agent.yaml`

### 4. Tests
- `tests/langgraph/declarative/test_schema.py`: Add tests for manifest_version
- `tests/langgraph/declarative/test_loader.py`: Update YAML fixtures to include manifest_version, add test for missing manifest_version
