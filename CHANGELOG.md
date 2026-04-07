## v1.1.0 (2026-04-07)

### Feat

- file-based prompt loading via instructions_path (#69)

## v1.0.0 (2026-04-01)

### BREAKING CHANGE

- call_llm no longer implicitly appends the LLM response
to messages. The state_updates field is now required — users must
explicitly declare which state fields to update.

## v0.21.0 (2026-03-30)

### Feat

- add skill unloading and per-skill tool tracking (#55)

## v0.20.0 (2026-03-29)

### Feat

- add list filtering and searching in CEL (#51)

## v0.19.0 (2026-03-29)

### Feat

- add message metadata access in CEL (additional_kwargs, type) (#52)

## v0.18.0 (2026-03-27)

### Feat

- add load_skills node type for programmatic skill loading (#49)

## v0.17.0 (2026-03-27)

### Feat

- add custom node type with node_execute hook (#48)

## v0.16.0 (2026-03-27)

### Feat

- add template() CEL function for prompt string templating (#47)

## v0.15.0 (2026-03-26)

### Feat

- add declarative on_error with retry and fallback routing (#37)

## v0.14.0 (2026-03-25)

### Feat

- add custom CEL functions for JSON parsing, safe access, and string manipulation (#35)

## v0.13.3 (2026-03-25)

### Fix

- always evaluate CEL expression for interrupt node value (#34)

## v0.13.2 (2026-03-23)

### Fix

- add explicit extensions field to skill cards (#31)

## v0.13.1 (2026-03-22)

### Fix

- trigger release

## v0.13.0 (2026-03-18)

### Feat

- add manifest_version field to declarative agent schema (#26)

## v0.12.1 (2026-03-18)

### Fix

- move streamlit to different folder (#24)

## v0.12.0 (2026-03-18)

### Feat

- add streamlit agent designer (#22)

### Fix

- getting streamlit to work (#23)

## v0.11.3 (2026-03-18)

### Fix

- adding specific sub-agents selection (#21)

## v0.11.2 (2026-03-17)

### Fix

- add default llm (#20)

## v0.11.1 (2026-03-16)

### Fix

- add fine grained control for prompts (#18)

## v0.11.0 (2026-03-13)

### Feat

- introducing hooks over json rpc (#17)

## v0.10.2 (2026-03-12)

### Fix

- fixes to make it work in other ecosystems (#15)

## v0.10.1 (2026-03-11)

### Fix

- misc improvs (#14)

## v0.10.0 (2026-03-10)

### Feat

- agent as tool (#13)

## v0.9.0 (2026-03-10)

### Feat

- adding tenant support (#12)

## v0.8.1 (2026-03-10)

### Fix

- combine skill and skill card registeries (#10)

## v0.8.0 (2026-03-10)

### Feat

- hooks are here (#8)

## v0.7.0 (2026-03-10)

### Feat

- interrupts are here (#7)

## v0.6.0 (2026-03-10)

### Feat

- skill declarative agent is here (#6)

## v0.5.0 (2026-03-09)

### Feat

- declarative agents using yaml (#5)

## v0.4.0 (2026-03-09)

### Feat

- add agent input and output schema (#4)

## v0.3.1 (2026-03-09)

### Fix

- improve interfaces (#3)

## v0.3.0 (2026-03-09)

### Feat

- sherma is here (#2)

## v0.2.0 (2026-03-05)

### Feat

- init repo (#1)
