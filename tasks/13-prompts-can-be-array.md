- Currently prompts can be declared/evaluted as a string in the declarative agent
  - prompt: 'prompts["discover-skills"]["instructions"]'
- Prompts also should support arrays and each element of that array can specify whether it's a system message, human message etc
  - Each element should also allow the prompt content to be evaluated as a string using CEL
- the inclusion of state["messages"] in the prompt for an llm has to be explicit
     and be declared in the YAML instead of it being automatically injected into
     prompt
