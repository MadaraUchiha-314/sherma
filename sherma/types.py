from enum import StrEnum

Markdown = str


class Protocol(StrEnum):
    A2A = "a2a"
    MCP = "mcp"
    CUSTOM = "custom"


class EntityType(StrEnum):
    PROMPT = "prompt"
    LLM = "llm"
    TOOL = "tool"
    SKILL = "skill"
    AGENT = "agent"
