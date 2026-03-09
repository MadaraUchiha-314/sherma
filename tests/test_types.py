from sherma.types import EntityType, Markdown, Protocol


def test_protocol_values():
    assert Protocol.A2A == "a2a"
    assert Protocol.MCP == "mcp"
    assert Protocol.CUSTOM == "custom"


def test_entity_type_values():
    assert EntityType.PROMPT == "prompt"
    assert EntityType.LLM == "llm"
    assert EntityType.TOOL == "tool"
    assert EntityType.SKILL == "skill"
    assert EntityType.AGENT == "agent"


def test_markdown_alias():
    value: Markdown = "# Hello"
    assert isinstance(value, str)
