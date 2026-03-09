"""Integration tests for the registry system."""

import pytest

from sherma.entities.llm import LLM
from sherma.entities.prompt import Prompt
from sherma.entities.tool import Tool
from sherma.registry.base import RegistryEntry
from sherma.registry.llm import LLMRegistry
from sherma.registry.prompt import PromptRegistry
from sherma.registry.tool import ToolRegistry


@pytest.mark.integration
@pytest.mark.asyncio
async def test_register_and_get_prompt():
    reg = PromptRegistry()
    p = Prompt(id="system", version="1.0.0", instructions="Be helpful.")
    await reg.add(RegistryEntry(id="system", version="1.0.0", instance=p))

    result = await reg.get("system", "==1.*")
    assert result.instructions == "Be helpful."
    assert result.id == "system"
    assert result.version == "1.0.0"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_register_multiple_versions():
    reg = LLMRegistry()
    v1 = LLM(id="gpt", version="1.0.0", model_name="gpt-3.5")
    v2 = LLM(id="gpt", version="2.0.0", model_name="gpt-4")
    await reg.add(RegistryEntry(id="gpt", version="1.0.0", instance=v1))
    await reg.add(RegistryEntry(id="gpt", version="2.0.0", instance=v2))

    latest = await reg.get("gpt", "*")
    assert latest.model_name == "gpt-4"

    v1_result = await reg.get("gpt", "==1.*")
    assert v1_result.model_name == "gpt-3.5"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_register_tool_with_factory():
    reg = ToolRegistry()

    def make_tool() -> Tool:
        def calculator(expr: str) -> str:
            return str(eval(expr))

        return Tool(id="calc", version="1.0.0", function=calculator)

    await reg.add(RegistryEntry(id="calc", version="1.0.0", factory=make_tool))
    tool = await reg.get("calc", "*")
    assert tool.function("1+1") == "2"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_wildcard_version_registration():
    reg = PromptRegistry()
    p = Prompt(id="default", version="*", instructions="Default prompt")
    await reg.add(RegistryEntry(id="default", version="*", instance=p))

    result = await reg.get("default", "*")
    assert result.instructions == "Default prompt"

    result2 = await reg.get("default", "==3.*")
    assert result2.instructions == "Default prompt"
