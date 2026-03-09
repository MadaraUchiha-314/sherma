import pytest

from sherma.entities.llm import LLM
from sherma.registry.base import RegistryEntry
from sherma.registry.llm import LLMRegistry


@pytest.mark.asyncio
async def test_llm_registry_local():
    reg = LLMRegistry()
    llm = LLM(id="gpt4", version="1.0.0", model_name="gpt-4")
    await reg.add(RegistryEntry(id="gpt4", version="1.0.0", instance=llm))
    result = await reg.get("gpt4", "==1.0.0")
    assert result.model_name == "gpt-4"


@pytest.mark.asyncio
async def test_llm_registry_remote():
    reg = LLMRegistry()
    await reg.add(
        RegistryEntry(
            id="remote-llm",
            version="1.0.0",
            remote=True,
            url="http://api.example.com/v1",
        )
    )
    result = await reg.get("remote-llm", "==1.0.0")
    assert result.model_name == "http://api.example.com/v1"
