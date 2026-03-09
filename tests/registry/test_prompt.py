import pytest

from sherma.entities.prompt import Prompt
from sherma.registry.base import RegistryEntry
from sherma.registry.prompt import PromptRegistry


@pytest.mark.asyncio
async def test_prompt_registry_local():
    reg = PromptRegistry()
    p = Prompt(id="sys", version="1.0.0", instructions="Be helpful.")
    await reg.add(RegistryEntry(id="sys", version="1.0.0", instance=p))
    result = await reg.get("sys", "==1.0.0")
    assert result.instructions == "Be helpful."
