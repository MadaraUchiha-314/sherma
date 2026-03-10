"""RegistryBundle: container for all per-tenant registry instances."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from sherma.entities.base import DEFAULT_TENANT_ID
from sherma.registry.agent import AgentRegistry
from sherma.registry.llm import LLMRegistry
from sherma.registry.prompt import PromptRegistry
from sherma.registry.skill import SkillRegistry
from sherma.registry.skill_card import SkillCardRegistry
from sherma.registry.tool import ToolRegistry


class RegistryBundle(BaseModel):
    """Container for all registry types and pre-constructed chat models."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tenant_id: str = DEFAULT_TENANT_ID
    tool_registry: ToolRegistry = Field(default_factory=ToolRegistry)
    llm_registry: LLMRegistry = Field(default_factory=LLMRegistry)
    prompt_registry: PromptRegistry = Field(default_factory=PromptRegistry)
    skill_registry: SkillRegistry = Field(default_factory=SkillRegistry)
    agent_registry: AgentRegistry = Field(default_factory=AgentRegistry)
    skill_card_registry: SkillCardRegistry = Field(default_factory=SkillCardRegistry)
    chat_models: dict[str, Any] = Field(default_factory=dict)
