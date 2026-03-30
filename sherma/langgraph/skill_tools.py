"""LangGraph skill tools for progressive skill disclosure."""

import re

from langchain_core.tools import BaseTool, tool

from sherma.hooks.manager import HookManager
from sherma.langgraph.tools import from_langgraph_tool
from sherma.logging import get_logger
from sherma.registry.base import RegistryEntry
from sherma.registry.skill import SkillRegistry, _parse_skill_md
from sherma.registry.tool import ToolRegistry
from sherma.skills.local_tools import load_local_tools_from_skill
from sherma.skills.mcp import load_mcp_tools_from_skill
from sherma.skills.resolver import SkillResolver

logger = get_logger(__name__)

_BARE_VERSION_RE = re.compile(r"^\d+(\.\d+)*$")


def _normalize_version(version: str) -> str:
    """Normalize a version string to a valid PEP 440 specifier.

    Bare versions like '1.0.0' become '==1.0.0'. Already-valid
    specifiers like '>=1.0.0' or '*' pass through unchanged.
    """
    if _BARE_VERSION_RE.match(version):
        return f"=={version}"
    return version


async def load_and_register_skill(
    skill_id: str,
    version: str,
    skill_registry: SkillRegistry,
    tool_registry: ToolRegistry,
    hook_manager: HookManager | None = None,
) -> tuple[str, list[str]]:
    """Load a skill's SKILL.md, register its tools, return (content, tool_ids).

    This is the shared core logic used by both the progressive-disclosure
    ``load_skill_md`` tool and the declarative ``load_skills`` node.
    """
    version = _normalize_version(version)
    logger.info("load_and_register_skill: skill_id=%s, version=%s", skill_id, version)

    # before_skill_load
    if hook_manager:
        from sherma.hooks.types import BeforeSkillLoadContext

        before_ctx = await hook_manager.run_hook(
            "before_skill_load",
            BeforeSkillLoadContext(
                node_context=None,
                skill_id=skill_id,
                version=version,
            ),
        )
        skill_id = before_ctx.skill_id
        version = before_ctx.version

    skill = await skill_registry.get(skill_id, version)
    skill_card = skill.skill_card
    if skill_card is None:
        return f"Error: skill '{skill_id}' has no skill card", []

    resolver = SkillResolver(skill_card)

    # Load SKILL.md content
    content = await resolver.load_file("SKILL.md")

    # Parse and update the skill in the registry with loaded content
    parsed_skill = _parse_skill_md(content, skill_card.id, skill_card.version)
    parsed_skill.skill_card = skill_card
    await skill_registry.add(
        RegistryEntry(
            id=skill_card.id,
            version=skill_card.version,
            instance=parsed_skill,
        )
    )

    # Load and register MCP tools
    mcp_tools = await load_mcp_tools_from_skill(skill_card)
    tools_loaded: list[str] = []
    for mcp_tool in mcp_tools:
        sherma_tool = from_langgraph_tool(mcp_tool)
        await tool_registry.add(
            RegistryEntry(
                id=sherma_tool.id,
                version=sherma_tool.version,
                instance=sherma_tool,
            )
        )
        tools_loaded.append(sherma_tool.id)

    # Load and register local tools
    local_tools = load_local_tools_from_skill(skill_card)
    for local_tool in local_tools:
        sherma_tool = from_langgraph_tool(local_tool)
        await tool_registry.add(
            RegistryEntry(
                id=sherma_tool.id,
                version=sherma_tool.version,
                instance=sherma_tool,
            )
        )
        tools_loaded.append(sherma_tool.id)

    # after_skill_load
    if hook_manager:
        from sherma.hooks.types import AfterSkillLoadContext

        after_ctx = await hook_manager.run_hook(
            "after_skill_load",
            AfterSkillLoadContext(
                node_context=None,
                skill_id=skill_id,
                version=version,
                content=content,
                tools_loaded=tools_loaded,
            ),
        )
        content = after_ctx.content

    return content, tools_loaded


async def _unload_skill(
    skill_id: str,
    version: str,
    skill_registry: SkillRegistry,
    hook_manager: HookManager | None = None,
) -> list[str]:
    """Mark a skill as unloaded and return its tool IDs.

    This is a run-local operation: it does **not** remove tools from the
    ``ToolRegistry`` (which is shared across runs).  Instead callers
    update ``__sherma__`` internal state so that downstream
    ``use_tools_from_loaded_skills`` nodes stop binding the skill's tools.
    """
    version = _normalize_version(version)
    logger.info("unload_skill: skill_id=%s, version=%s", skill_id, version)

    # before_skill_unload
    if hook_manager:
        from sherma.hooks.types import BeforeSkillUnloadContext

        before_ctx = await hook_manager.run_hook(
            "before_skill_unload",
            BeforeSkillUnloadContext(
                node_context=None,
                skill_id=skill_id,
                version=version,
            ),
        )
        skill_id = before_ctx.skill_id
        version = before_ctx.version

    skill = await skill_registry.get(skill_id, version)
    skill_card = skill.skill_card
    if skill_card is None:
        return []

    tools_unloaded: list[str] = []
    for mcp_id in skill_card.mcps:
        tools_unloaded.append(mcp_id)
    for tool_id in skill_card.local_tools:
        tools_unloaded.append(tool_id)

    # after_skill_unload
    if hook_manager:
        from sherma.hooks.types import AfterSkillUnloadContext

        await hook_manager.run_hook(
            "after_skill_unload",
            AfterSkillUnloadContext(
                node_context=None,
                skill_id=skill_id,
                version=version,
                tools_unloaded=tools_unloaded,
            ),
        )

    return tools_unloaded


def create_skill_tools(
    skill_registry: SkillRegistry,
    tool_registry: ToolRegistry,
    hook_manager: HookManager | None = None,
) -> list[BaseTool]:
    """Create LangGraph tools for progressive skill disclosure.

    Returns tools for listing skills, loading SKILL.md, listing/loading
    resources and assets.
    """

    @tool
    async def list_skills() -> list[dict[str, str]]:
        """List all available skills with their metadata."""
        logger.info("list_skills called")
        results: list[dict[str, str]] = []
        for versions in skill_registry._entries.values():
            for entry in versions.values():
                resolved = await skill_registry._resolve(entry)
                card = resolved.skill_card
                if card:
                    results.append(
                        {
                            "id": resolved.id,
                            "version": resolved.version,
                            "name": card.name,
                            "description": card.description,
                        }
                    )
                else:
                    results.append(
                        {
                            "id": resolved.id,
                            "version": resolved.version,
                            "name": resolved.front_matter.name,
                            "description": resolved.front_matter.description,
                        }
                    )
        logger.info("list_skills returning %d skills", len(results))
        return results

    @tool
    async def load_skill_md(skill_id: str, version: str = "*") -> str:
        """Load and activate a skill by reading its SKILL.md file.

        This also registers any MCP or local tools defined in the skill.
        """
        content, _tool_ids = await load_and_register_skill(
            skill_id, version, skill_registry, tool_registry, hook_manager
        )
        return content

    @tool
    async def list_skill_resources(skill_id: str, version: str = "*") -> list[str]:
        """List reference files available in a skill."""
        version = _normalize_version(version)
        skill = await skill_registry.get(skill_id, version)
        if skill.skill_card is None:
            return []
        resolver = SkillResolver(skill.skill_card)
        return resolver.list_files_by_prefix("references/")

    @tool
    async def load_skill_resource(
        skill_id: str, resource_path: str, version: str = "*"
    ) -> str:
        """Load a reference file. Use list_skill_resources first."""
        version = _normalize_version(version)
        skill = await skill_registry.get(skill_id, version)
        if skill.skill_card is None:
            return f"Error: skill '{skill_id}' has no skill card"
        available = [f for f in skill.skill_card.files if f.startswith("references/")]
        if resource_path not in available:
            return f"Error: '{resource_path}' not found. Available: {available}"
        resolver = SkillResolver(skill.skill_card)
        return await resolver.load_file(resource_path)

    @tool
    async def list_skill_assets(skill_id: str, version: str = "*") -> list[str]:
        """List asset files available in a skill."""
        version = _normalize_version(version)
        skill = await skill_registry.get(skill_id, version)
        if skill.skill_card is None:
            return []
        resolver = SkillResolver(skill.skill_card)
        return resolver.list_files_by_prefix("assets/")

    @tool
    async def load_skill_asset(
        skill_id: str, asset_path: str, version: str = "*"
    ) -> str:
        """Load an asset file. Use list_skill_assets first."""
        version = _normalize_version(version)
        skill = await skill_registry.get(skill_id, version)
        if skill.skill_card is None:
            return f"Error: skill '{skill_id}' has no skill card"
        available = [f for f in skill.skill_card.files if f.startswith("assets/")]
        if asset_path not in available:
            return f"Error: '{asset_path}' not found. Available: {available}"
        resolver = SkillResolver(skill.skill_card)
        return await resolver.load_file(asset_path)

    @tool
    async def unload_skill(skill_id: str, version: str = "*") -> str:
        """Unload a previously loaded skill so its tools are no longer bound.

        Call this when a skill is no longer needed to free context window
        space.  The skill's tools will no longer be available for use in
        subsequent LLM calls.  The skill can be re-loaded later with
        ``load_skill_md`` if needed again.
        """
        tools_removed = await _unload_skill(
            skill_id, version, skill_registry, hook_manager
        )
        if tools_removed:
            return (
                f"Skill '{skill_id}' unloaded. "
                f"Unbound tools: {', '.join(tools_removed)}"
            )
        return f"Skill '{skill_id}' unloaded (no tools were registered)."

    return [
        list_skills,
        load_skill_md,
        unload_skill,
        list_skill_resources,
        load_skill_resource,
        list_skill_assets,
        load_skill_asset,
    ]
