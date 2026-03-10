"""LangGraph skill tools for progressive skill disclosure."""

import re

from langchain_core.tools import BaseTool, tool

from sherma.hooks.manager import HookManager
from sherma.langgraph.tools import from_langgraph_tool
from sherma.logging import get_logger
from sherma.registry.base import RegistryEntry
from sherma.registry.skill import SkillRegistry, _parse_skill_md
from sherma.registry.skill_card import SkillCardRegistry
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


def create_skill_tools(
    skill_card_registry: SkillCardRegistry,
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
        for versions in skill_card_registry._entries.values():
            for entry in versions.values():
                resolved = await skill_card_registry._resolve(entry)
                results.append(
                    {
                        "id": resolved.id,
                        "version": resolved.version,
                        "name": resolved.name,
                        "description": resolved.description,
                    }
                )
        logger.info("list_skills returning %d skills", len(results))
        return results

    @tool
    async def load_skill_md(skill_id: str, version: str = "*") -> str:
        """Load and activate a skill by reading its SKILL.md file.

        This also registers any MCP or local tools defined in the skill.
        """
        version = _normalize_version(version)
        logger.info("load_skill_md called: skill_id=%s, version=%s", skill_id, version)

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

        skill_card = await skill_card_registry.get(skill_id, version)
        resolver = SkillResolver(skill_card)

        # Load SKILL.md content
        content = await resolver.load_file("SKILL.md")

        # Parse and store in skill registry
        skill = _parse_skill_md(content, skill_card.id, skill_card.version)
        await skill_registry.add(
            RegistryEntry(
                id=skill_card.id,
                version=skill_card.version,
                instance=skill,
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

        return content

    @tool
    async def list_skill_resources(skill_id: str, version: str = "*") -> list[str]:
        """List reference files available in a skill."""
        version = _normalize_version(version)
        skill_card = await skill_card_registry.get(skill_id, version)
        resolver = SkillResolver(skill_card)
        return resolver.list_files_by_prefix("references/")

    @tool
    async def load_skill_resource(
        skill_id: str, resource_path: str, version: str = "*"
    ) -> str:
        """Load a reference file. Use list_skill_resources first."""
        version = _normalize_version(version)
        skill_card = await skill_card_registry.get(skill_id, version)
        available = [f for f in skill_card.files if f.startswith("references/")]
        if resource_path not in available:
            return f"Error: '{resource_path}' not found. Available: {available}"
        resolver = SkillResolver(skill_card)
        return await resolver.load_file(resource_path)

    @tool
    async def list_skill_assets(skill_id: str, version: str = "*") -> list[str]:
        """List asset files available in a skill."""
        version = _normalize_version(version)
        skill_card = await skill_card_registry.get(skill_id, version)
        resolver = SkillResolver(skill_card)
        return resolver.list_files_by_prefix("assets/")

    @tool
    async def load_skill_asset(
        skill_id: str, asset_path: str, version: str = "*"
    ) -> str:
        """Load an asset file. Use list_skill_assets first."""
        version = _normalize_version(version)
        skill_card = await skill_card_registry.get(skill_id, version)
        available = [f for f in skill_card.files if f.startswith("assets/")]
        if asset_path not in available:
            return f"Error: '{asset_path}' not found. Available: {available}"
        resolver = SkillResolver(skill_card)
        return await resolver.load_file(asset_path)

    return [
        list_skills,
        load_skill_md,
        list_skill_resources,
        load_skill_resource,
        list_skill_assets,
        load_skill_asset,
    ]
