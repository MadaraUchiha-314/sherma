"""Local tool loading from skill cards."""

from langchain_core.tools import BaseTool

from sherma.entities.skill_card import SkillCard
from sherma.langgraph.declarative.loader import import_tool
from sherma.logging import get_logger

logger = get_logger(__name__)


def load_local_tools_from_skill(skill_card: SkillCard) -> list[BaseTool]:
    """Load local tools defined in a skill card.

    Uses the same import mechanism as declarative agent tool imports.
    """
    tools: list[BaseTool] = []
    for name, tool_def in skill_card.local_tools.items():
        lg_tool = import_tool(tool_def.import_path)
        tools.append(lg_tool)
        logger.debug("Loaded local tool '%s' from skill '%s'", name, skill_card.id)
    return tools
