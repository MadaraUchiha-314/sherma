"""Skill file resolver for local and remote skill cards."""

from pathlib import Path

from sherma.entities.skill_card import SkillCard
from sherma.http import get_http_client


class SkillResolver:
    """Resolves and loads files referenced by a SkillCard."""

    def __init__(self, skill_card: SkillCard) -> None:
        self._skill_card = skill_card

    def is_remote(self) -> bool:
        """Check if the skill card base_uri points to a remote location."""
        return self._skill_card.base_uri.startswith(
            "http://"
        ) or self._skill_card.base_uri.startswith("https://")

    def resolve_path(self, relative_path: str) -> str:
        """Resolve a relative file path against the base_uri."""
        if self.is_remote():
            base = self._skill_card.base_uri.rstrip("/")
            return f"{base}/{relative_path}"
        return str(Path(self._skill_card.base_uri) / relative_path)

    async def load_file(self, relative_path: str) -> str:
        """Load file content from a relative path."""
        resolved = self.resolve_path(relative_path)
        if self.is_remote():
            client = await get_http_client()
            response = await client.get(resolved)
            response.raise_for_status()
            return response.text
        return Path(resolved).read_text()

    def list_files_by_prefix(self, prefix: str) -> list[str]:
        """List files matching a given prefix."""
        return [f for f in self._skill_card.files if f.startswith(prefix)]
