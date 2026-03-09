from packaging.specifiers import SpecifierSet
from packaging.version import Version

WILDCARD = "*"


def parse_version(version_str: str) -> Version:
    """Parse a version string into a Version object."""
    return Version(version_str)


def matches_specifier(version_str: str, specifier_str: str) -> bool:
    """Check if a version string matches a PEP 440 specifier."""
    spec = SpecifierSet(specifier_str)
    return Version(version_str) in spec


def find_latest(versions: list[str]) -> str | None:
    """Find the latest version from a list of version strings.

    Ignores entries that are the wildcard '*'.
    """
    concrete = []
    for v in versions:
        if v != WILDCARD:
            concrete.append(v)
    if not concrete:
        return None
    return str(max(concrete, key=Version))


def find_best_match(versions: list[str], specifier_str: str) -> str | None:
    """Find the latest version matching a PEP 440 specifier.

    If specifier_str is '*', returns the latest concrete version.
    Entries with version '*' are excluded from specifier matching
    but used as fallback if no concrete match is found.
    """
    if specifier_str == WILDCARD:
        latest = find_latest(versions)
        if latest is not None:
            return latest
        if WILDCARD in versions:
            return WILDCARD
        return None

    spec = SpecifierSet(specifier_str)
    has_wildcard = WILDCARD in versions
    concrete = [v for v in versions if v != WILDCARD]
    matching = [v for v in concrete if Version(v) in spec]

    if matching:
        return str(max(matching, key=Version))
    if has_wildcard:
        return WILDCARD
    return None
