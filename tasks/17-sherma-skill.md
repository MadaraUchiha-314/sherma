- Create a skill that coding agents like claude, cursor etc can use to understand sherma and all the constructs that it provides
- this skill will be used by consumers of the sherma framework/library to create agents using the constructs that sherma provides
- the skill should be compatible with claude skills and https://agentskills.io/home
- when given an abmiguous request to build and agent, sherma skill should ask follow up questions so that it has all the context needed to create the agent
- balance between asking follow up questions and making decisions

## Chat Iterations

### Iteration 1: Initial implementation

Implemented the plan as written — created `skills/sherma/` with SKILL.md, skill-card.json, references (8 doc copies), and assets (4 example copies). Also created `.claude/skills/sherma/SKILL.md` as a separate file.

### Iteration 2: Avoid duplicate SKILL.md

User feedback: "do we need to write in .claude folder, aren't we already creating a skills/sherma folder at the root?"

Decided to keep both locations since `.claude/skills/sherma/SKILL.md` is required for Claude Code's `/sherma` slash command to work, while `skills/sherma/` is the agentskills.io format. But the content should not be duplicated.

### Iteration 3: Symlink the whole folder

User feedback: "the whole folder should be symlinked, not just the SKILL.md"

Changed approach: instead of symlinking just SKILL.md, symlinked the entire directory:
- `.claude/skills/sherma` → `../../skills/sherma` (directory symlink)

This means the single source of truth is `skills/sherma/` and Claude Code reads it through the symlink. The SKILL.md file has both agentskills.io frontmatter (`name`, `description`, `license`) and Claude Code frontmatter (`user-invocable`, `allowed-tools`, `argument-hint`) merged into one file.

Git stores symlinks as the path they point to, so this works when checked into GitHub.
