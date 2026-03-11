"""YAML loading and registry population for declarative agents."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import yaml
from langchain_core.tools import BaseTool

from sherma.entities.llm import LLM
from sherma.entities.prompt import Prompt
from sherma.entities.skill import Skill, SkillFrontMatter
from sherma.entities.skill_card import SkillCard
from sherma.exceptions import DeclarativeConfigError
from sherma.hooks.executor import HookExecutor
from sherma.hooks.manager import HookManager
from sherma.langgraph.declarative.schema import (
    CallLLMArgs,
    DeclarativeConfig,
)
from sherma.langgraph.tools import agent_to_langgraph_tool, from_langgraph_tool
from sherma.registry.base import RegistryEntry
from sherma.registry.bundle import RegistryBundle


def load_declarative_config(
    yaml_path: str | Path | None = None,
    yaml_content: str | None = None,
) -> DeclarativeConfig:
    """Load and validate a declarative config from YAML.

    Provide either yaml_path or yaml_content, not both.
    """
    if yaml_path is not None and yaml_content is not None:
        raise DeclarativeConfigError(
            "Provide either yaml_path or yaml_content, not both"
        )
    if yaml_path is None and yaml_content is None:
        raise DeclarativeConfigError("Provide either yaml_path or yaml_content")

    if yaml_path is not None:
        path = Path(yaml_path)
        if not path.exists():
            raise DeclarativeConfigError(f"YAML file not found: {path}")
        raw = path.read_text()
    else:
        raw = yaml_content

    try:
        data = yaml.safe_load(raw)  # type: ignore[arg-type]
    except yaml.YAMLError as exc:
        raise DeclarativeConfigError(f"Invalid YAML: {exc}") from exc

    if not isinstance(data, dict):
        raise DeclarativeConfigError("YAML root must be a mapping")

    return _parse_config(data)


def _parse_node_args(node_data: dict[str, Any]) -> dict[str, Any]:
    """Parse node args based on node type, converting dicts to proper models."""
    node_type = node_data.get("type")
    args = node_data.get("args", {})
    return {"name": node_data["name"], "type": node_type, "args": args}


def _parse_config(data: dict[str, Any]) -> DeclarativeConfig:
    """Parse raw YAML data into a DeclarativeConfig."""
    agents_data = data.get("agents", {})
    parsed_agents: dict[str, Any] = {}
    for agent_name, agent_data in agents_data.items():
        graph_data = agent_data.get("graph", {})
        nodes_data = graph_data.get("nodes", [])
        parsed_nodes = [_parse_node_args(n) for n in nodes_data]
        graph_data = {**graph_data, "nodes": parsed_nodes}
        parsed_agents[agent_name] = {**agent_data, "graph": graph_data}

    config_data = {**data, "agents": parsed_agents}
    try:
        return DeclarativeConfig.model_validate(config_data)
    except Exception as exc:
        raise DeclarativeConfigError(f"Config validation failed: {exc}") from exc


def import_tool(import_path: str) -> BaseTool:
    """Import a LangGraph tool from a dotted Python path.

    Example: "examples.tools.get_weather" imports the get_weather
    object from examples/tools.py.
    """
    module_path, _, attr_name = import_path.rpartition(".")
    if not module_path:
        raise DeclarativeConfigError(
            f"Invalid import_path '{import_path}': "
            f"must be a dotted path like 'module.tool_name'"
        )
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise DeclarativeConfigError(
            f"Cannot import module '{module_path}': {exc}"
        ) from exc

    if not hasattr(module, attr_name):
        raise DeclarativeConfigError(
            f"Module '{module_path}' has no attribute '{attr_name}'"
        )

    obj = getattr(module, attr_name)
    if isinstance(obj, BaseTool):
        return obj
    raise DeclarativeConfigError(
        f"'{import_path}' is not a LangGraph BaseTool, got {type(obj).__name__}"
    )


def _import_agent(import_path: str) -> Any:
    """Import an agent from a dotted Python path.

    Example: "my_agents.weather_agent" imports the weather_agent
    object from my_agents.py.
    """
    from sherma.entities.agent.base import Agent

    module_path, _, attr_name = import_path.rpartition(".")
    if not module_path:
        raise DeclarativeConfigError(
            f"Invalid import_path '{import_path}': "
            f"must be a dotted path like 'module.agent_name'"
        )
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise DeclarativeConfigError(
            f"Cannot import module '{module_path}': {exc}"
        ) from exc

    if not hasattr(module, attr_name):
        raise DeclarativeConfigError(
            f"Module '{module_path}' has no attribute '{attr_name}'"
        )

    obj = getattr(module, attr_name)
    if isinstance(obj, Agent):
        return obj
    raise DeclarativeConfigError(
        f"'{import_path}' is not an Agent, got {type(obj).__name__}"
    )


def _extract_bearer_token(http_async_client: Any) -> str | None:
    """Extract a Bearer token from an httpx.AsyncClient's default headers."""
    import httpx

    if not isinstance(http_async_client, httpx.AsyncClient):
        return None
    auth_header = http_async_client.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:]
    return None


def _build_chat_model_kwargs(
    provider: str,
    model_name: str,
    http_async_client: Any | None = None,
) -> dict[str, Any]:
    """Build constructor kwargs for a chat model without instantiating it."""
    api_key = _extract_bearer_token(http_async_client) if http_async_client else None

    if provider == "openai":
        kwargs: dict[str, Any] = {"model": model_name, "max_retries": 5}
        if http_async_client is not None:
            kwargs["http_async_client"] = http_async_client
        if api_key is not None:
            kwargs["api_key"] = api_key
        return kwargs

    if provider == "anthropic":
        kwargs_a: dict[str, Any] = {"model": model_name}
        if http_async_client is not None:
            kwargs_a["http_async_client"] = http_async_client
        if api_key is not None:
            kwargs_a["api_key"] = api_key
        return kwargs_a

    raise DeclarativeConfigError(
        f"Unsupported LLM provider '{provider}'. Supported: openai, anthropic"
    )


def _construct_chat_model(provider: str, kwargs: dict[str, Any]) -> Any:
    """Construct a chat model instance from provider and kwargs."""
    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise DeclarativeConfigError(
                "langchain-openai is required for provider 'openai'. "
                "Install with: uv add langchain-openai"
            ) from exc
        return ChatOpenAI(**kwargs)

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise DeclarativeConfigError(
                "langchain-anthropic is required for provider 'anthropic'. "
                "Install with: uv add langchain-anthropic"
            ) from exc
        return ChatAnthropic(**kwargs)  # type: ignore[call-arg]

    raise DeclarativeConfigError(
        f"Unsupported LLM provider '{provider}'. Supported: openai, anthropic"
    )


def create_chat_model(
    provider: str,
    model_name: str,
    http_async_client: Any | None = None,
) -> Any:
    """Create a LangChain chat model from provider name and model name.

    If *http_async_client* (an ``httpx.AsyncClient``) is provided it is
    forwarded to the underlying chat model.  When the client carries a
    ``Bearer`` token in its default ``Authorization`` header, that token
    is also passed as the ``api_key`` so that provider SDKs that require
    an explicit key (e.g. OpenAI) are satisfied.
    """
    kwargs = _build_chat_model_kwargs(provider, model_name, http_async_client)
    return _construct_chat_model(provider, kwargs)


async def populate_registries(
    config: DeclarativeConfig,
    registries: RegistryBundle,
    http_async_client: Any | None = None,
    hook_manager: HookManager | None = None,
) -> None:
    """Register entities declared in the config into registries."""
    from sherma.hooks.types import ChatModelCreateContext

    tenant_id = registries.tenant_id

    for llm_def in config.llms:
        await registries.llm_registry.add(
            RegistryEntry(
                id=llm_def.id,
                version=llm_def.version,
                tenant_id=tenant_id,
                instance=LLM(
                    id=llm_def.id,
                    version=llm_def.version,
                    tenant_id=tenant_id,
                    model_name=llm_def.model_name,
                ),
            )
        )
        # Auto-create chat model if not already provided
        if llm_def.id not in registries.chat_models:
            kwargs = _build_chat_model_kwargs(
                llm_def.provider, llm_def.model_name, http_async_client
            )

            if hook_manager is not None and hook_manager._executors:
                ctx = ChatModelCreateContext(
                    llm_id=llm_def.id,
                    provider=llm_def.provider,
                    model_name=llm_def.model_name,
                    kwargs=kwargs,
                )
                ctx = await hook_manager.run_hook("on_chat_model_create", ctx)
                if ctx.chat_model is not None:
                    registries.chat_models[llm_def.id] = ctx.chat_model
                else:
                    registries.chat_models[llm_def.id] = _construct_chat_model(
                        ctx.provider, ctx.kwargs
                    )
            else:
                registries.chat_models[llm_def.id] = _construct_chat_model(
                    llm_def.provider, kwargs
                )

    for prompt_def in config.prompts:
        await registries.prompt_registry.add(
            RegistryEntry(
                id=prompt_def.id,
                version=prompt_def.version,
                tenant_id=tenant_id,
                instance=Prompt(
                    id=prompt_def.id,
                    version=prompt_def.version,
                    tenant_id=tenant_id,
                    instructions=prompt_def.instructions,
                ),
            )
        )

    # Register skill cards as attributes on Skill entities
    for skill_def in config.skills:
        if skill_def.skill_card_path:
            path = Path(skill_def.skill_card_path)
            if not path.exists():
                raise DeclarativeConfigError(f"Skill card file not found: {path}")
            data = json.loads(path.read_text())
            # Resolve relative base_uri against the skill card file location
            base_uri = data.get("base_uri", "")
            if base_uri and not base_uri.startswith(("http://", "https://")):
                resolved_base = Path(base_uri)
                if not resolved_base.is_absolute():
                    resolved_base = (path.resolve().parent / resolved_base).resolve()
                data = {**data, "base_uri": str(resolved_base)}
            skill_card = SkillCard(
                id=skill_def.id,
                version=skill_def.version,
                tenant_id=tenant_id,
                **{
                    k: v
                    for k, v in data.items()
                    if k not in ("id", "version", "tenant_id")
                },
            )
            skill = Skill(
                id=skill_def.id,
                version=skill_def.version,
                tenant_id=tenant_id,
                front_matter=SkillFrontMatter(
                    name=skill_card.name,
                    description=skill_card.description,
                ),
                skill_card=skill_card,
            )
            await registries.skill_registry.add(
                RegistryEntry(
                    id=skill_def.id,
                    version=skill_def.version,
                    tenant_id=tenant_id,
                    instance=skill,
                )
            )
        elif skill_def.url:
            # For remote skills, create a Skill with a remote URL
            # The skill card will be fetched lazily via the skill registry
            from sherma.registry.skill_card import SkillCardRegistry

            temp_registry = SkillCardRegistry()
            await temp_registry.add(
                RegistryEntry(
                    id=skill_def.id,
                    version=skill_def.version,
                    tenant_id=tenant_id,
                    remote=True,
                    url=skill_def.url,
                )
            )
            fetched_card = await temp_registry.get(
                skill_def.id, f"=={skill_def.version}"
            )
            skill = Skill(
                id=skill_def.id,
                version=skill_def.version,
                tenant_id=tenant_id,
                front_matter=SkillFrontMatter(
                    name=fetched_card.name,
                    description=fetched_card.description,
                ),
                skill_card=fetched_card,
            )
            await registries.skill_registry.add(
                RegistryEntry(
                    id=skill_def.id,
                    version=skill_def.version,
                    tenant_id=tenant_id,
                    instance=skill,
                )
            )

    # Register skill tools when skills are declared
    if config.skills:
        from sherma.langgraph.skill_tools import create_skill_tools

        skill_tools = create_skill_tools(
            registries.skill_registry,
            registries.tool_registry,
        )
        for st in skill_tools:
            sherma_tool = from_langgraph_tool(st)
            sherma_tool.tenant_id = tenant_id
            await registries.tool_registry.add(
                RegistryEntry(
                    id=sherma_tool.id,
                    version=sherma_tool.version,
                    tenant_id=tenant_id,
                    instance=sherma_tool,
                )
            )

        # Eagerly register local tools from skill cards
        from sherma.skills.local_tools import load_local_tools_from_skill

        for skill_def in config.skills:
            if skill_def.skill_card_path:
                skill = await registries.skill_registry.get(
                    skill_def.id, f"=={skill_def.version}"
                )
                if skill.skill_card:
                    for lt in load_local_tools_from_skill(skill.skill_card):
                        sherma_tool = from_langgraph_tool(lt)
                        sherma_tool.tenant_id = tenant_id
                        await registries.tool_registry.add(
                            RegistryEntry(
                                id=sherma_tool.id,
                                version=sherma_tool.version,
                                tenant_id=tenant_id,
                                instance=sherma_tool,
                            )
                        )

    # Auto-import tools declared with import_path
    for tool_def in config.tools:
        if tool_def.import_path:
            lg_tool = import_tool(tool_def.import_path)
            sherma_tool = from_langgraph_tool(lg_tool)
            sherma_tool.id = tool_def.id
            sherma_tool.version = tool_def.version
            sherma_tool.tenant_id = tenant_id
            await registries.tool_registry.add(
                RegistryEntry(
                    id=tool_def.id,
                    version=tool_def.version,
                    tenant_id=tenant_id,
                    instance=sherma_tool,
                )
            )

    # Register sub-agents and wrap them as tools
    for sub_agent_def in config.sub_agents:
        if sub_agent_def.url:
            await registries.agent_registry.add(
                RegistryEntry(
                    id=sub_agent_def.id,
                    version=sub_agent_def.version,
                    tenant_id=tenant_id,
                    remote=True,
                    url=sub_agent_def.url,
                )
            )
        elif sub_agent_def.yaml_path:
            from sherma.langgraph.declarative.agent import DeclarativeAgent

            yaml_path = Path(sub_agent_def.yaml_path)
            if not yaml_path.exists():
                raise DeclarativeConfigError(
                    f"Sub-agent YAML file not found: {yaml_path}"
                )
            agent_instance = DeclarativeAgent(
                id=sub_agent_def.id,
                version=sub_agent_def.version,
                tenant_id=tenant_id,
                yaml_path=yaml_path,
                http_async_client=http_async_client,
            )
            await registries.agent_registry.add(
                RegistryEntry(
                    id=sub_agent_def.id,
                    version=sub_agent_def.version,
                    tenant_id=tenant_id,
                    instance=agent_instance,
                )
            )
        elif sub_agent_def.import_path:
            agent_instance = _import_agent(sub_agent_def.import_path)
            await registries.agent_registry.add(
                RegistryEntry(
                    id=sub_agent_def.id,
                    version=sub_agent_def.version,
                    tenant_id=tenant_id,
                    instance=agent_instance,
                )
            )
        # If none are set, the agent should already be registered

        agent = await registries.agent_registry.get(
            sub_agent_def.id, f"=={sub_agent_def.version}"
        )
        lg_tool = agent_to_langgraph_tool(agent)
        sherma_tool = from_langgraph_tool(lg_tool)
        sherma_tool.tenant_id = tenant_id
        await registries.tool_registry.add(
            RegistryEntry(
                id=sherma_tool.id,
                version=sherma_tool.version,
                tenant_id=tenant_id,
                instance=sherma_tool,
            )
        )


def validate_config(config: DeclarativeConfig, agent_name: str) -> None:
    """Validate declarative config constraints for a specific agent."""
    if agent_name not in config.agents:
        raise DeclarativeConfigError(f"Agent '{agent_name}' not found in config")

    agent_def = config.agents[agent_name]
    graph = agent_def.graph
    node_names = {n.name for n in graph.nodes}

    # Entry point must exist
    if graph.entry_point not in node_names:
        raise DeclarativeConfigError(
            f"Entry point '{graph.entry_point}' not found in nodes"
        )

    # All edge references must exist
    for edge in graph.edges:
        if edge.source not in node_names:
            raise DeclarativeConfigError(
                f"Edge source '{edge.source}' not found in nodes"
            )
        if edge.target is not None and edge.target not in node_names:
            if edge.target != "__end__":
                raise DeclarativeConfigError(
                    f"Edge target '{edge.target}' not found in nodes"
                )
        if edge.branches:
            if not edge.branches:
                raise DeclarativeConfigError(
                    f"Conditional edge from '{edge.source}' has no branches"
                )
            for branch in edge.branches:
                if branch.target not in node_names and branch.target != "__end__":
                    raise DeclarativeConfigError(
                        f"Branch target '{branch.target}' not found in nodes"
                    )
            if (
                edge.default
                and edge.default not in node_names
                and edge.default != "__end__"
            ):
                raise DeclarativeConfigError(
                    f"Default target '{edge.default}' not found in nodes"
                )

    # Check messages field exists if call_llm or interrupt nodes are present
    has_call_llm = any(n.type == "call_llm" for n in graph.nodes)
    has_interrupt = any(n.type == "interrupt" for n in graph.nodes)
    if has_call_llm or has_interrupt:
        field_names = {f.name for f in agent_def.state.fields}
        if "messages" not in field_names:
            raise DeclarativeConfigError(
                "State must include 'messages' field when "
                "call_llm or interrupt nodes exist"
            )

    # Validate call_llm with tools has corresponding tool_node
    for node in graph.nodes:
        if node.type == "call_llm":
            args: CallLLMArgs = node.args  # type: ignore[assignment]
            if args.tools:
                tool_node_exists = any(n.type == "tool_node" for n in graph.nodes)
                if not tool_node_exists:
                    raise DeclarativeConfigError(
                        f"call_llm node '{node.name}' has tools but no "
                        f"tool_node exists in the graph"
                    )

    # Validate that at most one dynamic tool flag is set
    for node in graph.nodes:
        if node.type == "call_llm":
            llm_args: CallLLMArgs = node.args  # type: ignore[assignment]
            exclusive_flags = sum(
                [
                    llm_args.use_tools_from_registry,
                    llm_args.use_tools_from_loaded_skills,
                    llm_args.use_sub_agents_as_tools,
                ]
            )
            if exclusive_flags > 1:
                raise DeclarativeConfigError(
                    f"call_llm node '{node.name}' cannot specify more than one of "
                    f"'use_tools_from_registry', 'use_tools_from_loaded_skills', "
                    f"and 'use_sub_agents_as_tools'"
                )
            if llm_args.use_sub_agents_as_tools and not config.sub_agents:
                raise DeclarativeConfigError(
                    f"call_llm node '{node.name}' uses 'use_sub_agents_as_tools' "
                    f"but no sub_agents are declared in the config"
                )

    # Validate that call_llm nodes with tool options have a tool_node
    has_tool_binding = any(
        n.type == "call_llm"
        and isinstance(n.args, CallLLMArgs)
        and (
            n.args.tools
            or n.args.use_tools_from_registry
            or n.args.use_tools_from_loaded_skills
            or n.args.use_sub_agents_as_tools
        )
        for n in graph.nodes
    )
    if has_tool_binding:
        has_tool_node = any(n.type == "tool_node" for n in graph.nodes)
        if not has_tool_node:
            raise DeclarativeConfigError(
                "A call_llm node with tools requires a tool_node in the graph"
            )


def populate_hooks(
    config: DeclarativeConfig,
    hook_manager: HookManager,
) -> None:
    """Import and register hook executors declared in the YAML config."""
    for hook_def in config.hooks:
        import_path = hook_def.import_path
        module_path, _, attr_name = import_path.rpartition(".")
        if not module_path:
            raise DeclarativeConfigError(
                f"Invalid hook import_path '{import_path}': "
                f"must be a dotted path like 'module.ClassName'"
            )
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise DeclarativeConfigError(
                f"Cannot import hook module '{module_path}': {exc}"
            ) from exc

        if not hasattr(module, attr_name):
            raise DeclarativeConfigError(
                f"Module '{module_path}' has no attribute '{attr_name}'"
            )

        cls = getattr(module, attr_name)
        instance = cls()
        if not isinstance(instance, HookExecutor):
            raise DeclarativeConfigError(
                f"'{import_path}' is not a HookExecutor, got {type(instance).__name__}"
            )
        hook_manager.register(instance)
