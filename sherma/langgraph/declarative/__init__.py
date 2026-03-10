"""Declarative agent module: define agents via YAML + CEL."""

from sherma.langgraph.declarative.agent import DeclarativeAgent
from sherma.langgraph.declarative.loader import load_declarative_config
from sherma.langgraph.declarative.schema import DeclarativeConfig
from sherma.langgraph.declarative.transform import inject_tool_nodes

__all__ = [
    "DeclarativeAgent",
    "DeclarativeConfig",
    "inject_tool_nodes",
    "load_declarative_config",
]
