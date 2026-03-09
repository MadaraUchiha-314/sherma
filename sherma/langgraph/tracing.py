from __future__ import annotations

from typing import Any

from langfuse.langchain import CallbackHandler


def get_langfuse_config(
    *,
    public_key: str | None = None,
    secret_key: str | None = None,
    host: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Create a LangGraph config dict with LangFuse callback handler.

    Returns a config suitable for passing to graph.ainvoke(input, config=...).
    """
    handler_kwargs: dict[str, Any] = {}
    if public_key is not None:
        handler_kwargs["public_key"] = public_key
    if secret_key is not None:
        handler_kwargs["secret_key"] = secret_key
    if host is not None:
        handler_kwargs["host"] = host
    handler_kwargs.update(kwargs)

    handler = CallbackHandler(**handler_kwargs)
    return {"callbacks": [handler]}
