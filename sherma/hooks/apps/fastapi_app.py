"""FastAPI application builder for remote hook servers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import FastAPI

    _fastapi_installed = True
else:
    try:
        from fastapi import FastAPI

        _fastapi_installed = True
    except ImportError:
        FastAPI = Any

        _fastapi_installed = False

from sherma.hooks.apps.jsonrpc_handler import dispatch_jsonrpc
from sherma.hooks.handler import HookHandler


class HookFastAPIApplication:
    """Builds a FastAPI application that serves a :class:`HookHandler`
    over JSON-RPC 2.0.

    Example::

        class MyHooks(HookHandler):
            async def before_llm_call(self, params):
                params["system_prompt"] += "\\nBe concise."
                return params

        app = HookFastAPIApplication(handler=MyHooks()).build()
    """

    def __init__(self, handler: HookHandler) -> None:
        if not _fastapi_installed:
            raise ImportError(
                "The 'fastapi' package is required to use "
                "HookFastAPIApplication. "
                "Install with: pip install fastapi"
            )
        self._handler = handler

    def add_routes_to_app(
        self,
        app: FastAPI,
        rpc_url: str = "/hooks",
    ) -> None:
        """Add hook routes to an existing FastAPI application."""
        from starlette.requests import Request
        from starlette.responses import JSONResponse

        @app.post(rpc_url)
        async def hooks_endpoint(request: Request) -> JSONResponse:
            body = await request.json()
            response = await dispatch_jsonrpc(self._handler, body)
            return JSONResponse(response)

    def build(
        self,
        rpc_url: str = "/hooks",
        **kwargs: Any,
    ) -> FastAPI:
        """Build and return a FastAPI application.

        Args:
            rpc_url: URL path for the JSON-RPC endpoint.
            **kwargs: Extra arguments forwarded to the
                ``FastAPI`` constructor.

        Returns:
            A configured FastAPI application.
        """
        app = FastAPI(**kwargs)
        self.add_routes_to_app(app, rpc_url=rpc_url)
        return app
