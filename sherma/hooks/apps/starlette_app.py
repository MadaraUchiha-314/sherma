"""Starlette application builder for remote hook servers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    _starlette_installed = True
else:
    try:
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import JSONResponse
        from starlette.routing import Route

        _starlette_installed = True
    except ImportError:
        Starlette = Any
        Request = Any
        JSONResponse = Any
        Route = Any

        _starlette_installed = False

from sherma.hooks.apps.jsonrpc_handler import dispatch_jsonrpc
from sherma.hooks.handler import HookHandler


class HookStarletteApplication:
    """Builds a Starlette application that serves a :class:`HookHandler`
    over JSON-RPC 2.0.

    Example::

        from starlette.applications import Starlette

        class MyHooks(HookHandler):
            async def before_llm_call(self, params):
                params["system_prompt"] += "\\nBe concise."
                return params

        app = HookStarletteApplication(handler=MyHooks()).build()
    """

    def __init__(self, handler: HookHandler) -> None:
        if not _starlette_installed:
            raise ImportError(
                "The 'starlette' package is required to use "
                "HookStarletteApplication. "
                "Install with: pip install starlette"
            )
        self._handler = handler

    async def _handle_request(self, request: Request) -> JSONResponse:
        body = await request.json()
        response = await dispatch_jsonrpc(self._handler, body)
        return JSONResponse(response)

    def routes(self, rpc_url: str = "/hooks") -> list[Route]:
        """Return Starlette routes for the hook endpoint."""
        return [
            Route(
                rpc_url,
                self._handle_request,
                methods=["POST"],
                name="hooks",
            )
        ]

    def add_routes_to_app(
        self,
        app: Starlette,
        rpc_url: str = "/hooks",
    ) -> None:
        """Add hook routes to an existing Starlette application."""
        app.routes.extend(self.routes(rpc_url=rpc_url))

    def build(
        self,
        rpc_url: str = "/hooks",
        **kwargs: Any,
    ) -> Starlette:
        """Build and return a Starlette application.

        Args:
            rpc_url: URL path for the JSON-RPC endpoint.
            **kwargs: Extra arguments forwarded to the
                ``Starlette`` constructor.

        Returns:
            A configured Starlette application.
        """
        app = Starlette(**kwargs)
        self.add_routes_to_app(app, rpc_url=rpc_url)
        return app
