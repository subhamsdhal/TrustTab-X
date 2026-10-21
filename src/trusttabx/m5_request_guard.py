"""Bound request bodies before parsing and cap local HTTP processing time."""

from __future__ import annotations

import asyncio

from starlette.responses import JSONResponse


class RequestGuard:
    def __init__(self, app, *, max_bytes: int, timeout_seconds: float):
        self.app = app
        self.max_bytes = max_bytes
        self.timeout_seconds = timeout_seconds

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {name.lower(): value for name, value in scope.get("headers", [])}
        try:
            declared = int(headers.get(b"content-length", b"0"))
        except ValueError:
            declared = -1
        if declared < 0 or declared > self.max_bytes:
            await JSONResponse({"detail": {"code": "request_too_large_or_invalid_length"}},
                               status_code=413)(scope, receive, send)
            return
        messages = []
        size = 0
        if scope.get("method") in {"POST", "PUT", "PATCH"}:
            while True:
                try:
                    message = await asyncio.wait_for(receive(), timeout=self.timeout_seconds)
                except TimeoutError:
                    await JSONResponse({"detail": {"code": "request_body_timeout"}},
                                       status_code=408)(scope, receive, send)
                    return
                if message["type"] == "http.disconnect":
                    return
                size += len(message.get("body", b""))
                if size > self.max_bytes:
                    await JSONResponse({"detail": {"code": "request_too_large"}},
                                       status_code=413)(scope, receive, send)
                    return
                messages.append(message)
                if not message.get("more_body", False):
                    break

        async def replay():
            if messages:
                return messages.pop(0)
            return await receive()

        try:
            await asyncio.wait_for(self.app(scope, replay, send), timeout=self.timeout_seconds)
        except TimeoutError:
            await JSONResponse({"detail": {"code": "request_timeout"}},
                               status_code=504)(scope, receive, send)
