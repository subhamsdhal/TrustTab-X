"""Fail-closed local/internal service controls; never persist credential values."""

from __future__ import annotations

import os
import secrets
from collections import deque
from dataclasses import dataclass
from threading import Lock
from time import monotonic

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-TrustTabX-Key", auto_error=False)


@dataclass(frozen=True)
class SecurityConfig:
    operator_key: str
    admin_key: str
    max_request_bytes: int
    request_timeout_seconds: float
    rate_limit_per_minute: int
    allowed_hosts: list[str]

    @classmethod
    def from_env(cls) -> SecurityConfig:
        operator = os.getenv("TRUSTTABX_API_KEY", "")
        admin = os.getenv("TRUSTTABX_ADMIN_KEY", "")
        if len(operator) < 32 or len(admin) < 32 or secrets.compare_digest(operator, admin):
            raise ValueError("Distinct TRUSTTABX_API_KEY and TRUSTTABX_ADMIN_KEY of 32+ chars required")
        max_bytes = int(os.getenv("TRUSTTABX_MAX_REQUEST_BYTES", "262144"))
        timeout = float(os.getenv("TRUSTTABX_REQUEST_TIMEOUT_SECONDS", "15"))
        rate = int(os.getenv("TRUSTTABX_RATE_LIMIT_PER_MINUTE", "120"))
        if not 4096 <= max_bytes <= 1048576 or not 1 <= timeout <= 120 or not 1 <= rate <= 10000:
            raise ValueError("Invalid M5 request limit configuration")
        hosts = [item.strip() for item in os.getenv(
            "TRUSTTABX_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")]
        if not hosts or any(not host or host == "*" for host in hosts):
            raise ValueError("Explicit non-wildcard TRUSTTABX_ALLOWED_HOSTS required")
        return cls(operator, admin, max_bytes, timeout, rate, hosts)


class RateLimiter:
    """Per-process sliding minute window, bounded by configured rate and identities."""

    def __init__(self, limit: int):
        self.limit = limit
        self.window: dict[str, deque[float]] = {}
        self.lock = Lock()

    def allow(self, identity: str) -> bool:
        now = monotonic()
        with self.lock:
            ticks = self.window.setdefault(identity, deque())
            while ticks and ticks[0] <= now - 60:
                ticks.popleft()
            if len(ticks) >= self.limit:
                return False
            ticks.append(now)
            return True


def _authorize(request: Request, key: str | None, *, admin_only: bool) -> str:
    config: SecurityConfig = request.app.state.security
    role = "invalid"
    if key and secrets.compare_digest(key, config.admin_key):
        role = "admin"
    elif key and secrets.compare_digest(key, config.operator_key):
        role = "operator"
    if not request.app.state.rate_limiter.allow(role):
        raise HTTPException(429, detail={"code": "rate_limited"}, headers={"Retry-After": "60"})
    if role == "invalid":
        raise HTTPException(401, detail={"code": "authentication_required"})
    if admin_only and role != "admin":
        raise HTTPException(403, detail={"code": "admin_required"})
    origin = request.headers.get("origin")
    if origin and origin != str(request.base_url).rstrip("/"):
        raise HTTPException(403, detail={"code": "origin_rejected"})
    request.state.operation_identity = role
    return role


def require_operator(request: Request, key: str | None = Security(API_KEY_HEADER)) -> str:
    return _authorize(request, key, admin_only=False)


def require_admin(request: Request, key: str | None = Security(API_KEY_HEADER)) -> str:
    return _authorize(request, key, admin_only=True)
