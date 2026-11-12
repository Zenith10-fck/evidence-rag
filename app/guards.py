"""Request guards, enforced in code.

- API key: header `X-API-Key`, compared with every configured caller key in constant time
  (hmac.compare_digest), so the response time does not reveal how many characters were right.
  Missing or wrong: 401. In demo mode no key is needed (nothing private is served, nothing written).
- Usage limits on the routes that cost money (/ask, /retrieve): per caller, per minute and per day
  (`app.limits`). Over a limit: 429 with Retry-After. In demo mode the limit is per client IP.
- Question length, id format: pydantic models in `app.schemas` (422).
- PDF upload limits: `app.ingest_pdf.check_pdf_upload` / `extract_pages` (422).
- Retrieval size `k` and retry count: server settings only, not request fields.
"""

from __future__ import annotations

import hmac

from fastapi import Depends, Header, HTTPException, Request, Response, status

from app.config import Settings
from app.limits import UsageLimiter

DEMO_CALLER = "demo"


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def key_matches(given: str | None, expected: str) -> bool:
    if not given or not expected:
        return False
    return hmac.compare_digest(given.encode("utf-8"), expected.encode("utf-8"))


def caller_for_key(given: str | None, callers: dict[str, str]) -> str | None:
    """The caller name whose key matches, else None. Every key is compared (no early exit), so the
    time taken does not depend on which caller matched."""
    found = None
    for name, key in callers.items():
        if key_matches(given, key) and found is None:
            found = name
    return found


def require_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(get_app_settings),
) -> str:
    """Returns the caller name (used for limits and logs; the key itself is never logged)."""
    if settings.is_demo:
        caller = DEMO_CALLER
    else:
        caller = caller_for_key(x_api_key, request.app.state.callers)
        if caller is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="missing or wrong X-API-Key header",
            )
    request.state.caller = caller
    return caller


def enforce_usage_limit(
    request: Request,
    response: Response,
    caller: str = Depends(require_api_key),
    settings: Settings = Depends(get_app_settings),
) -> str:
    """Count this request against the caller's limits; 429 if it does not fit."""
    if settings.is_demo:
        limiter: UsageLimiter = request.app.state.demo_limiter
        bucket = f"ip:{request.client.host if request.client else 'unknown'}"
    else:
        limiter = request.app.state.limiter
        bucket = caller
    decision = limiter.check(bucket)
    headers = {}
    if decision.remaining_minute is not None:
        headers["X-RateLimit-Remaining-Minute"] = str(decision.remaining_minute)
    if decision.remaining_day is not None:
        headers["X-RateLimit-Remaining-Day"] = str(decision.remaining_day)
    if not decision.allowed:
        headers["Retry-After"] = str(decision.retry_after_s)
        what = "per-minute" if decision.reason == "per_minute" else "daily"
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"{what} limit reached for this caller; retry after "
            f"{decision.retry_after_s} s",
            headers=headers,
        )
    response.headers.update(headers)
    return caller
