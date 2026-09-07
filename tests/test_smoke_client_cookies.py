"""Cookie handling in scripts/webapp/smoke_reviewer_pilot.py (issue #126).

The first deploy that actually had LOCALLATIN_SMOKE_USERNAME/PASSWORD set failed
at the first PI/admin call:

    SMOKE FAILED: GET /api/stats returned 401, expected 200

Signin itself had succeeded. The cause was the cookie jar, not the credentials:
web/config.production.yaml sets ``auth.secure_cookies: true`` so the session
cookie is marked Secure, and deploy/deploy.sh checks the service over
``http://127.0.0.1:8080``. A stock ``http.cookiejar.CookieJar`` stores such a
cookie but never sends it back over plain http, so every authenticated request
went out anonymous.

These tests pin both halves of the fix: the session cookie comes back on the
loopback origin the deploy uses, and a non-loopback http origin still does not
get it.

The script is loaded by path rather than imported as a module, because
scripts/webapp is not a package and tests/ runs on Python 3.10 in CI where the
web package cannot import at all.
"""

from __future__ import annotations

import email.message
import importlib.util
import sys
from pathlib import Path
from urllib.request import Request

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "webapp" / "smoke_reviewer_pilot.py"

# Mirrors the cookie web/routers/auth.py sets under the production config:
# httponly, Secure, SameSite=lax.
SECURE_SESSION_COOKIE = "ll_session=abc123; Path=/; HttpOnly; Secure; SameSite=lax"


@pytest.fixture(scope="module")
def smoke_module():
    spec = importlib.util.spec_from_file_location("smoke_reviewer_pilot", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeResponse:
    """The two-method duck type CookieJar.extract_cookies needs."""

    def __init__(self, set_cookie: str) -> None:
        self._headers = email.message.Message()
        self._headers["Set-Cookie"] = set_cookie

    def info(self) -> email.message.Message:
        return self._headers


def _round_trip(jar, origin: str) -> str | None:
    """Store a Secure session cookie from `origin`, then ask what goes back."""
    jar.extract_cookies(
        _FakeResponse(SECURE_SESSION_COOKIE),
        Request(f"{origin}/api/auth/signin", method="POST"),
    )
    follow_up = Request(f"{origin}/api/stats")
    jar.add_cookie_header(follow_up)
    return follow_up.get_header("Cookie")


# Every member of LOOPBACK_HOSTS, in the spellings a base URL can carry. `::1`
# is here because it is the one member the original suite never exercised, and
# because it is the case a naive host parse gets wrong: http.cookiejar's own
# request_host() strips the port with a `:\d+$` regex and keeps the brackets,
# yielding "[::1]", which would never match the frozenset.
@pytest.mark.parametrize(
    "origin",
    [
        "http://127.0.0.1:8080",
        "http://localhost:8080",
        "http://[::1]:8080",
    ],
)
def test_secure_cookie_is_returned_to_a_loopback_origin(smoke_module, origin):
    from http.cookiejar import CookieJar

    jar = CookieJar(policy=smoke_module.LoopbackSecureCookiePolicy())
    assert _round_trip(jar, origin) == "ll_session=abc123"


# The security argument for this policy is that the host test is exact match on
# a frozenset, not a substring or suffix test. These are the near-miss hostnames
# a widening refactor would start accepting, so together they pin the semantics:
# `localhost.evil.com` / `127.0.0.1.nip.io` catch a substring test (`"127.0.0.1"
# in hostname`), and `evil.localhost` / `attacker-127.0.0.1` catch a suffix test
# (`hostname.endswith("localhost")`). A prefix-shaped case alone passes the
# suffix widening, which is why both shapes are here.
@pytest.mark.parametrize(
    "origin",
    [
        "http://ai.csr.uky.edu",
        "http://localhost.evil.com",
        "http://127.0.0.1.nip.io",
        "http://evil.localhost",
        "http://attacker-127.0.0.1",
    ],
)
def test_secure_cookie_is_withheld_from_a_non_loopback_http_origin(
    smoke_module, origin
):
    from http.cookiejar import CookieJar

    jar = CookieJar(policy=smoke_module.LoopbackSecureCookiePolicy())
    assert _round_trip(jar, origin) is None


def test_stock_policy_reproduces_the_401():
    """The bug itself, so the regression cannot come back unnoticed."""
    from http.cookiejar import CookieJar

    assert _round_trip(CookieJar(), "http://127.0.0.1:8080") is None


def test_client_uses_the_relaxed_policy(smoke_module):
    """The policy must actually be wired into the client the smoke run builds.

    Asserted behaviourally, by round-tripping a Secure cookie through the jar
    the client's own opener carries: that survives a stdlib rename of the
    private policy attribute, and it tests the property the deploy depends on
    rather than the type that implements it.
    """
    client = smoke_module.SmokeClient("http://127.0.0.1:8080")
    jars = [
        handler.cookiejar
        for handler in client.opener.handlers
        if hasattr(handler, "cookiejar")
    ]
    assert jars, "SmokeClient built no cookie-processing handler"
    for jar in jars:
        assert _round_trip(jar, "http://127.0.0.1:8080") == "ll_session=abc123"
        assert _round_trip(jar, "http://localhost.evil.com") is None
