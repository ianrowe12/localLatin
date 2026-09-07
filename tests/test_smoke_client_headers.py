"""Response-header handling in scripts/webapp/smoke_reviewer_pilot.py (issue #127).

The first deploy whose authenticated smoke run got past signin (the cookie fix
in #129 bought that) failed at the next header-shaped assertion:

    SMOKE FAILED: /api/feedback/export did not return text/csv

The endpoint was fine. ``SmokeClient.send`` collapsed the response headers into
a plain ``dict``, throwing away the case-insensitivity of the
``email.message.Message`` urllib parses them into, and uvicorn writes header
names lowercase. So ``headers.get("Content-Type")`` was always ``""`` and the
two content-type assertions in the script -- text/csv for
``/api/feedback/export``, application/pdf for ``/api/packets/review`` -- could
never pass against the real server, only against a mock.

web/tests could not have caught this: it reaches the same endpoints through
httpx, whose headers are case-insensitive already, so the bug lived entirely in
the deploy client.

These tests drive a real socket, because that is the layer the bug lived at: a
unit test on the mapping alone would have passed even with the old
``dict(...)``, given headers spelled the way the assertion spelled them.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "webapp" / "smoke_reviewer_pilot.py"


@pytest.fixture(scope="module")
def smoke_module():
    spec = importlib.util.spec_from_file_location("smoke_reviewer_pilot", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _LowercaseHeaderHandler(BaseHTTPRequestHandler):
    """Answers with lowercase header names, the way uvicorn writes them."""

    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802 (stdlib signature)
        body = b"query_id,filename\n1,BN2123.89r.5.txt\n"
        self.send_response(200)
        self.send_header("content-type", "text/csv; charset=utf-8")
        self.send_header("content-disposition", "attachment; filename=feedback_export.csv")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:  # noqa: ANN002 (stdlib signature)
        pass


@pytest.fixture()
def lowercase_header_server():
    server = HTTPServer(("127.0.0.1", 0), _LowercaseHeaderHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_content_type_check_survives_lowercase_header_names(
    smoke_module, lowercase_header_server
):
    """The assertion the deploy actually makes, over a real socket."""
    client = smoke_module.SmokeClient(lowercase_header_server)
    body, headers = client.request("GET", "/api/feedback/export")

    # Spelled exactly as smoke_reviewer_pilot.py spells it at the call site.
    assert "text/csv" in headers.get("Content-Type", "")
    assert b"query_id" in body


@pytest.mark.parametrize("spelling", ["Content-Type", "content-type", "CONTENT-TYPE"])
def test_any_spelling_reaches_the_same_header(
    smoke_module, lowercase_header_server, spelling
):
    client = smoke_module.SmokeClient(lowercase_header_server)
    _, headers = client.request("GET", "/api/feedback/export")

    assert headers.get(spelling, "") == "text/csv; charset=utf-8"
    assert headers[spelling] == "text/csv; charset=utf-8"
    assert spelling in headers


def test_a_missing_header_still_reads_as_absent(smoke_module, lowercase_header_server):
    """The mapping must not invent headers: a real absence has to stay one."""
    client = smoke_module.SmokeClient(lowercase_header_server)
    _, headers = client.request("GET", "/api/feedback/export")

    assert headers.get("X-Not-Sent") is None
    assert headers.get("X-Not-Sent", "") == ""
    assert "X-Not-Sent" not in headers


def test_a_plain_dict_would_have_failed(smoke_module, lowercase_header_server):
    """Pin the bug itself, so the cheaper `dict(...)` cannot come back.

    This is the line smoke_reviewer_pilot.py used to run, against the same
    server, producing the empty string that failed the deploy.
    """
    from urllib.request import urlopen

    with urlopen(f"{lowercase_header_server}/api/feedback/export", timeout=10) as response:
        naive = dict(response.headers.items())

    assert naive.get("Content-Type", "") == ""
    assert smoke_module.ResponseHeaders(naive.items()).get("Content-Type") is not None
