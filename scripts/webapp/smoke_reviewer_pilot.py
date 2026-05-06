#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from http.cookiejar import CookieJar
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPCookieProcessor, Request, build_opener


class SmokeClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.opener = build_opener(HTTPCookieProcessor(CookieJar()))

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict | None = None,
        expect: int = 200,
    ) -> tuple[bytes, dict[str, str]]:
        data = None
        headers = {}
        if json_body is not None:
            data = json.dumps(json_body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with self.opener.open(req, timeout=60) as response:
                body = response.read()
                status = response.status
                response_headers = dict(response.headers.items())
        except HTTPError as exc:
            body = exc.read()
            status = exc.code
            response_headers = dict(exc.headers.items())
        except URLError as exc:
            raise RuntimeError(f"{method} {path} failed: {exc}") from exc

        if status != expect:
            snippet = body[:300].decode("utf-8", errors="replace")
            raise RuntimeError(f"{method} {path} returned {status}, expected {expect}: {snippet}")
        return body, response_headers

    def json(self, method: str, path: str, *, json_body: dict | None = None) -> dict | list:
        body, _ = self.request(method, path, json_body=json_body)
        return json.loads(body.decode("utf-8"))


def assert_text(body: bytes, needle: str, label: str) -> None:
    text = body.decode("utf-8", errors="replace")
    if needle not in text:
        raise RuntimeError(f"{label} did not include {needle!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke check the deployed LocalLatin reviewer pilot.")
    parser.add_argument("--base-url", required=True, help="Origin to check, e.g. https://ai.csr.uky.edu")
    parser.add_argument("--username", required=True, help="PI/admin username for authenticated checks")
    parser.add_argument("--password", required=True, help="PI/admin password for authenticated checks")
    parser.add_argument(
        "--write-check",
        action="store_true",
        help="Submit one legacy_unresolved feedback row to verify DB write/read.",
    )
    args = parser.parse_args()

    client = SmokeClient(args.base_url)

    body, _ = client.request("GET", "/")
    assert_text(body, "LocalLatin", "frontend root")
    spa_body, _ = client.request("GET", "/review")
    assert_text(spa_body, "LocalLatin", "SPA refresh")

    models = client.json("GET", "/api/models")
    if not models:
        raise RuntimeError("/api/models returned no models")
    model = models[0]["slug"]

    signed_in = client.json(
        "POST",
        "/api/auth/signin",
        json_body={"username": args.username, "password": args.password},
    )
    if signed_in.get("role") != "pi_admin":
        raise RuntimeError("Smoke account must be PI/admin to verify stats, exports, and packets")

    stats = client.json("GET", "/api/stats")
    if "total_queries" not in stats:
        raise RuntimeError("/api/stats response missing total_queries")

    queries = client.json("GET", "/api/queries?status=all&page_size=1")
    if not queries.get("items"):
        raise RuntimeError("/api/queries returned no query items")
    query_id = queries["items"][0]["file_id"]

    predictions = client.json(
        "GET",
        f"/api/query/{query_id}/predictions?{urlencode({'model': model, 'top_k': 1})}",
    )
    if not predictions.get("predictions"):
        raise RuntimeError("/api/query/{id}/predictions returned no predictions")

    examples = client.json("GET", "/api/token_map_examples")
    if not examples:
        raise RuntimeError("/api/token_map_examples returned no token-map artifacts")
    example_id = examples[0]["example_id"]
    token_map = client.json("GET", f"/api/token_map/{example_id}")
    if "query_tokens" not in token_map:
        raise RuntimeError("/api/token_map/{id} response missing query_tokens")

    csv_body, csv_headers = client.request("GET", "/api/feedback/export")
    if "text/csv" not in csv_headers.get("Content-Type", ""):
        raise RuntimeError("/api/feedback/export did not return text/csv")
    assert_text(csv_body, "query_id", "CSV export")

    packet_body, packet_headers = client.request(
        "GET",
        f"/api/packets/review/{query_id}?{urlencode({'model': model, 'top_k': 1})}",
    )
    if "application/pdf" not in packet_headers.get("Content-Type", ""):
        raise RuntimeError("/api/packets/review did not return application/pdf")
    if not packet_body.startswith(b"%PDF"):
        raise RuntimeError("PDF packet response did not start with a PDF header")

    if args.write_check:
        note = "DEPLOY SMOKE legacy_unresolved write/read check"
        client.request(
            "POST",
            "/api/feedback",
            json_body={
                "query_id": query_id,
                "model_slug": model,
                "correct_rank": None,
                "correct_dir": None,
                "notes": note,
            },
            expect=201,
        )
        filtered, _ = client.request(
            "GET",
            f"/api/feedback/export?{urlencode({'model': model})}",
        )
        assert_text(filtered, note, "write-check CSV export")

    print("Reviewer pilot smoke checks passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"SMOKE FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
