#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
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

    def json(
        self,
        method: str,
        path: str,
        *,
        json_body: dict | None = None,
        expect: int = 200,
    ) -> dict | list:
        body, _ = self.request(method, path, json_body=json_body, expect=expect)
        return json.loads(body.decode("utf-8"))


def assert_text(body: bytes, needle: str, label: str) -> None:
    text = body.decode("utf-8", errors="replace")
    if needle not in text:
        raise RuntimeError(f"{label} did not include {needle!r}")


# Mirrors web/variants.py. A deployment that serves fewer than these four has a
# missing predictions CSV, which is exactly the failure the data release is
# meant to prevent, so the smoke run treats it as fatal rather than degraded.
EXPECTED_VARIANTS = ("raw", "abtt", "sif", "sif_abtt")

# The attribution artifacts call the no-post-processing variant "baseline"
# while the prediction CSVs call it "raw" (see web/routers/token_map.py and the
# frontend's toAttributionVariant).
ATTRIBUTION_VARIANT = {
    "raw": "baseline",
    "abtt": "abtt",
    "sif": "sif",
    "sif_abtt": "sif_abtt",
}


def check_models_advertise_variants(models: list) -> tuple[list[str], str]:
    """Every model must advertise all four variants. Returns (variants, default)."""
    for entry in models:
        advertised = entry.get("available_variants") or []
        missing = [v for v in EXPECTED_VARIANTS if v not in advertised]
        if missing:
            raise RuntimeError(
                f"/api/models: {entry.get('slug')} does not advertise variants {missing} "
                f"(advertised: {advertised}) — a predictions CSV is missing on the host"
            )
        if entry.get("default_variant") not in advertised:
            raise RuntimeError(
                f"/api/models: {entry.get('slug')} default_variant "
                f"{entry.get('default_variant')!r} is not among {advertised}"
            )
    return list(models[0]["available_variants"]), models[0]["default_variant"]


def check_predictions_per_variant(
    client: SmokeClient,
    query_id: int,
    model: str,
    variants: list[str],
    default_variant: str,
) -> None:
    """Each variant serves its own ranked list, and omitting ?variant= uses the default."""
    top1: dict[str, str] = {}
    for variant in variants:
        params = urlencode({"model": model, "variant": variant, "top_k": 3})
        payload = client.json("GET", f"/api/query/{query_id}/predictions?{params}")
        predictions = payload.get("predictions") or []
        if not predictions:
            raise RuntimeError(f"predictions for variant {variant!r} came back empty")
        if payload.get("variant") != variant:
            raise RuntimeError(
                f"predictions echoed variant {payload.get('variant')!r}, asked for {variant!r}"
            )
        top1[variant] = predictions[0]["dir_name"]

    implicit = client.json(
        "GET",
        f"/api/query/{query_id}/predictions?{urlencode({'model': model, 'top_k': 1})}",
    )
    if implicit.get("variant") != default_variant:
        raise RuntimeError(
            f"predictions without ?variant= served {implicit.get('variant')!r}, "
            f"expected the configured default {default_variant!r}"
        )
    print(f"  variants OK for query {query_id} ({model}): {top1}")


def check_token_map_per_variant(client: SmokeClient, example_id: int, variants: list[str]) -> None:
    """The filtered token-map fetch the webapp makes must be non-empty per variant.

    Filtered exactly as the frontend fetches it: unfiltered, this response
    carries all 7 x 4 attribution matrices (20+ MB on the largest artifact) and
    would dominate the smoke run.
    """
    for variant in variants:
        attribution = ATTRIBUTION_VARIANT.get(variant, variant)
        params = urlencode({"method": "ig", "variant": attribution})
        token_map = client.json("GET", f"/api/token_map/{example_id}?{params}")
        if not token_map.get("query_tokens"):
            raise RuntimeError(f"token_map {example_id} variant {attribution!r} has no query_tokens")
        if attribution not in (token_map.get("available_variants") or []):
            raise RuntimeError(
                f"token_map {example_id} does not carry attribution variant {attribution!r} "
                f"(has {token_map.get('available_variants')}) — stale IG artifacts on the host"
            )
        matrix = (token_map.get("pair_matrices") or {}).get("ig", {}).get(attribution)
        if not matrix:
            raise RuntimeError(
                f"token_map {example_id} returned no ig/{attribution} matrix for the filtered fetch"
            )
    print(f"  token maps OK for example {example_id} across {len(variants)} variants")


def check_notes_round_trip(
    client: SmokeClient,
    query_id: int,
    model: str,
    variant: str,
    other_variant: str,
) -> None:
    """POST feedback with notes + multi-select, then read it back through /feedback/latest.

    This is the reload path the reviewer sees: reopening a query must prefill
    the notes and the selected ranks saved for *that* variant, and must not
    leak a note saved under a different variant.
    """
    note = f"DEPLOY SMOKE notes round-trip {variant} {int(time.time())}"
    created = client.json(
        "POST",
        "/api/feedback",
        json_body={
            "query_id": query_id,
            "model_slug": model,
            "variant": variant,
            "selected_ranks": [1, 2],
            "notes": note,
        },
        expect=201,
    )
    if created.get("variant") != variant:
        raise RuntimeError(f"feedback saved under variant {created.get('variant')!r}, sent {variant!r}")

    params = urlencode({"query_id": query_id, "model": model, "variant": variant})
    latest = client.json("GET", f"/api/feedback/latest?{params}")
    if not latest:
        raise RuntimeError(f"/api/feedback/latest returned nothing for variant {variant!r}")
    if latest.get("notes") != note:
        raise RuntimeError(
            f"/api/feedback/latest notes did not round-trip: {latest.get('notes')!r} != {note!r}"
        )
    if latest.get("selected_ranks") != [1, 2]:
        raise RuntimeError(
            f"/api/feedback/latest selected_ranks did not round-trip: {latest.get('selected_ranks')!r}"
        )
    if latest.get("variant") != variant:
        raise RuntimeError(f"/api/feedback/latest returned variant {latest.get('variant')!r}")

    other_params = urlencode({"query_id": query_id, "model": model, "variant": other_variant})
    other = client.json("GET", f"/api/feedback/latest?{other_params}")
    if other and other.get("notes") == note:
        raise RuntimeError(
            f"note saved for {variant!r} leaked into the {other_variant!r} prefill"
        )
    print(f"  notes + multi-select round-tripped for variant {variant!r}")


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
    variants, default_variant = check_models_advertise_variants(models)
    print(f"  /api/models advertises {variants}, default {default_variant!r}")

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

    smoke_username = f"smoke_reviewer_{int(time.time())}"
    smoke_password = "correct horse battery staple"
    pending_client = SmokeClient(args.base_url)
    registered = pending_client.json(
        "POST",
        "/api/auth/register",
        json_body={
            "username": smoke_username,
            "display_name": "Deploy Smoke Reviewer",
            "password": smoke_password,
        },
        expect=201,
    )
    if registered.get("status") != "pending_approval":
        raise RuntimeError("Smoke reviewer registration did not return pending_approval")
    account = registered.get("account") or {}
    if account.get("approval_status") != "pending":
        raise RuntimeError("Smoke reviewer account was not created as pending")
    account_id = account.get("id")

    pending_client.request("GET", "/api/queries?status=all&page_size=1", expect=401)
    pending_client.request(
        "POST",
        "/api/auth/signin",
        json_body={"username": smoke_username, "password": smoke_password},
        expect=403,
    )

    pending_accounts = client.json("GET", "/api/auth/accounts?status=pending")
    if not any(item.get("username") == smoke_username for item in pending_accounts):
        raise RuntimeError("PI/admin could not see smoke reviewer in pending accounts")
    approved = client.json("POST", f"/api/auth/accounts/{account_id}/approve")
    if approved.get("approval_status") != "approved":
        raise RuntimeError("PI/admin account approval did not mark reviewer approved")

    approved_client = SmokeClient(args.base_url)
    approved_signin = approved_client.json(
        "POST",
        "/api/auth/signin",
        json_body={"username": smoke_username, "password": smoke_password},
    )
    if approved_signin.get("role") != "reviewer":
        raise RuntimeError("Approved smoke reviewer could not sign in as reviewer")
    approved_client.request("GET", "/api/queries?status=all&page_size=1")
    approved_client.request("GET", "/api/stats", expect=403)
    client.json("POST", f"/api/auth/accounts/{account_id}/reject")

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

    check_predictions_per_variant(client, query_id, model, variants, default_variant)

    examples = client.json("GET", "/api/token_map_examples")
    if not examples:
        raise RuntimeError("/api/token_map_examples returned no token-map artifacts")
    example_id = examples[0]["example_id"]
    check_token_map_per_variant(client, example_id, variants)

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

        # The reviewer-facing reload path, per variant. Writes, so it lives
        # behind --write-check with the other DB mutations.
        other = next(v for v in variants if v != default_variant)
        check_notes_round_trip(client, query_id, model, default_variant, other)
        check_notes_round_trip(client, query_id, model, other, default_variant)

    print("Reviewer pilot smoke checks passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"SMOKE FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
