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

    def send(
        self,
        method: str,
        path: str,
        *,
        json_body: dict | None = None,
    ) -> tuple[int, bytes, dict[str, str]]:
        """Perform the request and report the status instead of asserting one.

        Used by checks whose expected outcome is genuinely one of several -- a
        reviewer directory can only be seeded from a query that has neither been
        seeded already (409) nor been guard-excluded (422), and which of those a
        given query is cannot be known before asking. Every check that expects a
        single status still goes through request(), which asserts it.
        """
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
                return response.status, response.read(), dict(response.headers.items())
        except HTTPError as exc:
            return exc.code, exc.read(), dict(exc.headers.items())
        except URLError as exc:
            raise RuntimeError(f"{method} {path} failed: {exc}") from exc

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict | None = None,
        expect: int = 200,
    ) -> tuple[bytes, dict[str, str]]:
        status, body, response_headers = self.send(method, path, json_body=json_body)
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

# web/bands.py. The PI fixed these two numbers in the 2026-08-25 meeting, and the
# backend now owns them so that the reviewer-directory badge and the frontend
# banding cannot drift apart. The smoke run pins the values, not just the shape:
# a deploy that quietly moved the no-match line would change what every reviewer
# is told to do without anybody noticing.
EXPECTED_BANDS = {"no_match": 0.5, "verify": 0.7}

# The shape web/models.py::FeedbackEntry promises. reviewer_username is the
# issue #96 addition that lets a shared note be attributed to a person.
FEEDBACK_ENTRY_KEYS = (
    "query_id",
    "model_slug",
    "variant",
    "outcome",
    "notes",
    "reviewer",
    "reviewer_username",
    "timestamp",
)


def check_confidence_bands(models: list) -> dict:
    """/api/models serves the 0.5 / 0.7 confidence thresholds the frontend renders.

    The frontend keeps literals of its own only as a pre-flight fallback and
    otherwise reads these (utils/confidenceBands.ts::bandsFrom). If the key is
    absent the UI silently falls back, so reviewers would still see bands and
    nobody would learn that the deployed backend is older than the frontend.
    That makes the presence of the key, not the appearance of the banner, the
    thing worth asserting here.
    """
    for entry in models:
        bands = entry.get("confidence_bands")
        if not isinstance(bands, dict):
            raise RuntimeError(
                f"/api/models: {entry.get('slug')} carries no confidence_bands object "
                f"(got {bands!r}) — the deployed backend predates issue #94"
            )
        for name, expected in EXPECTED_BANDS.items():
            actual = bands.get(name)
            if not isinstance(actual, (int, float)):
                raise RuntimeError(
                    f"/api/models: {entry.get('slug')} confidence_bands.{name} is {actual!r}, "
                    "expected a number"
                )
            if abs(float(actual) - expected) > 1e-9:
                raise RuntimeError(
                    f"/api/models: {entry.get('slug')} confidence_bands.{name} is {actual}, "
                    f"expected {expected} — the deployed thresholds are not the ones agreed "
                    "with the PI"
                )
        if float(bands["no_match"]) >= float(bands["verify"]):
            raise RuntimeError(
                f"/api/models: {entry.get('slug')} bands are not ordered: "
                f"no_match {bands['no_match']} >= verify {bands['verify']}"
            )
    print(f"  /api/models serves confidence bands {EXPECTED_BANDS} for {len(models)} model(s)")
    return dict(EXPECTED_BANDS)


def check_reviewer_dir_support(client: SmokeClient, models: list) -> None:
    """Every model must advertise reviewer-directory support and list its directories.

    supports_reviewer_dirs is `slug in store.qq_paths`, so a false here means the
    model's qq_sim_<slug>.npz did not reach the host. That is the same class of
    failure as a missing predictions CSV -- the feature degrades silently, with
    reviewer-created directories simply never appearing as candidates -- and the
    data release exists to prevent exactly it, so it is fatal rather than a
    warning.
    """
    unsupported = [m.get("slug") for m in models if not m.get("supports_reviewer_dirs")]
    if unsupported:
        raise RuntimeError(
            f"/api/models: {unsupported} report supports_reviewer_dirs=false — their "
            "qq_sim_<slug>.npz matrices are missing from the host, so reviewer-created "
            "directories would never be scored as candidates"
        )

    slug = models[0]["slug"]
    listed = client.json("GET", f"/api/reviewer_dirs?{urlencode({'model': slug})}")
    if not isinstance(listed, list):
        raise RuntimeError(f"/api/reviewer_dirs returned {type(listed).__name__}, expected a list")
    print(
        f"  reviewer directories supported by all {len(models)} model(s); "
        f"{len(listed)} existing for {slug!r}"
    )


def check_predictions_carry_reviewer_dirs(
    client: SmokeClient, query_id: int, model: str, variant: str
) -> None:
    """A predictions response carries the seeded_dirs field the new-directory UI reads.

    Read-only: the field is present and well-typed even when the reviewer has
    created nothing, which is the state a fresh deploy is in.
    """
    params = urlencode({"model": model, "variant": variant, "top_k": 3})
    payload = client.json("GET", f"/api/query/{query_id}/predictions?{params}")
    seeded = payload.get("seeded_dirs")
    if not isinstance(seeded, list):
        raise RuntimeError(
            f"/api/query/{query_id}/predictions has no seeded_dirs list (got {seeded!r}) — "
            "the deployed backend predates issue #95"
        )
    for prediction in payload.get("predictions") or []:
        if prediction.get("source") not in ("model", "reviewer"):
            raise RuntimeError(
                f"prediction rank {prediction.get('rank')} has source "
                f"{prediction.get('source')!r}, expected 'model' or 'reviewer'"
            )
    print(f"  predictions carry seeded_dirs and per-candidate source for query {query_id}")


def check_shared_note_shape(client: SmokeClient, query_id: int, model: str, variant: str) -> None:
    """/api/feedback/latest answers with the shared-note prefill shape, or null.

    Read-only, so it must tolerate a query nobody has reviewed yet: the endpoint
    legitimately answers JSON null there. When a row does come back, every field
    the reviewer's reload path reads must be present -- in particular
    reviewer_username, which is what lets the UI say whose note it is showing.
    A backend predating issue #96 answers without that key and the attribution
    line would silently render blank.
    """
    params = urlencode({"query_id": query_id, "model": model, "variant": variant})
    latest = client.json("GET", f"/api/feedback/latest?{params}")
    if latest is None:
        print(f"  /api/feedback/latest shape OK for query {query_id} (no notes filed yet)")
        return
    if not isinstance(latest, dict):
        raise RuntimeError(
            f"/api/feedback/latest returned {type(latest).__name__}, expected an object or null"
        )
    missing = [key for key in FEEDBACK_ENTRY_KEYS if key not in latest]
    if missing:
        raise RuntimeError(
            f"/api/feedback/latest is missing field(s) {missing} — the deployed backend "
            "predates issue #96 and shared-note attribution would render blank"
        )
    if latest.get("variant") != variant:
        raise RuntimeError(
            f"/api/feedback/latest returned variant {latest.get('variant')!r}, asked for {variant!r}"
        )
    print(
        f"  /api/feedback/latest shape OK for query {query_id} "
        f"(note by {latest.get('reviewer_username') or latest.get('reviewer')!r})"
    )


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

    /feedback/latest is shared across reviewers (issue #96), so the read-back
    asserts that the row this run just appended is the newest one for the
    query -- which it is, having been written moments earlier -- rather than
    that it belongs to the smoke account. The leak check still holds either
    way: the note text is stamped with a timestamp and is unique per run.
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

    # The same single row also proves the DB write reaches the CSV export, so
    # no second throwaway row is needed for that.
    exported, _ = client.request("GET", f"/api/feedback/export?{urlencode({'model': model})}")
    assert_text(exported, note, "write-check CSV export")

    print(f"  notes + multi-select round-tripped for variant {variant!r} (1 row written)")


def check_reviewer_dir_create_round_trip(
    client: SmokeClient,
    model: str,
    variant: str,
    candidate_query_ids: list[int],
) -> None:
    """Seed one reviewer directory, then prove a second seed on the same query is 409.

    WRITE CHECK. Reviewer directories are append-only and nothing can ever delete
    one, so this leaves exactly one permanent directory on the deployment per
    run -- which is why it sits behind the write flag rather than running on
    every deploy. It is scoped to one directory deliberately: the 409 half needs
    no second create, because the duplicate guard is keyed on the seed query.

    Seeding can legitimately fail on a given query -- 409 if some earlier run or
    a real reviewer already seeded it, 422 if the document is guard-excluded and
    could therefore never be matched -- so the check walks candidates until one
    is accepted rather than assuming the first is usable.
    """
    seeded_id = None
    seed_query = None
    skipped: list[str] = []
    for query_id in candidate_query_ids:
        status, body, _ = client.send(
            "POST",
            "/api/reviewer_dirs",
            json_body={
                "query_file_id": query_id,
                "label": f"DEPLOY SMOKE directory {int(time.time())}",
                "model_slug": model,
                "variant": variant,
            },
        )
        if status == 201:
            created = json.loads(body.decode("utf-8"))
            seeded_id = created.get("dir_id")
            seed_query = query_id
            if created.get("status") != "awaiting_match":
                raise RuntimeError(
                    f"a freshly created reviewer directory reported status "
                    f"{created.get('status')!r}, expected 'awaiting_match'"
                )
            if created.get("member_query_ids") != [query_id]:
                raise RuntimeError(
                    f"new reviewer directory members are {created.get('member_query_ids')!r}, "
                    f"expected exactly [{query_id}]"
                )
            break
        if status in (409, 422):
            skipped.append(f"{query_id}:{status}")
            continue
        snippet = body[:300].decode("utf-8", errors="replace")
        raise RuntimeError(
            f"POST /api/reviewer_dirs for query {query_id} returned {status}, "
            f"expected 201, 409 or 422: {snippet}"
        )

    if seeded_id is None:
        raise RuntimeError(
            "could not seed a reviewer directory from any candidate query "
            f"(tried {len(candidate_query_ids)}: {skipped}) — every one was already "
            "seeded or guard-excluded"
        )

    # The duplicate guard is what stops a double-click creating a second
    # permanent artefact, so it is the half most worth proving.
    duplicate, dup_body, _ = client.send(
        "POST",
        "/api/reviewer_dirs",
        json_body={"query_file_id": seed_query, "model_slug": model, "variant": variant},
    )
    if duplicate != 409:
        snippet = dup_body[:300].decode("utf-8", errors="replace")
        raise RuntimeError(
            f"re-seeding query {seed_query} returned {duplicate}, expected 409 — the "
            f"duplicate guard is not holding: {snippet}"
        )

    fetched = client.json("GET", f"/api/reviewer_dirs/{seeded_id}?{urlencode({'model': model})}")
    if fetched.get("seed_query_id") != seed_query:
        raise RuntimeError(
            f"reviewer directory {seeded_id} reports seed_query_id "
            f"{fetched.get('seed_query_id')!r}, expected {seed_query}"
        )
    print(
        f"  reviewer directory {seeded_id} seeded from query {seed_query} "
        "and re-seeding it returned 409 (1 permanent directory created)"
    )


def check_password_change_round_trip(base_url: str, username: str, password: str) -> None:
    """Change the smoke account's password and change it straight back.

    WRITE CHECK. This is the one check that can invalidate the credentials the
    deploy pipeline itself depends on, so it is deliberately narrow: it runs on
    its own session, restores the original password, and then proves the
    restore by signing in again from a clean client.

    The interim password is derived from the original by a fixed rule rather
    than generated, so that a failure between the two changes is recoverable by
    reading this source -- no secret is ever printed, and there is no random
    value that would be lost with the process.
    """
    interim = f"{password}.smoke-rotate"

    session = SmokeClient(base_url)
    session.json("POST", "/api/auth/signin", json_body={"username": username, "password": password})

    changed = session.json(
        "POST",
        "/api/auth/change_password",
        json_body={"current_password": password, "new_password": interim},
    )
    if changed.get("must_change_password"):
        raise RuntimeError("change_password left must_change_password set")

    status, body, _ = session.send(
        "POST",
        "/api/auth/change_password",
        json_body={"current_password": interim, "new_password": password},
    )
    if status != 200:
        snippet = body[:300].decode("utf-8", errors="replace")
        raise RuntimeError(
            f"RESTORE FAILED: the smoke account password is still the interim value "
            f"(original + '.smoke-rotate'). Rotate it with the 'Provision smoke account' "
            f"workflow before the next deploy. Server said {status}: {snippet}"
        )

    verifier = SmokeClient(base_url)
    verify_status, verify_body, _ = verifier.send(
        "POST", "/api/auth/signin", json_body={"username": username, "password": password}
    )
    if verify_status != 200:
        snippet = verify_body[:300].decode("utf-8", errors="replace")
        raise RuntimeError(
            f"RESTORE UNVERIFIED: signing in with the original password returned "
            f"{verify_status}. Rotate the smoke account with the 'Provision smoke account' "
            f"workflow: {snippet}"
        )

    # The old session must be gone: change_password keeps only the caller's.
    session.request("GET", "/api/auth/me")
    print("  password change round-tripped on the smoke account (original password restored)")


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
    check_confidence_bands(models)

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

    # A page rather than a single row: the reviewer-directory write check needs
    # spare candidates to walk when the first is already seeded or is
    # guard-excluded.
    queries = client.json("GET", "/api/queries?status=all&page_size=25")
    if not queries.get("items"):
        raise RuntimeError("/api/queries returned no query items")
    candidate_query_ids = [item["file_id"] for item in queries["items"]]
    query_id = candidate_query_ids[0]

    check_reviewer_dir_support(client, models)

    predictions = client.json(
        "GET",
        f"/api/query/{query_id}/predictions?{urlencode({'model': model, 'top_k': 1})}",
    )
    if not predictions.get("predictions"):
        raise RuntimeError("/api/query/{id}/predictions returned no predictions")

    check_predictions_per_variant(client, query_id, model, variants, default_variant)
    check_predictions_carry_reviewer_dirs(client, query_id, model, default_variant)
    check_shared_note_shape(client, query_id, model, default_variant)

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
        # Exactly ONE row is written into the production feedback DB, and it
        # carries every write-path assertion: the DB accepts a write, the CSV
        # export reads it back, the reviewer's reload path prefills notes and
        # a multi-select answer, and a note saved under one variant does not
        # leak into another variant's prefill. Each smoke run leaving three
        # rows behind was three times the reviewer-visible litter for no
        # additional coverage.
        other = next(v for v in variants if v != default_variant)
        check_notes_round_trip(client, query_id, model, default_variant, other)

        # Also permanent, and also deliberately one per run: reviewer
        # directories can never be deleted, so this leaves a single directory
        # behind. Read-only deploys prove the endpoint is reachable and the
        # matrices are present (check_reviewer_dir_support) without creating
        # anything.
        check_reviewer_dir_create_round_trip(client, model, default_variant, candidate_query_ids)

        # Last, because it is the only check that can invalidate the credentials
        # the deploy pipeline itself uses. It restores the original password and
        # verifies the restore before returning.
        check_password_change_round_trip(args.base_url, args.username, args.password)

    print("Reviewer pilot smoke checks passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"SMOKE FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
