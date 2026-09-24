#!/usr/bin/env python3
"""GET-only reviewer readiness check using an explicitly supplied existing session.

No application imports, sign-in, cookie persistence, or reviewer-record writes.
Authenticated GETs can update the existing session's last_seen_at on the server.
"""

from __future__ import annotations

import argparse
import http.client
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from email.message import Message
from pathlib import Path
from typing import BinaryIO, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


# Deliberately independent of /api/models and the different research model list.
MODELS = (
    "bowphs_LaTa",
    "bowphs_PhilTa",
    "sentence-transformers_LaBSE",
    "google_mt5-base",
    "KaLM-Embedding_KaLM-embedding-multilingual-mini-instruct-v2.5",
    "Qwen_Qwen3-Embedding-0.6B",
)
# The ordinary case is documented in docs/research/abigail_demo_script.md.
FILENAMES = ("BAV1341.16r.7.txt", "BAV1341.22v.9.txt")
VARIANT = "sif_abtt"
EXCLUSIONS = ("excluded_blank_source", "excluded_zero_norm")
PAGE_SIZE = 200
MAX_PAGES = 50
MAX_BODY = 8 * 1024 * 1024
REPOSITORY = "ianrowe12/localLatin"
State = Literal["PASS", "FAIL", "BLOCKED", "EXCLUDED", "NOT_READY"]


@dataclass
class Check:
    status: State = "BLOCKED"
    code: str = "not_checked"
    http_status: int | None = None


@dataclass
class QueryCheck(Check):
    filename: str = ""
    file_id: int | None = None
    text_available: bool = False


@dataclass
class Case(Check):
    filename: str = ""
    model: str = ""
    variant: str = VARIANT
    file_id: int | None = None
    source_status: str | None = None
    model_ranks: list[int] = field(default_factory=list)
    model_candidates_with_text: int = 0
    reviewer_candidate_count: int = 0


class Problem(Exception):
    """Only fixed diagnostic codes, never response or exception text."""

    def __init__(
        self, code: str, status: State = "FAIL", http_status: int | None = None,
    ) -> None:
        super().__init__(code)
        self.check = Check(status, code, http_status)


def apply_problem(check: Check, problem: Problem) -> None:
    check.status = problem.check.status
    check.code = problem.check.code
    check.http_status = problem.check.http_status


def integer(value: object) -> bool:
    return type(value) is int and 0 <= value <= 2**31 - 1


def nonblank(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(
        self, req: Request, fp: BinaryIO, code: int, msg: str,
        headers: Message, newurl: str,
    ) -> None:
        return None


def validate_base_url(base_url: str) -> str:
    try:
        parsed = urlsplit(base_url)
        valid = (
            parsed.scheme in ("https", "http")
            and parsed.hostname
            and parsed.username is None
            and parsed.password is None
            and parsed.port != 0
            and not any(char in base_url for char in ("?", "#", "\\"))
            and not any(ord(char) <= 32 or ord(char) >= 127 for char in base_url)
            and re.fullmatch(r"[/A-Za-z0-9_-]*", parsed.path)
            and (
                parsed.scheme == "https"
                or parsed.hostname in ("127.0.0.1", "localhost", "::1")
            )
        )
    except ValueError:
        valid = False
    if not valid:
        raise Problem("invalid_base_url", "BLOCKED")
    return base_url.rstrip("/")


class ReadClient:
    def __init__(self, base_url: str, token: str) -> None:
        self.base_url = validate_base_url(base_url)
        self.cookie = f"locallatin_session={token}"
        # Do not inherit proxies or persist/accept replacement cookies.
        self.opener = build_opener(ProxyHandler({}), NoRedirect())

    def get(self, path: str, **params: str | int) -> object:
        if path not in ("/api/auth/me", "/api/models", "/api/queries") and not re.fullmatch(
            r"/api/query/[0-9]+(?:/predictions)?", path,
        ):
            raise Problem("route_not_allowed")
        url = self.base_url + path
        if params:
            url += "?" + urlencode(params)
        request = Request(
            url, method="GET",
            headers={"Accept": "application/json", "Cookie": self.cookie},
        )
        try:
            with self.opener.open(request, timeout=15) as response:
                if response.status != 200:
                    raise Problem("unexpected_http_status", http_status=response.status)
                raw = response.read(MAX_BODY + 1)
        except HTTPError as exc:
            status = exc.code
            exc.close()
            if status in (401, 403):
                raise Problem("authentication_required", "BLOCKED", status) from None
            if 300 <= status < 400:
                raise Problem("redirect_refused", "BLOCKED", status) from None
            raise Problem("http_error", http_status=status) from None
        except (URLError, OSError, http.client.HTTPException, ValueError):
            raise Problem("transport_error") from None
        if len(raw) > MAX_BODY:
            raise Problem("response_too_large")
        try:
            return json.loads(raw)
        except (ValueError, UnicodeError, RecursionError):
            raise Problem("invalid_json") from None


def check_auth(client: ReadClient) -> Check:
    data = client.get("/api/auth/me")
    if (
        not isinstance(data, dict)
        or not integer(data.get("id"))
        or data.get("role") not in ("reviewer", "pi_admin")
        or data.get("approval_status") != "approved"
        or data.get("must_change_password") is not False
    ):
        raise Problem("authorized_session_not_confirmed", "BLOCKED")
    return Check("PASS", "existing_session_authorized", 200)


def check_catalog(client: ReadClient) -> Check:
    data = client.get("/api/models")
    if not isinstance(data, list) or any(not isinstance(item, dict) for item in data):
        raise Problem("invalid_model_catalog")
    slugs = [item.get("slug") for item in data]
    if len(slugs) != len(MODELS) or any(slugs.count(model) != 1 for model in MODELS):
        raise Problem("expected_six_models_missing_or_duplicated")
    for item in data:
        variants = item.get("available_variants")
        if (
            item.get("default_variant") != VARIANT
            or not isinstance(variants, list)
            or VARIANT not in variants
        ):
            raise Problem("default_pipeline_unavailable")
    return Check("PASS", "expected_six_models_present", 200)


def resolve_query(client: ReadClient, query: QueryCheck) -> None:
    matches: list[int] = []
    seen: set[int] = set()
    total: int | None = None
    for page in range(1, MAX_PAGES + 1):
        data = client.get(
            "/api/queries", search=query.filename, page=page,
            page_size=PAGE_SIZE, status="all", sort="file_id",
        )
        if (
            not isinstance(data, dict)
            or type(data.get("page")) is not int or data["page"] != page
            or type(data.get("page_size")) is not int or data["page_size"] != PAGE_SIZE
            or not integer(data.get("total"))
            or not isinstance(data.get("items"), list)
            or type(data.get("has_more")) is not bool
        ):
            raise Problem("invalid_query_page")
        if total is None:
            total = data["total"]
        if total != data["total"] or total > MAX_PAGES * PAGE_SIZE:
            raise Problem("unstable_or_excessive_query_pages")
        expected_count = min(PAGE_SIZE, max(0, total - (page - 1) * PAGE_SIZE))
        if (
            len(data["items"]) != expected_count
            or data["has_more"] != (page * PAGE_SIZE < total)
        ):
            raise Problem("incomplete_query_page")
        for item in data["items"]:
            if (
                not isinstance(item, dict)
                or not integer(item.get("file_id"))
                or not isinstance(item.get("filename"), str)
                or item["file_id"] in seen
            ):
                raise Problem("invalid_or_duplicate_query_identity")
            seen.add(item["file_id"])
            if item["filename"] == query.filename:
                matches.append(item["file_id"])
        if not data["has_more"]:
            break
    if len(matches) != 1:
        raise Problem("filename_not_unique" if matches else "filename_not_found")
    query.file_id = matches[0]
    detail = client.get(f"/api/query/{query.file_id}")
    if (
        not isinstance(detail, dict)
        or not integer(detail.get("file_id"))
        or detail["file_id"] != query.file_id
        or detail.get("filename") != query.filename
        or not isinstance(detail.get("text"), str)
    ):
        raise Problem("query_identity_or_text_invalid")
    query.text_available = nonblank(detail["text"])
    query.status, query.code, query.http_status = "PASS", "exact_filename_resolved", 200


def candidate_text_available(prediction: dict) -> bool:
    files = prediction.get("dir_files")
    candidates = prediction.get("candidate_files")
    if (
        not isinstance(files, list) or not files
        or not all(nonblank(name) for name in files)
        or len(set(files)) != len(files)
        or not isinstance(candidates, list) or len(candidates) != len(files)
    ):
        return False
    names = []
    for candidate in candidates:
        if (
            not isinstance(candidate, dict)
            or not nonblank(candidate.get("filename"))
            or not nonblank(candidate.get("text"))
        ):
            return False
        names.append(candidate["filename"])
    return len(set(names)) == len(names) and set(names) == set(files)


def check_prediction(client: ReadClient, query: QueryCheck, case: Case) -> None:
    data = client.get(
        f"/api/query/{case.file_id}/predictions",
        model=case.model, variant=VARIANT, top_k=10,
    )
    case.http_status = 200
    if (
        not isinstance(data, dict)
        or not integer(data.get("file_id")) or data["file_id"] != case.file_id
        or data.get("filename") != case.filename
        or data.get("model") != case.model or data.get("variant") != VARIANT
    ):
        raise Problem("prediction_identity_mismatch", http_status=200)
    status = data.get("status")
    if status not in (None, "ok", *EXCLUSIONS):
        raise Problem("unknown_source_status", http_status=200)
    case.source_status = status
    predictions = data.get("predictions")
    if not isinstance(predictions, list):
        raise Problem("invalid_predictions", http_status=200)
    model_predictions = []
    for prediction in predictions:
        if not isinstance(prediction, dict):
            raise Problem("invalid_predictions", http_status=200)
        source = prediction.get("source", "model")
        if source == "model":
            model_predictions.append(prediction)
        elif source == "reviewer":
            case.reviewer_candidate_count += 1
        else:
            raise Problem("invalid_candidate_source", http_status=200)
    if status in EXCLUSIONS:
        if model_predictions:
            raise Problem("exclusion_with_model_ranks", http_status=200)
        case.status, case.code = "EXCLUDED", status
        return
    if not model_predictions:
        raise Problem("unexplained_empty_model_ranking", http_status=200)
    ranks = [item.get("rank") for item in model_predictions]
    if any(type(rank) is not int for rank in ranks) or ranks != list(range(1, 11)):
        raise Problem("expected_model_ranks_1_to_10", http_status=200)
    case.model_ranks = ranks
    directories = [item.get("dir_name") for item in model_predictions]
    if not all(nonblank(name) for name in directories) or len(set(directories)) != 10:
        raise Problem("invalid_model_directories", http_status=200)
    scores = [item.get("score") for item in model_predictions]
    if any(
        type(score) not in (int, float)
        or not -1.000001 <= score <= 1.000001 or not math.isfinite(score)
        for score in scores
    ):
        raise Problem("invalid_model_score", http_status=200)
    if any(left < right for left, right in zip(scores, scores[1:])):
        raise Problem("model_scores_not_ranked", http_status=200)
    case.model_candidates_with_text = sum(
        candidate_text_available(item) for item in model_predictions
    )
    if case.model_candidates_with_text != 10:
        raise Problem("missing_model_candidate_text", http_status=200)
    if not query.text_available:
        raise Problem("blank_query_without_exclusion", http_status=200)
    case.status, case.code = "PASS", "usable_model_ranking"


def local_head() -> str:
    try:
        result = subprocess.run(
            ["git", "--no-pager", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    value = result.stdout.strip()
    return value if result.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", value) else "unknown"


def gh_get(route: str) -> object:
    environment = os.environ.copy()
    for name in ("GH_TOKEN", "GITHUB_TOKEN"):
        environment.pop(name, None)
    environment["GH_PROMPT_DISABLED"] = "1"
    try:
        result = subprocess.run(
            ["gh", "api", "--hostname", "github.com", "--method", "GET", route],
            env=environment, capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, UnicodeError, subprocess.TimeoutExpired):
        raise Problem("github_read_unavailable", "BLOCKED") from None
    if result.returncode != 0:
        raise Problem("github_read_unavailable", "BLOCKED")
    try:
        return json.loads(result.stdout)
    except (ValueError, RecursionError):
        raise Problem("invalid_github_response", "BLOCKED") from None


def deployment_evidence(run_id: int, provenance: dict) -> None:
    """Record one requested run attempt, not a claim about running bytes."""
    evidence = {"status": "UNKNOWN", "run_id": run_id, "workflow_head_sha": "unknown"}
    provenance["deployment"] = evidence
    provenance["configured_release"] = "unknown"
    try:
        user = gh_get("user")
        if not isinstance(user, dict) or user.get("login") != "ianrowe12":
            raise Problem("github_identity_not_confirmed", "BLOCKED")
        run = gh_get(f"repos/{REPOSITORY}/actions/runs/{run_id}")
        if (
            not isinstance(run, dict)
            or type(run.get("id")) is not int or run["id"] != run_id
            or not integer(run.get("run_attempt")) or run["run_attempt"] < 1
            or not isinstance(run.get("head_sha"), str)
            or not re.fullmatch(r"[0-9a-f]{40}", run["head_sha"])
        ):
            raise Problem("invalid_workflow_identity", "BLOCKED")
        evidence["workflow_head_sha"] = run["head_sha"]
        evidence["run_attempt"] = run["run_attempt"]
        jobs = []
        total = None
        for page in range(1, MAX_PAGES + 1):
            data = gh_get(
                f"repos/{REPOSITORY}/actions/runs/{run_id}/attempts/"
                f"{run['run_attempt']}/jobs?per_page=100&page={page}"
            )
            if (
                not isinstance(data, dict) or not integer(data.get("total_count"))
                or not isinstance(data.get("jobs"), list)
                or any(not isinstance(job, dict) for job in data["jobs"])
            ):
                raise Problem("invalid_deployment_jobs", "BLOCKED")
            if total is None:
                total = data["total_count"]
            if (
                total != data["total_count"] or total > MAX_PAGES * 100
                or len(data["jobs"]) != min(100, max(0, total - (page - 1) * 100))
            ):
                raise Problem("incomplete_deployment_jobs", "BLOCKED")
            jobs.extend(job for job in data["jobs"] if job.get("name") == "Deploy production")
            if page * 100 >= total:
                break
        if len(jobs) != 1:
            evidence["code"] = "deployment_job_not_unique_or_absent"
        else:
            job = jobs[0]
            if job.get("head_sha") != run["head_sha"]:
                evidence["code"] = "deployment_head_mismatch"
            elif job.get("status") != "completed":
                evidence["status"] = "PENDING"
            else:
                evidence["status"] = {
                    "success": "COMPLETED", "skipped": "SKIPPED",
                    "failure": "FAILED", "cancelled": "FAILED",
                    "timed_out": "FAILED", "action_required": "FAILED",
                    "startup_failure": "FAILED",
                }.get(job.get("conclusion") if isinstance(job.get("conclusion"), str) else "", "UNKNOWN")
        try:
            configured = gh_get(f"repos/{REPOSITORY}/actions/variables/DATA_RELEASE_TAG")
        except Problem as problem:
            evidence["configured_release_code"] = problem.check.code
        else:
            value = configured.get("value") if isinstance(configured, dict) else None
            # Do not echo arbitrary repository-variable contents into reports.
            if isinstance(value, str) and re.fullmatch(r"data-[0-9]{8}", value):
                provenance["configured_release"] = value
    except Problem as problem:
        evidence["status"], evidence["code"] = "BLOCKED", problem.check.code


def initial_report() -> dict:
    return {
        "schema_version": 1,
        "scope": "existing_session_get_only_pinned_retrieval",
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "BLOCKED",
        "code": "not_checked",
        "authentication": asdict(Check()),
        "catalog": asdict(Check()),
        "queries": [asdict(QueryCheck(filename=name)) for name in FILENAMES],
        "cases": [
            asdict(Case(filename=name, model=model))
            for name in FILENAMES for model in MODELS
        ],
        "provenance": {
            "observed_service": {
                "source_sha": "unknown", "data_release": "unknown", "data_digest": "unknown",
            },
            "local_diagnostic_head": local_head(),
            "local_target_disk": "not_inspected",
            "supplied_labels": "not_collected",
            "configured_release": "not_requested",
            "deployment": {"status": "NOT_REQUESTED"},
        },
    }


def diagnose(client: ReadClient, report: dict) -> None:
    queries = [QueryCheck(filename=name) for name in FILENAMES]
    cases = [Case(filename=name, model=model) for name in FILENAMES for model in MODELS]
    try:
        auth = check_auth(client)
    except Problem as problem:
        auth = problem.check
    report["authentication"] = asdict(auth)
    if auth.status != "PASS":
        for check in [*queries, *cases]:
            check.code = "authorized_session_required"
        report["status"], report["code"] = auth.status, auth.code
    else:
        try:
            catalog = check_catalog(client)
        except Problem as problem:
            catalog = problem.check
        report["catalog"] = asdict(catalog)
        for query in queries:
            try:
                resolve_query(client, query)
            except Problem as problem:
                apply_problem(query, problem)
            for case in (item for item in cases if item.filename == query.filename):
                case.file_id = query.file_id
                if query.status != "PASS":
                    case.status, case.code = query.status, query.code
                    case.http_status = query.http_status
                    continue
                try:
                    check_prediction(client, query, case)
                except Problem as problem:
                    apply_problem(case, problem)
        states = {check.status for check in [catalog, *queries, *cases]}
        report["status"] = (
            "FAIL" if "FAIL" in states else "BLOCKED" if "BLOCKED" in states
            else "NOT_READY" if "EXCLUDED" in states else "PASS"
        )
        report["code"] = {
            "PASS": "all_pinned_cases_usable",
            "FAIL": "readiness_failure",
            "BLOCKED": "access_blocked",
            "NOT_READY": "expected_manuscript_excluded",
        }[report["status"]]
    report["queries"] = [asdict(query) for query in queries]
    report["cases"] = [asdict(case) for case in cases]


class SafeParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        # argparse's default error includes unrecognized secret-bearing argv.
        raise Problem("invalid_arguments", "BLOCKED")


def read_session(args: argparse.Namespace) -> str:
    if not args.session_file and not args.session_stdin:
        raise Problem("existing_session_not_supplied", "BLOCKED")
    try:
        if args.session_file:
            with Path(args.session_file).open(encoding="ascii") as handle:
                token = handle.read(1025)
        else:
            token = sys.stdin.read(1025)
    except (OSError, UnicodeError, ValueError):
        raise Problem("session_input_unreadable", "BLOCKED") from None
    if len(token) > 1024:
        raise Problem("invalid_session_input", "BLOCKED")
    token = token.rstrip("\r\n")
    if not re.fullmatch(r"[A-Za-z0-9_-]{16,512}", token):
        raise Problem("invalid_session_input", "BLOCKED")
    return token


def main(argv: list[str] | None = None) -> int:
    report = initial_report()
    parser = SafeParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--base-url", required=True, help="Explicit trusted service origin and optional base path")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--session-file", help="Private file containing only an existing session token, not a header")
    source.add_argument("--session-stdin", action="store_true", help="Read the existing session token from stdin")
    parser.add_argument(
        "--deployment-run", type=int,
        help="Optional GitHub run ID; inspect its Deploy production job through saved gh login",
    )
    try:
        args = parser.parse_args(argv)
        if args.deployment_run is not None and not 1 <= args.deployment_run <= 10**15:
            raise Problem("invalid_arguments", "BLOCKED")
        token = read_session(args)
        diagnose(ReadClient(args.base_url, token), report)
        if args.deployment_run is not None:
            deployment_evidence(args.deployment_run, report["provenance"])
    except Problem as problem:
        report["status"], report["code"] = problem.check.status, problem.check.code
    report["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return {"PASS": 0, "FAIL": 1, "BLOCKED": 2, "NOT_READY": 3}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
