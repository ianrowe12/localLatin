"""Exercise the diagnostic CLI against an isolated HTTP service, never production."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODELS = (
    "bowphs_LaTa",
    "bowphs_PhilTa",
    "sentence-transformers_LaBSE",
    "google_mt5-base",
    "KaLM-Embedding_KaLM-embedding-multilingual-mini-instruct-v2.5",
    "Qwen_Qwen3-Embedding-0.6B",
)
FILENAMES = ("BAV1341.16r.7.txt", "BAV1341.22v.9.txt")
QUERY_IDS = (713, 904)
SECRET = "synthetic_private_session_token_not_a_real_session"
PRIVATE = "PRIVATE manuscript notes reviewer label password"


@pytest.fixture(scope="module")
def diagnostic():
    spec = importlib.util.spec_from_file_location(
        "check_reviewer_readiness",
        ROOT / "scripts/webapp/check_reviewer_readiness.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prediction(file_id, model):
    return {
        "file_id": file_id,
        "filename": FILENAMES[QUERY_IDS.index(file_id)],
        "model": model,
        "variant": "sif_abtt",
        "status": "ok",
        "predictions": [
            {
                "rank": rank,
                "dir_name": f"source-{rank}",
                "score": 0.8 - rank * 0.02,
                "source": "model",
                "dir_files": [f"witness-{rank}.txt"],
                "candidate_files": [
                    {"filename": f"witness-{rank}.txt", "text": PRIVATE}
                ],
                "preview_text": PRIVATE,
            }
            for rank in range(1, 11)
        ],
        "seeded_dirs": [{"label": PRIVATE, "created_by": PRIVATE}],
    }


@pytest.fixture()
def service(tmp_path):
    calls = []

    def respond(path, query):
        if path == "/api/auth/me":
            return 200, {
                "id": 27, "role": "reviewer", "approval_status": "approved",
                "must_change_password": False, "username": PRIVATE,
            }, {}
        if path == "/api/models":
            return 200, [
                {"slug": model, "default_variant": "sif_abtt",
                 "available_variants": ["sif_abtt"], "prediction_count": 99999}
                for model in MODELS
            ], {}
        if path == "/api/queries":
            name = query["search"][0]
            return 200, {
                "page": int(query["page"][0]),
                "page_size": int(query["page_size"][0]),
                "total": 1, "has_more": False,
                "items": [{"filename": name, "file_id": QUERY_IDS[FILENAMES.index(name)],
                           "text_preview": PRIVATE}],
            }, {}
        if path.startswith("/api/query/"):
            file_id = int(path.split("/")[3])
            if path.endswith("/predictions"):
                return 200, prediction(file_id, query["model"][0]), {}
            return 200, {
                "file_id": file_id, "filename": FILENAMES[QUERY_IDS.index(file_id)],
                "text": PRIVATE, "tokens": [{"text": PRIVATE}],
            }, {}
        return 404, {"detail": PRIVATE}, {}

    state = {"respond": respond, "calls": calls}

    class Handler(BaseHTTPRequestHandler):
        def handle_request(self):
            url = urlsplit(self.path)
            calls.append((self.command, url.path, parse_qs(url.query),
                          self.headers.get("Cookie")))
            status, data, headers = state["respond"](url.path, parse_qs(url.query))
            body = data if isinstance(data, bytes) else json.dumps(data).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            for name, value in headers.items():
                self.send_header(name, value)
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                # The size-limit test closes before consuming the whole body.
                state["client_disconnects"] = state.get("client_disconnects", 0) + 1

        do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = handle_request

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True,
    )
    thread.start()
    state["url"] = f"http://127.0.0.1:{server.server_port}"
    cookie_file = tmp_path / "session"
    cookie_file.write_text(SECRET + "\n")
    cookie_file.chmod(0o600)
    state["args"] = ["--base-url", state["url"], "--session-file", str(cookie_file)]
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert {call[0] for call in calls} <= {"GET"}
        assert all(
            call[1] in {"/api/auth/me", "/api/models", "/api/queries"}
            or call[1] in {
                route for fid in QUERY_IDS
                for route in (f"/api/query/{fid}", f"/api/query/{fid}/predictions")
            }
            for call in calls
        )


def run(diagnostic, service, capsys, extra=()):
    code = diagnostic.main(service["args"] + list(extra))
    captured = capsys.readouterr()
    assert captured.err == ""
    assert PRIVATE not in captured.out
    assert SECRET not in captured.out
    assert service["url"] not in captured.out
    report = json.loads(captured.out)
    assert set(report) == {
        "schema_version", "scope", "started_at", "finished_at", "status", "code",
        "authentication", "catalog", "queries", "cases", "provenance",
    }
    for check in (report["authentication"], report["catalog"]):
        assert set(check) == {"status", "code", "http_status"}
    for query in report["queries"]:
        assert set(query) == {
            "status", "code", "http_status", "filename", "file_id", "text_available",
        }
        assert query["filename"] in FILENAMES
    for case in report["cases"]:
        assert set(case) == {
            "status", "code", "http_status", "filename", "file_id", "model",
            "variant", "source_status", "model_ranks", "model_candidates_with_text",
            "reviewer_candidate_count",
        }
        assert case["filename"] in FILENAMES
        assert case["model"] in MODELS
    return code, report


def test_all_six_models_and_both_exact_filenames_use_only_expected_gets(
    diagnostic, service, capsys,
):
    code, report = run(diagnostic, service, capsys)
    assert code == 0
    assert report["status"] == "PASS"
    assert len(report["cases"]) == 12
    assert all(case["status"] == "PASS" for case in report["cases"])
    assert all(case["model_ranks"] == list(range(1, 11)) for case in report["cases"])
    assert all(case["model_candidates_with_text"] == 10 for case in report["cases"])
    expected = {"/api/auth/me", "/api/models", "/api/queries"}
    expected.update(f"/api/query/{fid}" for fid in QUERY_IDS)
    expected.update(f"/api/query/{fid}/predictions" for fid in QUERY_IDS)
    assert {call[1] for call in service["calls"]} == expected
    assert {call[0] for call in service["calls"]} == {"GET"}
    assert all(call[3] == f"locallatin_session={SECRET}" for call in service["calls"])
    prediction_calls = [call for call in service["calls"] if call[1].endswith("/predictions")]
    assert len(prediction_calls) == 12
    assert {(call[1], call[2]["model"][0]) for call in prediction_calls} == {
        (f"/api/query/{fid}/predictions", model) for fid in QUERY_IDS for model in MODELS
    }
    assert all(call[2]["variant"] == ["sif_abtt"] for call in prediction_calls)
    assert all(call[2]["top_k"] == ["10"] for call in prediction_calls)
    assert report["provenance"]["observed_service"]["source_sha"] == "unknown"
    assert report["provenance"]["deployment"]["status"] == "NOT_REQUESTED"


def change_response(service, transform):
    original = service["respond"]

    def respond(path, query):
        status, data, headers = original(path, query)
        return transform(path, query, status, data, headers)

    service["respond"] = respond


def change_philta(service, change):
    def transform(path, query, status, data, headers):
        if path.endswith("/predictions") and query["model"] == ["bowphs_PhilTa"]:
            change(data)
        return status, data, headers
    change_response(service, transform)


@pytest.mark.parametrize("missing", MODELS)
def test_missing_expected_model_cannot_shrink_the_check(
    diagnostic, service, capsys, missing,
):
    change_response(service, lambda path, query, status, data, headers: (
        status,
        [item for item in data if item["slug"] != missing] if path == "/api/models" else data,
        headers,
    ))
    code, report = run(diagnostic, service, capsys)
    assert code == 1
    assert report["catalog"]["code"] == "expected_six_models_missing_or_duplicated"
    assert len([call for call in service["calls"] if call[1].endswith("/predictions")]) == 12


@pytest.mark.parametrize("bad_status", [400, 404, 422, 429, 500, 503])
def test_philta_http_failure_is_not_hidden_by_kalm(
    diagnostic, service, capsys, bad_status,
):
    def transform(path, query, status, data, headers):
        if path.endswith("/predictions") and query["model"] == ["bowphs_PhilTa"]:
            return bad_status, {"detail": PRIVATE, "Cookie": SECRET}, {}
        return status, data, headers
    change_response(service, transform)
    code, report = run(diagnostic, service, capsys)
    assert code == 1
    philta = [case for case in report["cases"] if case["model"] == "bowphs_PhilTa"]
    assert all(case["status"] == "FAIL" and case["http_status"] == bad_status for case in philta)
    assert all(case["status"] == "PASS" for case in report["cases"] if case["model"] == MODELS[4])


@pytest.mark.parametrize("field,value", [
    ("file_id", True), ("file_id", 0), ("filename", PRIVATE),
    ("model", PRIVATE), ("variant", "raw"),
])
def test_prediction_identity_must_match_resolved_request(
    diagnostic, service, capsys, field, value,
):
    change_philta(service, lambda data: data.update({field: value}))
    code, report = run(diagnostic, service, capsys)
    assert code == 1
    assert all(
        case["code"] == "prediction_identity_mismatch"
        for case in report["cases"] if case["model"] == "bowphs_PhilTa"
    )


@pytest.mark.parametrize("defect", [
    "empty", "reviewer_only", "nine_ranks", "duplicate_rank", "sparse_rank",
    "boolean_rank", "reversed_ranks", "blank_text", "missing_text", "missing_files",
    "duplicate_file", "wrong_file", "duplicate_directory", "unsorted_scores",
    "unknown_status", "unknown_source", "malformed_prediction",
])
def test_unusable_model_evidence_fails_despite_reviewer_extras(
    diagnostic, service, capsys, defect,
):
    def change(data):
        rows = data["predictions"]
        if defect in ("empty", "reviewer_only"):
            rows.clear()
        elif defect == "nine_ranks":
            rows.pop()
        elif defect == "duplicate_rank":
            rows[1]["rank"] = 1
        elif defect == "sparse_rank":
            rows[-1]["rank"] = 11
        elif defect == "boolean_rank":
            rows[0]["rank"] = True
        elif defect == "reversed_ranks":
            rows.reverse()
        elif defect == "blank_text":
            rows[0]["candidate_files"][0]["text"] = " \n\t"
        elif defect == "missing_text":
            rows[0]["candidate_files"] = None
        elif defect == "missing_files":
            rows[0]["dir_files"] = []
        elif defect == "duplicate_file":
            rows[0]["candidate_files"] *= 2
        elif defect == "wrong_file":
            rows[0]["candidate_files"][0]["filename"] = PRIVATE
        elif defect == "duplicate_directory":
            rows[1]["dir_name"] = rows[0]["dir_name"]
        elif defect == "unsorted_scores":
            rows[-1]["score"] = 0.99
        elif defect == "unknown_status":
            data["status"] = PRIVATE
        elif defect == "unknown_source":
            rows[0]["source"] = PRIVATE
        elif defect == "malformed_prediction":
            rows[0] = PRIVATE
        if defect != "empty":
            rows.append({
                "rank": 11, "source": "reviewer", "label": PRIVATE,
                "dir_name": PRIVATE, "score": 0.9,
                "candidate_files": [{"filename": PRIVATE, "text": PRIVATE}],
            })
    change_philta(service, change)
    code, report = run(diagnostic, service, capsys)
    assert code == 1
    assert all(
        case["status"] == "FAIL"
        for case in report["cases"] if case["model"] == "bowphs_PhilTa"
    )


@pytest.mark.parametrize("score", [None, True, "0.9", float("nan"), float("inf"), 2, -(10**100)])
def test_scores_must_be_finite_numbers(diagnostic, service, capsys, score):
    change_philta(service, lambda data: data["predictions"][0].update(score=score))
    code, report = run(diagnostic, service, capsys)
    assert code == 1
    assert all(
        case["code"] == "invalid_model_score"
        for case in report["cases"] if case["model"] == "bowphs_PhilTa"
    )


@pytest.mark.parametrize("status", [None, "absent"])
@pytest.mark.parametrize("empty", [True, False])
def test_legacy_status_is_usable_only_with_actual_model_evidence(
    diagnostic, service, capsys, status, empty,
):
    def change(data):
        if status is None:
            data["status"] = None
        else:
            del data["status"]
        if empty:
            data["predictions"] = []
        else:
            for row in data["predictions"]:
                del row["source"]
    change_philta(service, change)
    code, report = run(diagnostic, service, capsys)
    assert code == (1 if empty else 0)
    assert all(
        case["code"] == ("unexplained_empty_model_ranking" if empty else "usable_model_ranking")
        for case in report["cases"] if case["model"] == "bowphs_PhilTa"
    )


@pytest.mark.parametrize("status", ["excluded_blank_source", "excluded_zero_norm"])
def test_explicit_exclusion_is_non_evaluable_not_a_request_failure_or_pass(
    diagnostic, service, capsys, status,
):
    change_philta(service, lambda data: data.update(
        status=status, predictions=[{"source": "reviewer", "label": PRIVATE}],
    ))
    code, report = run(diagnostic, service, capsys)
    assert code == 3
    assert report["status"] == "NOT_READY"
    assert report["code"] == "expected_manuscript_excluded"
    assert all(
        case["status"] == "EXCLUDED" and case["source_status"] == status
        for case in report["cases"] if case["model"] == "bowphs_PhilTa"
    )


def test_exclusion_cannot_contradict_model_ranks(diagnostic, service, capsys):
    change_philta(service, lambda data: data.update(status="excluded_blank_source"))
    code, report = run(diagnostic, service, capsys)
    assert code == 1
    assert report["cases"][1]["code"] == "exclusion_with_model_ranks"


@pytest.mark.parametrize("stage", ["auth", "predictions"])
@pytest.mark.parametrize("status", [401, 403])
def test_expired_or_password_gated_auth_is_blocked_and_redacted(
    diagnostic, service, capsys, stage, status,
):
    def transform(path, query, old_status, data, headers):
        if (stage == "auth" and path == "/api/auth/me") or (
            stage == "predictions" and path.endswith("/predictions")
        ):
            return status, {"detail": PRIVATE, "headers": SECRET}, {}
        return old_status, data, headers
    change_response(service, transform)
    code, report = run(diagnostic, service, capsys)
    assert code == 2
    assert report["status"] == "BLOCKED"
    if stage == "auth":
        assert len(service["calls"]) == 1


def test_auth_me_password_gate_blocks_without_attempting_password_change(
    diagnostic, service, capsys,
):
    def transform(path, query, status, data, headers):
        if path == "/api/auth/me":
            data["must_change_password"] = True
        return status, data, headers
    change_response(service, transform)
    assert run(diagnostic, service, capsys)[0] == 2
    assert len(service["calls"]) == 1


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
@pytest.mark.parametrize("destination", ["same_origin", "other_origin"])
def test_redirects_never_forward_the_cookie_or_print_locations(
    diagnostic, service, capsys, status, destination,
):
    target = service["url"] if destination == "same_origin" else "https://do-not-contact.invalid"
    change_response(service, lambda *args: (
        status, PRIVATE.encode(), {"Location": target + "/login?session=" + SECRET},
    ))
    code, report = run(diagnostic, service, capsys)
    assert code == 2
    assert report["authentication"]["code"] == "redirect_refused"
    assert len(service["calls"]) == 1


@pytest.mark.parametrize("body", [
    PRIVATE.encode(), b"\xff" + PRIVATE.encode(), b'{"text": "' + PRIVATE.encode(),
])
def test_invalid_json_never_appears_in_output(diagnostic, service, capsys, body):
    change_response(service, lambda *args: (200, body, {}))
    code, report = run(diagnostic, service, capsys)
    assert code == 1
    assert report["authentication"]["code"] == "invalid_json"


def test_missing_auth_is_blocked_without_any_network(diagnostic, service, capsys):
    service["args"] = ["--base-url", service["url"]]
    code, report = run(diagnostic, service, capsys)
    assert code == 2
    assert report["code"] == "existing_session_not_supplied"
    assert service["calls"] == []


def test_stdin_session_does_not_accept_or_store_replacement_cookie(
    diagnostic, service, capsys, monkeypatch,
):
    service["args"] = ["--base-url", service["url"], "--session-stdin"]
    monkeypatch.setattr(sys, "stdin", io.StringIO(SECRET + "\n"))
    change_response(service, lambda path, query, status, data, headers: (
        status, data, {"Set-Cookie": "locallatin_session=unrequested_new_session"},
    ))
    assert run(diagnostic, service, capsys)[0] == 0
    assert all(call[3] == f"locallatin_session={SECRET}" for call in service["calls"])


@pytest.mark.parametrize("value", [
    "locallatin_session=" + SECRET, SECRET + "\nAuthorization: " + SECRET,
    SECRET + "; other=cookie", "",
])
def test_session_input_rejects_headers_and_multiple_cookies(
    diagnostic, service, capsys, monkeypatch, value,
):
    monkeypatch.setattr(sys, "stdin", io.StringIO(value))
    service["args"] = ["--base-url", service["url"], "--session-stdin"]
    assert run(diagnostic, service, capsys)[0] == 2
    assert service["calls"] == []


@pytest.mark.parametrize("args", [
    ["--cookie", SECRET], ["--password=" + SECRET], ["--base-url"],
    ["--session-file", "\0" + SECRET],
    ["--base-url", "https://host.invalid/?session=" + SECRET],
    ["--base-url", "https://user:" + SECRET + "@host.invalid"],
    ["--base-url", "http://localhost.evil.invalid"],
])
def test_bad_cli_arguments_never_echo_secret_values(
    diagnostic, service, capsys, args,
):
    assert run(diagnostic, service, capsys, args)[0] == 2
    assert service["calls"] == []


@pytest.mark.parametrize("duplicate", [False, True])
def test_exact_filename_resolution_checks_all_pages_for_uniqueness(
    diagnostic, service, capsys, duplicate,
):
    def transform(path, query, status, data, headers):
        if path == "/api/queries":
            name = query["search"][0]
            page = int(query["page"][0])
            if page == 1:
                data["items"] = [
                    {"file_id": 10000 + i, "filename": name + ".extra", "text_preview": PRIVATE}
                    for i in range(200)
                ]
                if duplicate:
                    data["items"][0]["filename"] = name
            data["total"], data["has_more"] = 201, page == 1
        return status, data, headers
    change_response(service, transform)
    code, report = run(diagnostic, service, capsys)
    assert code == (1 if duplicate else 0)
    assert len([call for call in service["calls"] if call[1] == "/api/queries"]) == 4
    if duplicate:
        assert all(query["code"] == "filename_not_unique" for query in report["queries"])
    else:
        assert [query["file_id"] for query in report["queries"]] == list(QUERY_IDS)


def test_substring_match_is_not_the_pinned_filename(diagnostic, service, capsys):
    def transform(path, query, status, data, headers):
        if path == "/api/queries":
            data["items"][0]["filename"] += ".extra"
        return status, data, headers
    change_response(service, transform)
    code, report = run(diagnostic, service, capsys)
    assert code == 1
    assert all(query["code"] == "filename_not_found" for query in report["queries"])


def test_transport_exception_is_redacted(diagnostic, service, capsys, monkeypatch):
    def failed_open(*args, **kwargs):
        raise OSError(PRIVATE + SECRET)
    monkeypatch.setattr(diagnostic, "build_opener", lambda *args: type(
        "BrokenConnection", (), {"open": failed_open},
    )())
    code, report = run(diagnostic, service, capsys)
    assert code == 1
    assert report["authentication"]["code"] == "transport_error"


@pytest.mark.parametrize("defect", ["duplicate", "research_model", "wrong_variant"])
def test_catalog_contract_is_independent_of_discovery(
    diagnostic, service, capsys, defect,
):
    def transform(path, query, status, data, headers):
        if path == "/api/models":
            if defect == "duplicate":
                data[1] = data[0]
            elif defect == "research_model":
                data[3]["slug"] = "Qwen_Qwen3-Embedding-8B"
            else:
                data[0]["default_variant"] = "raw"
        return status, data, headers
    change_response(service, transform)
    code, report = run(diagnostic, service, capsys)
    assert code == 1
    assert report["catalog"]["status"] == "FAIL"


@pytest.mark.parametrize("defect", [
    "wrong_page", "wrong_page_size", "short_page", "false_has_more", "duplicate_id",
])
def test_query_pagination_does_not_accept_partial_or_inconsistent_results(
    diagnostic, service, capsys, defect,
):
    def transform(path, query, status, data, headers):
        if path == "/api/queries":
            if defect == "wrong_page":
                data["page"] = True
            elif defect == "wrong_page_size":
                data["page_size"] = 1
            elif defect == "short_page":
                data["total"] = 2
            elif defect == "false_has_more":
                data["has_more"] = True
            elif defect == "duplicate_id":
                data["items"] *= 2
                data["total"] = 2
        return status, data, headers
    change_response(service, transform)
    code, report = run(diagnostic, service, capsys)
    assert code == 1
    assert all(query["status"] == "FAIL" for query in report["queries"])
    assert not any(call[1].endswith("/predictions") for call in service["calls"])


@pytest.mark.parametrize("defect", ["wrong_identity", "blank_query", "oversized", "invalid_json"])
def test_detail_and_prediction_payload_failures_are_redacted(
    diagnostic, service, capsys, defect,
):
    def transform(path, query, status, data, headers):
        if path == f"/api/query/{QUERY_IDS[0]}":
            if defect == "wrong_identity":
                data["filename"] = PRIVATE
            elif defect == "blank_query":
                data["text"] = " \t"
        if path.endswith("/predictions"):
            if defect == "oversized":
                return 200, PRIVATE.encode() * 200000, {}
            if defect == "invalid_json":
                return 200, PRIVATE.encode(), {}
        return status, data, headers
    change_response(service, transform)
    code, report = run(diagnostic, service, capsys)
    assert code == 1
    assert any(case["status"] == "FAIL" for case in report["cases"])
    if defect == "oversized":
        assert all(case["code"] == "response_too_large" for case in report["cases"])


def test_arbitrary_service_metadata_is_never_reported_as_provenance(
    diagnostic, service, capsys,
):
    def transform(path, query, status, data, headers):
        if isinstance(data, dict):
            data.update(
                source_sha=PRIVATE, data_release=SECRET, headers=SECRET,
                password=SECRET, session_url=service["url"] + "/" + SECRET,
            )
        return status, data, headers
    change_response(service, transform)
    code, report = run(diagnostic, service, capsys)
    assert code == 0
    assert report["provenance"]["observed_service"] == {
        "source_sha": "unknown", "data_release": "unknown", "data_digest": "unknown",
    }


def test_no_optional_github_reads_without_explicit_run(
    diagnostic, service, capsys, monkeypatch,
):
    original_run = diagnostic.subprocess.run

    def no_github(command, **kwargs):
        assert command[0] != "gh"
        return original_run(command, **kwargs)

    monkeypatch.setattr(diagnostic.subprocess, "run", no_github)
    assert run(diagnostic, service, capsys)[0] == 0


@pytest.mark.parametrize("extra", [[], ["--password", SECRET]])
def test_cli_entrypoint_blocks_without_auth_and_never_echoes_secret_argv(service, extra):
    import subprocess

    result = subprocess.run(
        [
            sys.executable, str(ROOT / "scripts/webapp/check_reviewer_readiness.py"),
            "--base-url", service["url"], *extra,
        ],
        capture_output=True, text=True, timeout=10, check=False,
    )
    assert result.returncode == 2
    assert result.stderr == ""
    assert SECRET not in result.stdout
    assert service["url"] not in result.stdout
    assert json.loads(result.stdout)["status"] == "BLOCKED"
    assert service["calls"] == []


@pytest.mark.parametrize("paginated", [False, True])
@pytest.mark.parametrize("job_conclusion,expected", [
    ("success", "COMPLETED"), ("skipped", "SKIPPED"),
    ("failure", "FAILED"), ("cancelled", "FAILED"),
])
def test_deployment_evidence_inspects_the_actual_job_not_workflow_success(
    diagnostic, service, capsys, monkeypatch, job_conclusion, expected, paginated,
):
    calls = []
    original_run = diagnostic.subprocess.run

    def gh_run(command, **kwargs):
        if command[0] != "gh":
            return original_run(command, **kwargs)
        calls.append((command, kwargs))
        route = command[-1]
        if route == "user":
            data = {"login": "ianrowe12"}
        elif "/attempts/1/jobs?" in route:
            deploy = {"name": "Deploy production", "status": "completed",
                      "conclusion": job_conclusion, "head_sha": "a" * 40}
            other = {"name": "Test", "status": "completed", "conclusion": "success"}
            if paginated:
                data = {"total_count": 101, "jobs": (
                    [other] * 100 if route.endswith("page=1") else [deploy]
                )}
            else:
                data = {"total_count": 2, "jobs": [other, deploy]}
        elif route.endswith("/actions/runs/42"):
            data = {
                "id": 42, "run_attempt": 1, "head_sha": "a" * 40,
                "status": "completed", "conclusion": "success",
            }
        elif route.endswith("/actions/variables/DATA_RELEASE_TAG"):
            data = {"value": "data-20260907"}
        else:
            pytest.fail("Unexpected gh route")
        return diagnostic.subprocess.CompletedProcess(command, 0, json.dumps(data), "")

    monkeypatch.setenv("GH_TOKEN", SECRET)
    monkeypatch.setenv("GITHUB_TOKEN", SECRET)
    monkeypatch.setattr(diagnostic.subprocess, "run", gh_run)
    code, report = run(diagnostic, service, capsys, ["--deployment-run", "42"])
    assert code == 0  # Retrieval readiness and deployment evidence are separate.
    deployment = report["provenance"]["deployment"]
    assert deployment["status"] == expected
    assert deployment["workflow_head_sha"] == "a" * 40
    assert report["provenance"]["observed_service"]["source_sha"] == "unknown"
    assert report["provenance"]["configured_release"] == "data-20260907"
    assert len(calls) == (5 if paginated else 4)
    assert all(command[1:6] == ["api", "--hostname", "github.com", "--method", "GET"] for command, _ in calls)
    assert all("GH_TOKEN" not in options["env"] and "GITHUB_TOKEN" not in options["env"] for _, options in calls)


@pytest.mark.parametrize("defect", [
    "wrong_account", "error_body", "invalid_json", "exception", "missing_job",
    "duplicate_job", "wrong_head", "pending", "private_release",
])
def test_unavailable_or_inconsistent_deployment_evidence_cannot_claim_running_bytes(
    diagnostic, service, capsys, monkeypatch, defect,
):
    original_run = diagnostic.subprocess.run
    calls = []

    def gh_run(command, **kwargs):
        if command[0] != "gh":
            return original_run(command, **kwargs)
        calls.append(command)
        if defect == "exception":
            raise diagnostic.subprocess.TimeoutExpired(command + [SECRET], 1, output=PRIVATE)
        if defect in ("error_body", "invalid_json"):
            return diagnostic.subprocess.CompletedProcess(
                command, 1 if defect == "error_body" else 0, PRIVATE + SECRET, PRIVATE + SECRET,
            )
        route = command[-1]
        if route == "user":
            data = {"login": "another_account" if defect == "wrong_account" else "ianrowe12"}
        elif "/attempts/" in route:
            jobs = [{
                "name": "Deploy production", "status": "completed", "conclusion": "success",
                "head_sha": "a" * 40,
            }]
            if defect == "missing_job":
                jobs = []
            elif defect == "duplicate_job":
                jobs *= 2
            elif defect == "wrong_head":
                jobs[0]["head_sha"] = "b" * 40
            elif defect == "pending":
                jobs[0]["status"] = "in_progress"
            data = {"jobs": jobs, "total_count": len(jobs)}
        elif "/variables/" in route:
            data = {"value": PRIVATE}
        else:
            data = {"id": 42, "run_attempt": 1, "head_sha": "a" * 40}
        return diagnostic.subprocess.CompletedProcess(command, 0, json.dumps(data), "")

    monkeypatch.setattr(diagnostic.subprocess, "run", gh_run)
    code, report = run(diagnostic, service, capsys, ["--deployment-run", "42"])
    assert code == 0
    evidence = report["provenance"]["deployment"]
    if defect in ("wrong_account", "error_body", "invalid_json", "exception"):
        assert evidence["status"] == "BLOCKED"
        assert len(calls) == 1
    elif defect == "pending":
        assert evidence["status"] == "PENDING"
    elif defect == "private_release":
        assert evidence["status"] == "COMPLETED"
    else:
        assert evidence["status"] == "UNKNOWN"
    assert report["provenance"]["configured_release"] == "unknown"
    assert report["provenance"]["observed_service"]["source_sha"] == "unknown"
