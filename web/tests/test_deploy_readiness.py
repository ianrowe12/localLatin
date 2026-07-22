from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from web.app import create_app
from web.config import load_settings


ROOT = Path(__file__).resolve().parents[2]


def test_production_config_uses_explicit_data_root_and_persistent_feedback_db() -> None:
    settings = load_settings(ROOT / "web" / "config.production.yaml")

    assert settings.app.host == "127.0.0.1"
    assert settings.app.port == 8080
    assert settings.paths.data_root == "/homes/ipro222/localLatin"
    assert settings.paths.feedback_db == "data/feedback.db"
    assert settings.cors.allow_origins == ["https://ai.csr.uky.edu"]
    assert settings.auth.secure_cookies is True


def test_systemd_unit_uses_single_worker_and_explicit_environment() -> None:
    unit = (ROOT / "deploy" / "locallatin.service").read_text(encoding="utf-8")

    assert "WorkingDirectory=/homes/ipro222/localLatin" in unit
    assert "Environment=LOCALLATIN_CONFIG=/homes/ipro222/localLatin/web/config.production.yaml" in unit
    assert "Environment=LOCALLATIN_DATA_ROOT=/homes/ipro222/localLatin" in unit
    assert ".venv/bin/uvicorn" in unit
    assert "--port 8080" in unit
    assert "--workers 1" in unit
    assert "--workers 2" not in unit


def test_deploy_script_installs_dependencies_and_fails_health_checks() -> None:
    deploy = (ROOT / "deploy" / "deploy.sh").read_text(encoding="utf-8")

    assert '-m venv "${VENV_DIR}"' in deploy
    assert 'bin/python" -m pip install -r' in deploy
    assert "DEPLOY_PATH:-/homes/ipro222/localLatin" in deploy
    assert "PUBLIC_BASE_PATH:-/}" in deploy
    assert 'VITE_BASE_PATH="${PUBLIC_BASE_PATH}" npm run build' in deploy
    assert "http://127.0.0.1:8080" in deploy
    assert "/api/models" in deploy
    assert "/api/auth/signin" in deploy
    assert '[[ "${auth_status}" == "401" ]]' in deploy
    assert "API did not become healthy" in deploy
    assert "exit 1" in deploy
    assert "smoke_reviewer_pilot.py" in deploy
    assert "npm ci --prefer-offline --include=dev" in deploy


def test_smoke_script_is_valid_python() -> None:
    script = ROOT / "scripts" / "webapp" / "smoke_reviewer_pilot.py"

    compile(script.read_text(encoding="utf-8"), str(script), "exec")


def test_feedback_backup_script_is_valid_and_change_aware() -> None:
    script = ROOT / "scripts" / "webapp" / "backup_feedback_db.py"
    source = script.read_text(encoding="utf-8")

    compile(source, str(script), "exec")
    assert "sqlite3.connect" in source
    assert ".backup(" in source
    assert "PRAGMA integrity_check" in source
    assert "feedback_fingerprint" in source
    assert "LOCALLATIN_BACKUP_HOST" in source
    assert "LOCALLATIN_LOCAL_DB" in source
    assert "locallatin-feedback" in source


def test_deployment_ultraplan_covers_hosting_actions_and_tdd() -> None:
    plan = (ROOT / "deploy" / "IMPLEMENTATION_ULTRAPLAN.md").read_text(
        encoding="utf-8"
    )

    assert "Backend Hosting" in plan
    assert "Frontend Hosting" in plan
    assert "GitHub Actions Deployment" in plan
    assert "Test Driven Development Plan" in plan
    assert "python -m pytest web/tests" in plan
    assert "npm run build" in plan
    assert "deploy/deploy.sh" in plan


def test_github_actions_workflow_tests_then_deploys_after_main_push() -> None:
    workflow = (ROOT / ".github" / "workflows" / "deploy.yml").read_text(
        encoding="utf-8"
    )

    assert "push:" in workflow
    assert "main" in workflow
    assert "python -m pytest web/tests" in workflow
    assert "npm run build" in workflow
    assert "needs: test" in workflow
    assert "self-hosted" in workflow
    assert "locallatin" in workflow
    assert "production" in workflow
    assert "ENABLE_PRODUCTION_DEPLOY" in workflow
    assert "DEPLOY_PATH" in workflow
    assert "PUBLIC_BASE_PATH" in workflow
    assert "git pull --ff-only origin main" in workflow
    assert 'PUBLIC_BASE_PATH="${PUBLIC_BASE_PATH}" bash deploy/deploy.sh' in workflow


def test_nginx_template_scopes_locallatin_to_root_domain() -> None:
    nginx = (ROOT / "deploy" / "nginx.conf").read_text(encoding="utf-8")

    assert "location / {" in nginx
    assert "proxy_pass http://127.0.0.1:8080" in nginx
    assert "X-Forwarded-Proto $scheme" in nginx
    assert "locallatin/" not in nginx
    assert "X-Forwarded-Prefix" not in nginx


def test_static_frontend_rewrites_spa_routes_without_masking_missing_assets() -> None:
    client = TestClient(create_app())
    index = (ROOT / "web" / "static" / "index.html").read_text(encoding="utf-8")

    root = client.get("/")
    assert root.status_code == 200
    assert "LocalLatin" in root.text

    review = client.get("/review")
    assert review.status_code == 200
    assert review.text == index

    asset = next((ROOT / "web" / "static" / "assets").glob("index-*.js"))
    asset_response = client.get(f"/assets/{asset.name}")
    assert asset_response.status_code == 200
    assert "javascript" in asset_response.headers["content-type"]
    assert asset_response.content == asset.read_bytes()

    asset_head = client.head(
        f"/assets/{asset.name}", headers={"accept-encoding": "identity"}
    )
    assert asset_head.status_code == 200
    assert asset_head.headers["content-length"] == str(asset.stat().st_size)

    gzipped_asset = client.get(
        f"/assets/{asset.name}", headers={"accept-encoding": "gzip"}
    )
    assert gzipped_asset.status_code == 200
    assert gzipped_asset.headers["content-encoding"] == "gzip"
    assert int(gzipped_asset.headers["content-length"]) < asset.stat().st_size
    assert gzipped_asset.content == asset.read_bytes()

    missing_asset = client.get("/assets/missing-bundle.js")
    assert missing_asset.status_code == 404
