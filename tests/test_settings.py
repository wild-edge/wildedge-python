from __future__ import annotations

from wildedge import constants
from wildedge.settings import (
    RUN_DSN_ENV,
    RUN_FLUSH_TIMEOUT_ENV,
    RUN_INTEGRATIONS_ENV,
    parse_bool,
    read_client_env,
    read_runtime_env,
    resolve_app_identity,
)


def test_parse_bool_variants():
    assert parse_bool("1") is True
    assert parse_bool("true") is True
    assert parse_bool("yes") is True
    assert parse_bool("on") is True
    assert parse_bool("0") is False
    assert parse_bool(None) is False


def test_read_client_env_prefers_explicit_values():
    env = {
        constants.ENV_DSN: "https://a@ingest.wildedge.dev/proj",
        constants.ENV_DEBUG: "1",
        constants.ENV_APP_IDENTITY: "env-app",
    }
    s = read_client_env(
        dsn="https://b@ingest.wildedge.dev/proj2",
        debug=False,
        app_identity="explicit-app",
        environ=env,
    )
    assert s.dsn == "https://b@ingest.wildedge.dev/proj2"
    assert s.debug is False
    assert s.app_identity == "explicit-app"


def test_read_runtime_env_uses_run_over_base_dsn():
    env = {
        constants.ENV_DSN: "https://base@ingest.wildedge.dev/base",
        RUN_DSN_ENV: "https://run@ingest.wildedge.dev/run",
        RUN_FLUSH_TIMEOUT_ENV: "7.5",
        RUN_INTEGRATIONS_ENV: "onnx,timm",
    }
    s = read_runtime_env(all_integrations=["onnx", "timm"], environ=env)
    assert s.dsn == "https://run@ingest.wildedge.dev/run"
    assert s.flush_timeout == 7.5
    assert s.integrations == ["onnx", "timm"]


def test_resolve_app_identity_fallbacks_to_project_key():
    assert resolve_app_identity(explicit=None, project_key="proj", environ={}) == "proj"
