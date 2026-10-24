"""Import isolation must hold in a fresh interpreter, regardless of installed ML packages."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from wildedge.hubs.base import BaseHubTracker
from wildedge.hubs.registry import HUB_SPECS
from wildedge.integrations.base import BaseExtractor
from wildedge.integrations.registry import EXTRACTOR_SPECS, INTEGRATION_SPECS
from wildedge.loading import resolve

ROOT = Path(__file__).resolve().parents[1]
OPTIONAL = (
    "torch",
    "timm",
    "tensorflow",
    "keras",
    "mlx",
    "mlx_lm",
    "mlx_whisper",
    "faster_whisper",
    "ctranslate2",
    "onnxruntime",
    "ultralytics",
    "transformers",
    "llama_cpp",
    "openai",
    "anthropic",
    "huggingface_hub",
    "numpy",
)


def isolated(tmp_path, code, *, working=(), sources=None):
    """Make every optional package discoverable but broken unless explicitly supplied."""
    for name in OPTIONAL:
        path = tmp_path / name
        path.mkdir()
        (path / "__init__.py").write_text(
            (sources or {}).get(
                name,
                ""
                if name in working
                else f'raise RuntimeError("unexpected import: {name}")\n',
            )
        )
    env = dict(os.environ)
    env.pop("WILDEDGE_AUTOLOAD", None)
    env.pop("WILDEDGE_DSN", None)
    bootstrap = f"import sys; sys.path[:0] = [{str(tmp_path)!r}, {str(ROOT)!r}]\n"
    result = subprocess.run(
        [sys.executable, "-c", bootstrap + code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_import_and_unconfigured_init_ignore_broken_installed_frameworks(tmp_path):
    isolated(
        tmp_path,
        f"""
import importlib.util
# These are installed/discoverable, rather than merely absent.
for name in {OPTIONAL!r}:
    assert importlib.util.find_spec(name) is not None
import wildedge
import wildedge.cli
from wildedge.hubs import BaseHubTracker
client = wildedge.init(dsn="", integrations=["openai"])
assert client.noop
client.register_model(object(), model_id="manual-model", auto_instrument=False)
assert not ({set(OPTIONAL)!r} & set(sys.modules))
assert not any(name.startswith("wildedge.integrations.") and name.rsplit(".", 1)[-1] not in ("base", "registry", "common") for name in sys.modules)
client.close()
""",
    )


def test_selected_integration_does_not_import_siblings(tmp_path):
    isolated(
        tmp_path,
        """
from unittest.mock import patch
import wildedge
from wildedge.platforms.device_info import DeviceInfo
with patch("wildedge.client.Consumer"), patch("wildedge.client.Transmitter"), patch("wildedge.paths.default_sdk_state_dir", return_value=__import__("pathlib").Path("state")):
    client = wildedge.WildEdge(dsn="https://secret@example.invalid/project", device=DeviceInfo("test", "linux"), sampling_interval_s=None)
    client.instrument("openai")
    assert "openai" in sys.modules
    assert "wildedge.integrations.openai" in sys.modules
    assert not ({"torch", "timm", "mlx", "tensorflow", "huggingface_hub", "anthropic"} & set(sys.modules))
    client.close()
""",
        sources={"openai": "class OpenAI: pass\nclass AsyncOpenAI: pass\n"},
    )


@pytest.mark.parametrize(
    ("hub", "package"), [("huggingface", "huggingface_hub"), ("torchhub", "torch")]
)
def test_public_hub_reexports_load_only_the_requested_tracker(tmp_path, hub, package):
    output = isolated(
        tmp_path,
        f"""
import json
from wildedge import hubs
tracker = getattr(hubs, {("HuggingFaceHubTracker" if hub == "huggingface" else "TorchHubTracker")!r})
assert tracker().name == {hub!r}
print(json.dumps(sorted(set(sys.modules) & {set(OPTIONAL)!r})))
""",
        working=(package,),
    )
    # HF's file_download is absent in this minimal fixture, but importing its root
    # still proves the optional package is touched only for the requested tracker.
    assert json.loads(output) == [package]


def test_manual_pytorch_subclass_registration_does_not_need_timm(tmp_path):
    isolated(
        tmp_path,
        """
from unittest.mock import patch
from pathlib import Path
import wildedge
from wildedge.platforms.device_info import DeviceInfo
# The base belongs to torch; the application subclass belongs to __main__.
Module = type("Module", (), {"__module__": "torch.nn"})
class ApplicationModel(Module):
    pass
with patch("wildedge.client.Consumer"), patch("wildedge.client.Transmitter"), patch("wildedge.paths.default_sdk_state_dir", return_value=Path("state")):
    client = wildedge.WildEdge(dsn="https://secret@example.invalid/project", device=DeviceInfo("test", "linux"), sampling_interval_s=None)
    handle = client.register_model(ApplicationModel(), model_id="application-model", auto_instrument=False)
    assert handle.info.model_format == "pytorch"
    assert "torch" in sys.modules
    assert not ({"timm", "tensorflow", "transformers", "huggingface_hub"} & set(sys.modules))
    client.close()
""",
        working=("torch", "numpy"),
    )


@pytest.mark.parametrize(
    ("base_name", "base_module", "format_name", "working"),
    [
        ("Model", "keras.src.models.model", "tensorflow", ("numpy",)),
        ("Model", "tf_keras.src.engine.training", "tensorflow", ("numpy",)),
        ("Module", "mlx.nn.layers.base", "mlx", ("mlx", "numpy")),
    ],
)
def test_manual_registration_does_not_load_companion_runtime(
    tmp_path, base_name, base_module, format_name, working
):
    isolated(
        tmp_path,
        f"""
from unittest.mock import patch
from pathlib import Path
import wildedge
from wildedge.platforms.device_info import DeviceInfo
Base = type({base_name!r}, (), {{"__module__": {base_module!r}}})
class ApplicationModel(Base):
    name = "application-model"
    weights = []
with patch("wildedge.client.Consumer"), patch("wildedge.client.Transmitter"), patch("wildedge.paths.default_sdk_state_dir", return_value=Path("state")):
    client = wildedge.WildEdge(dsn="https://secret@example.invalid/project", device=DeviceInfo("test", "linux"), sampling_interval_s=None)
    handle = client.register_model(ApplicationModel(), model_id="application-model", auto_instrument=False)
    assert handle.info.model_format == {format_name!r}
    assert not ({{"tensorflow", "mlx_lm", "timm", "transformers"}} & set(sys.modules))
    client.close()
""",
        working=working,
    )


@pytest.mark.parametrize(
    ("adapter", "runtime_name", "runtime_attr", "flag", "extractor", "method"),
    [
        (
            "pytorch",
            "timm",
            "_timm",
            "_timm_patched",
            "PytorchExtractor",
            "install_timm_patch",
        ),
        (
            "mlx",
            "mlx_lm",
            "_mlx_lm",
            "_mlx_patched",
            "MlxExtractor",
            "install_auto_load_patch",
        ),
        (
            "tensorflow",
            "tensorflow",
            "_tf",
            "_tf_auto_patched",
            "TensorflowExtractor",
            "install_auto_load_patch",
        ),
    ],
)
def test_activation_imports_runtime_after_extractor_was_used(
    monkeypatch, adapter, runtime_name, runtime_attr, flag, extractor, method
):
    module = importlib.import_module(f"wildedge.integrations.{adapter}")
    runtime = ModuleType(runtime_name)
    if adapter == "pytorch":
        runtime.create_model = lambda *_a, **_kw: None
    elif adapter == "mlx":
        runtime.load = lambda *_a, **_kw: None
        runtime.generate = lambda *_a, **_kw: None
    else:
        runtime.keras = SimpleNamespace(
            models=SimpleNamespace(load_model=lambda *_a, **_kw: None)
        )
        runtime.saved_model = SimpleNamespace(load=lambda *_a, **_kw: None)
    monkeypatch.setitem(sys.modules, runtime_name, runtime)
    monkeypatch.setattr(module, runtime_attr, None)
    monkeypatch.setattr(module, flag, False)
    getattr(getattr(module, extractor), method)(lambda: None)
    assert getattr(module, runtime_attr) is runtime
    assert getattr(module, flag) is True


def test_registry_entries_point_at_real_code():
    for spec in INTEGRATION_SPECS:
        if spec.installer is not None:
            assert callable(resolve(spec.installer)), spec.name
    for spec in EXTRACTOR_SPECS:
        assert isinstance(spec.load(), BaseExtractor), spec.class_name
    for spec in HUB_SPECS:
        assert issubclass(resolve(spec.tracker), BaseHubTracker), spec.name
