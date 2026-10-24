"""Tests for ONNX, GGUF, and Keras integration extractors."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import pytest

from wildedge.attachments import AttachmentManager, AttachmentStore
from wildedge.integrations import gguf as gguf_integration
from wildedge.integrations.gguf import (
    GgufExtractor,
    call_arguments,
    make_gguf_generation_config,
    make_gguf_output_meta,
    make_gguf_stream_callbacks,
    parse_quantization,
)
from wildedge.integrations.gguf import (
    detect_accelerator as gguf_detect_accelerator,
)
from wildedge.integrations.keras import KerasExtractor
from wildedge.integrations.keras import detect_accelerator as keras_detect_accelerator
from wildedge.integrations.onnx import (
    OnnxExtractor,
    model_file_stem,
    model_id_from_path,
    parse_model_version,
)
from wildedge.integrations.onnx import (
    detect_accelerator as onnx_detect_accelerator,
)
from wildedge.integrations.onnx import (
    detect_quantization as onnx_detect_quantization,
)
from wildedge.model import ModelHandle, ModelInfo

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def make_handle(publish_spy) -> ModelHandle:
    info = ModelInfo(
        model_name="test",
        model_version="1.0",
        model_source="local",
        model_format="test",
    )
    return ModelHandle(model_id="m", info=info, publish=publish_spy)


# ---------------------------------------------------------------------------
# Fake objects; no ML libraries required
# ---------------------------------------------------------------------------

# ONNX


class _FakeMeta:
    graph_name = "resnet50"
    version = 1


class _FakeInput:
    type = "tensor(float32)"


class _FakeOrtSession:
    def get_modelmeta(self):
        return _FakeMeta()

    def get_inputs(self):
        return [_FakeInput()]

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, output_names, input_feed, run_options=None):
        return [[1.0]]


# GGUF


class Llama:  # must match the name check in GgufExtractor.can_handle
    """Mirrors llama-cpp-python's call graph: every entry point runs create_completion."""

    model_path = "/models/llama-3-Q4_K_M.gguf"
    n_gpu_layers = 0
    metadata = {"general.architecture": "llama", "general.version": "3.0"}

    def create_completion(self, prompt="", **kwargs):
        if kwargs.get("stream"):
            return iter(
                [
                    {"choices": [{"text": "hi", "finish_reason": None}]},
                    {
                        "choices": [{"text": "!", "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
                    },
                ]
            )
        return {
            "choices": [{"text": "hi!", "finish_reason": "length"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }

    def __call__(self, prompt="", **kwargs):
        return self.create_completion(prompt=prompt, **kwargs)

    def create_chat_completion(self, messages, **kwargs):
        # The chat handler formats the messages, then calls llama.create_completion.
        prompt = "".join(str(m["content"]) for m in messages)
        result = self.create_completion(prompt=prompt, **kwargs)
        if kwargs.get("stream"):
            return result
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "hi!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": result["usage"],
        }


class _FailingLlama(Llama):
    def create_completion(self, prompt="", **kwargs):
        raise RuntimeError("cuda oom")


# Keras


class _KerasBase:
    """Looks like keras.Model to the MRO check."""


_KerasBase.__name__ = "Model"
_KerasBase.__module__ = "keras.src.trainers.trainer"


class _FakeKerasModel(_KerasBase):
    name = "my_seq"
    weights = []

    def __call__(self, *args, **kwargs):
        return [1.0, 2.0]


class _FailingKerasModel(_KerasBase):
    name = "failing"
    weights = []

    def __call__(self, *args, **kwargs):
        raise RuntimeError("gpu oom")


# ---------------------------------------------------------------------------
# ONNX
# ---------------------------------------------------------------------------


class TestOnnxExtractor:
    extractor = OnnxExtractor()

    def test_can_handle_true_when_ort_available(self):
        with patch("wildedge.integrations.onnx.ort") as mock_ort:
            mock_ort.InferenceSession = _FakeOrtSession
            assert self.extractor.can_handle(_FakeOrtSession()) is True

    def test_can_handle_false_when_ort_not_installed(self):
        with patch("wildedge.integrations.onnx.ort", None):
            assert self.extractor.can_handle(_FakeOrtSession()) is False

    def test_can_handle_false_for_wrong_type(self):
        with patch("wildedge.integrations.onnx.ort") as mock_ort:
            mock_ort.InferenceSession = _FakeOrtSession
            assert self.extractor.can_handle(object()) is False

    def test_detect_accelerator_cuda_provider(self):
        session = MagicMock()
        session.get_providers.return_value = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        assert onnx_detect_accelerator(session) == "cuda"

    def test_detect_accelerator_defaults_to_cpu(self):
        session = MagicMock()
        session.get_providers.return_value = ["CPUExecutionProvider"]
        assert onnx_detect_accelerator(session) == "cpu"

    def test_detect_quantization_int8(self):
        session = MagicMock()
        session.get_inputs.return_value = [MagicMock(type="tensor(int8)")]
        assert onnx_detect_quantization(session) == "int8"

    def test_detect_quantization_f32(self):
        session = MagicMock()
        session.get_inputs.return_value = [MagicMock(type="tensor(float32)")]
        assert onnx_detect_quantization(session) == "f32"

    def test_extract_info_uses_graph_name_as_model_id(self):
        model_id, info = self.extractor.extract_info(_FakeOrtSession(), {})
        assert model_id == "resnet50"
        assert info.model_format == "onnx"

    def test_extract_info_none_model_id_when_graph_name_empty(self):
        session = _FakeOrtSession()
        session.get_modelmeta = lambda: MagicMock(graph_name="", version=0)
        model_id, _ = self.extractor.extract_info(session, {})
        assert model_id is None

    def _piper_session(self, path="/voices/pl_PL-gosia-medium.onnx"):
        session = _FakeOrtSession()
        session.get_modelmeta = lambda: MagicMock(
            graph_name="main_graph", version=2**63 - 1
        )
        session._model_path = path
        return session

    def test_main_graph_name_falls_back_to_file_stem(self):
        model_id, info = self.extractor.extract_info(self._piper_session(), {})
        assert model_id == "pl_PL-gosia-medium"
        assert info.model_name == "pl_PL-gosia-medium"

    def test_int64_max_version_is_unset(self):
        _, info = self.extractor.extract_info(self._piper_session(), {})
        assert info.model_version == "unknown"

    def test_different_voices_get_different_ids(self):
        a, _ = self.extractor.extract_info(self._piper_session("/v/en_US-amy.onnx"), {})
        b, _ = self.extractor.extract_info(
            self._piper_session("/v/de_DE-thorsten.onnx"), {}
        )
        assert a != b

    def test_hub_path_keeps_repo_id_and_names_model_after_file(self):
        path = (
            "/hf/hub/models--rhasspy--piper-voices/snapshots/abc/pl/"
            "pl_PL-gosia-medium.onnx"
        )
        model_id, info = self.extractor.extract_info(self._piper_session(path), {})
        assert model_id == "rhasspy/piper-voices"
        assert info.model_name == "pl_PL-gosia-medium"

    def test_session_from_bytes_without_name_still_needs_an_id(self):
        session = self._piper_session(path=None)
        model_id, info = self.extractor.extract_info(session, {})
        assert model_id is None
        assert info.model_name == "onnx-model"

    def test_id_override_names_model_when_graph_and_file_are_generic(self):
        session = self._piper_session(path="/models/model.onnx")
        model_id, info = self.extractor.extract_info(session, {"id": "my-voice"})
        assert model_id == "my-voice"
        assert info.model_name == "my-voice"

    def test_real_graph_name_wins_over_file_stem(self):
        session = _FakeOrtSession()
        session._model_path = "/models/exported-v3.onnx"
        model_id, info = self.extractor.extract_info(session, {})
        assert model_id == "resnet50"
        assert info.model_name == "resnet50"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (None, None),
            (0, None),
            (-1, None),
            (2**63 - 1, None),
            (1, "1"),
            (42, "42"),
            ("7", "7"),
            ("v2", "v2"),
            ("  ", None),
        ],
    )
    def test_parse_model_version(self, raw, expected):
        assert parse_model_version(raw) == expected

    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("/m/pl_PL-gosia-medium.onnx", "pl_PL-gosia-medium"),
            ("C:\\voices\\en_US-amy.onnx", "en_US-amy"),
            ("/m/model.onnx", None),
            ("/m/main_graph.onnx", None),
            ("", None),
            (None, None),
            (b"bytes", None),
        ],
    )
    def test_model_file_stem(self, path, expected):
        assert model_file_stem(path) == expected

    def test_model_id_from_path_skips_generic_stems(self):
        assert model_id_from_path("/m/main_graph.onnx") is None
        assert model_id_from_path("/m/yolo.onnx") == "yolo"

    def test_install_hooks_publishes_inference_on_run(self, publish_spy):
        session = _FakeOrtSession()
        handle = make_handle(publish_spy)
        self.extractor.install_hooks(session, handle)
        session.run(None, {})
        assert len(publish_spy.events) == 1
        assert publish_spy.events[0]["event_type"] == "inference"

    def test_install_hooks_tracks_error_on_exception(self, publish_spy):
        def _fail(*a, **kw):
            raise RuntimeError("inference failed")

        session = _FakeOrtSession()
        session.run = _fail
        handle = make_handle(publish_spy)
        self.extractor.install_hooks(session, handle)
        with pytest.raises(RuntimeError):
            session.run(None, {})
        assert publish_spy.events[0]["event_type"] == "error"


# ---------------------------------------------------------------------------
# GGUF
# ---------------------------------------------------------------------------


class TestGgufExtractor:
    extractor = GgufExtractor()

    def test_can_handle_matches_llama_class(self):
        assert self.extractor.can_handle(Llama()) is True

    def test_can_handle_rejects_other_types(self):
        assert self.extractor.can_handle(object()) is False

    def test_parse_quantization_q4_k_m(self):
        assert parse_quantization("llama-3-Q4_K_M.gguf") == "q4_k_m"

    def test_parse_quantization_f16(self):
        assert parse_quantization("model-F16.gguf") == "f16"

    def test_parse_quantization_none_for_unknown(self):
        assert parse_quantization("model.gguf") is None

    def test_detect_accelerator_cpu_when_no_gpu_layers(self):
        llm = Llama()
        llm.n_gpu_layers = 0
        assert gguf_detect_accelerator(llm) == "cpu"

    def test_detect_accelerator_non_cpu_when_gpu_layers_offloaded(self):
        llm = Llama()
        llm.n_gpu_layers = 32
        with patch(
            "wildedge.integrations.gguf.CURRENT_PLATFORM.gpu_accelerator_for_offload",
            return_value="cuda",
        ):
            assert gguf_detect_accelerator(llm) == "cuda"

    def test_extract_info_uses_filename_stem_as_model_id(self):
        model_id, info = self.extractor.extract_info(Llama(), {})
        assert model_id == "llama-3-Q4_K_M"
        assert info.model_format == "gguf"
        assert info.quantization == "q4_k_m"

    def test_extract_info_reads_family_from_metadata(self):
        _, info = self.extractor.extract_info(Llama(), {})
        assert info.model_family == "llama"

    def test_install_hooks_publishes_inference_on_call(self, publish_spy):
        llm = Llama()
        handle = make_handle(publish_spy)
        self.extractor.install_hooks(llm, handle)
        llm()
        assert len(publish_spy.events) == 1
        assert publish_spy.events[0]["event_type"] == "inference"

    def test_install_hooks_includes_token_counts(self, publish_spy):
        llm = Llama()
        handle = make_handle(publish_spy)
        self.extractor.install_hooks(llm, handle)
        llm()
        event = publish_spy.events[0]
        assert event["inference"]["output_meta"]["tokens_in"] == 10
        assert event["inference"]["output_meta"]["tokens_out"] == 20

    def test_install_hooks_tracks_error_on_exception(self, publish_spy):
        llm = _FailingLlama()
        handle = make_handle(publish_spy)
        self.extractor.install_hooks(llm, handle)
        with pytest.raises(RuntimeError):
            llm()
        assert publish_spy.events[0]["event_type"] == "error"

    def test_install_hooks_tracks_chat_completion(self, publish_spy):
        llm = Llama()
        handle = make_handle(publish_spy)
        self.extractor.install_hooks(llm, handle)
        llm.create_chat_completion(messages=[{"role": "user", "content": "hello"}])
        assert len(publish_spy.events) == 1
        event = publish_spy.events[0]
        assert event["event_type"] == "inference"
        assert event["inference"]["output_meta"]["tokens_out"] == 20

    def test_install_hooks_tracks_create_completion(self, publish_spy):
        llm = Llama()
        handle = make_handle(publish_spy)
        self.extractor.install_hooks(llm, handle)
        llm.create_completion(prompt="hello")
        assert len(publish_spy.events) == 1
        assert publish_spy.events[0]["inference"]["input_meta"]["char_count"] == 5

    def test_install_hooks_tracks_streamed_chat_completion(self, publish_spy):
        llm = Llama()
        handle = make_handle(publish_spy)
        self.extractor.install_hooks(llm, handle)
        chunks = list(llm.create_completion(prompt="hello", stream=True))
        assert len(chunks) == 2
        assert len(publish_spy.events) == 1
        meta = publish_spy.events[0]["inference"]["output_meta"]
        assert meta["tokens_out"] == 20
        assert meta["stop_reason"] == "stop"

    def test_install_hooks_tracks_chat_completion_error(self, publish_spy):
        llm = _FailingLlama()
        handle = make_handle(publish_spy)
        self.extractor.install_hooks(llm, handle)
        with pytest.raises(RuntimeError):
            llm.create_chat_completion(messages=[{"role": "user", "content": "hi"}])
        assert [e["event_type"] for e in publish_spy.events] == ["error"]

    def test_install_hooks_twice_does_not_double_track(self, publish_spy):
        llm = Llama()
        handle = make_handle(publish_spy)
        self.extractor.install_hooks(llm, handle)
        self.extractor.install_hooks(llm, handle)
        llm.create_chat_completion(messages=[{"role": "user", "content": "hello"}])
        assert len(publish_spy.events) == 1

    def test_non_streaming_stop_reason(self, publish_spy):
        llm = Llama()
        self.extractor.install_hooks(llm, make_handle(publish_spy))
        llm.create_completion(prompt="hello")
        meta = publish_spy.events[0]["inference"]["output_meta"]
        assert meta["stop_reason"] == "length"

    def test_generation_config_from_kwargs(self, publish_spy):
        llm = Llama()
        self.extractor.install_hooks(llm, make_handle(publish_spy))
        llm.create_chat_completion(
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.7,
            top_p=0.9,
            max_tokens=64,
            repeat_penalty=1.1,
            stop=["</s>", "User:"],
        )
        config = publish_spy.events[0]["inference"]["generation_config"]
        assert config == {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 64,
            "repetition_penalty": 1.1,
            "stop_sequences_count": 2,
        }

    def test_chat_input_meta(self, publish_spy):
        llm = Llama()
        self.extractor.install_hooks(llm, make_handle(publish_spy))
        llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "what time is it"},
            ]
        )
        meta = publish_spy.events[0]["inference"]["input_meta"]
        assert meta["prompt_type"] == "chat"
        assert meta["turn_index"] == 1
        assert meta["word_count"] == 4
        assert meta["token_count"] == 10

    def test_streamed_chat_input_meta(self, publish_spy):
        llm = Llama()
        self.extractor.install_hooks(llm, make_handle(publish_spy))
        stream = llm.create_chat_completion(
            messages=[{"role": "user", "content": "hello"}], stream=True
        )
        list(stream)
        assert len(publish_spy.events) == 1
        inference = publish_spy.events[0]["inference"]
        assert inference["input_meta"]["prompt_type"] == "chat"
        assert inference["output_meta"]["stop_reason"] == "stop"

    def test_completion_input_meta_is_not_chat(self, publish_spy):
        llm = Llama()
        self.extractor.install_hooks(llm, make_handle(publish_spy))
        llm.create_completion(prompt="hello")
        assert "prompt_type" not in publish_spy.events[0]["inference"]["input_meta"]

    def test_no_attachments_when_disabled(self, publish_spy):
        llm = Llama()
        self.extractor.install_hooks(llm, make_handle(publish_spy))
        llm.create_chat_completion(messages=[{"role": "user", "content": "hello"}])
        assert "attachments" not in publish_spy.events[0]

    @pytest.mark.parametrize("streaming", [False, True])
    @pytest.mark.parametrize("chat", [False, True])
    def test_enabled_uploads_never_capture_payloads(
        self, publish_spy, tmp_path, streaming, chat
    ):
        store = AttachmentStore(str(tmp_path))
        manager = AttachmentManager(store)
        llm = Llama()
        handle = make_handle(publish_spy)
        handle.capture_attachments = manager.capture
        self.extractor.install_hooks(llm, handle)
        if chat:
            result = llm.create_chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "private prompt"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,PRIVATE"},
                            },
                        ],
                    }
                ],
                stream=streaming,
            )
        else:
            result = llm.create_completion(prompt="private prompt", stream=streaming)
        if streaming:
            assert list(result)
        assert len(publish_spy.events) == 1
        assert "attachments" not in publish_spy.events[0]
        assert store.length() == 0

    def test_stop_given_as_string_counts_one_sequence(self):
        config = make_gguf_generation_config({"stop": "</s>"})
        assert config.stop_sequences_count == 1

    def test_generation_config_none_without_sampling_arguments(self):
        assert make_gguf_generation_config({"stop": []}) is None

    def test_output_meta_none_without_tokens_ttft_or_stop_reason(self):
        assert make_gguf_output_meta(None, None, None, None, 10) is None

    def test_call_arguments_fill_defaults_and_flatten_kwargs(self):
        def create_completion(self, prompt, temperature=0.8, **kwargs):
            pass

        args = call_arguments(
            inspect.signature(create_completion), object(), ("hi",), {"seed": 1}
        )
        assert args["prompt"] == "hi"
        assert args["temperature"] == 0.8
        assert args["seed"] == 1
        assert "kwargs" not in args

    def test_call_arguments_fall_back_when_binding_fails(self):
        def create_completion(self, prompt):
            pass

        args = call_arguments(
            inspect.signature(create_completion), object(), ("hi", "extra"), {}
        )
        assert args == {"prompt": "hi"}

    def test_call_arguments_without_signature(self):
        assert call_arguments(None, object(), (), {"prompt": "p", "top_p": 0.5}) == {
            "prompt": "p",
            "top_p": 0.5,
        }

    def test_patched_call_survives_unreadable_signature(self, publish_spy):
        with patch.object(
            gguf_integration.inspect, "signature", side_effect=ValueError("no sig")
        ):
            patched = gguf_integration.build_patched_call(Llama.create_completion)
        llm = Llama()
        setattr(llm, gguf_integration.GGUF_HANDLE_ATTR, make_handle(publish_spy))
        patched(llm, "hello", temperature=0.1)
        event = publish_spy.events[0]["inference"]
        assert event["input_meta"]["char_count"] == 5
        assert event["generation_config"] == {"temperature": 0.1}

    def test_generation_config_error_does_not_block_tracking(self, publish_spy):
        llm = Llama()
        self.extractor.install_hooks(llm, make_handle(publish_spy))
        with patch.object(
            gguf_integration,
            "make_gguf_generation_config",
            side_effect=RuntimeError("boom"),
        ):
            llm.create_completion(prompt="hello")
        assert len(publish_spy.events) == 1
        assert "generation_config" not in publish_spy.events[0]["inference"]

    def test_malformed_usage_still_tracks(self, publish_spy):
        class OddLlama(Llama):
            def create_completion(self, prompt="", **kwargs):
                return {"usage": "n/a"}

        llm = OddLlama()
        self.extractor.install_hooks(llm, make_handle(publish_spy))
        llm.create_completion(prompt="hello")
        assert len(publish_spy.events) == 1

    def test_chat_without_handle_is_untracked(self, publish_spy):
        llm = Llama()
        self.extractor.install_hooks(llm, make_handle(publish_spy))
        setattr(llm, gguf_integration.GGUF_HANDLE_ATTR, None)
        result = llm.create_chat_completion(messages=[{"role": "user", "content": "x"}])
        assert result["choices"][0]["message"]["content"] == "hi!"
        assert publish_spy.events == []

    def test_stream_callbacks_read_chat_deltas_and_skip_non_dicts(self, publish_spy):
        captured = []
        handle = make_handle(publish_spy)
        handle.capture_attachments = lambda a, *_: captured.extend(a) or []
        on_chunk, on_done = make_gguf_stream_callbacks(handle, "p")
        on_chunk("not a dict")
        on_chunk({"choices": [{"delta": {"role": "assistant"}}]})
        on_chunk({"choices": [{"delta": {"content": "he"}}]})
        on_chunk({"choices": [{"delta": {"content": "y"}, "finish_reason": "stop"}]})
        on_done(10, 2)
        assert captured == []
        meta = publish_spy.events[0]["inference"]["output_meta"]
        assert meta["stop_reason"] == "stop"


# ---------------------------------------------------------------------------
# Keras
# ---------------------------------------------------------------------------


class TestKerasExtractor:
    extractor = KerasExtractor()

    def test_can_handle_keras_model(self):
        assert self.extractor.can_handle(_FakeKerasModel()) is True

    def test_can_handle_rejects_plain_object(self):
        assert self.extractor.can_handle(object()) is False

    def test_extract_info_uses_model_name_as_model_id(self):
        model_id, info = self.extractor.extract_info(_FakeKerasModel(), {})
        assert model_id == "my_seq"
        assert info.model_format == "keras"

    def test_extract_info_override_model_id(self):
        model_id, _ = self.extractor.extract_info(_FakeKerasModel(), {"id": "my-bert"})
        assert model_id == "my-bert"

    def test_detect_accelerator_gpu_from_weight_device(self):
        model = _FakeKerasModel()
        model.weights = [MagicMock(device="/GPU:0")]
        with patch(
            "wildedge.platforms.CURRENT_PLATFORM.gpu_accelerator_for_offload",
            return_value="cuda",
        ):
            assert keras_detect_accelerator(model) == "cuda"

    def test_install_hooks_publishes_inference_on_call(self, publish_spy):
        model = _FakeKerasModel()
        handle = make_handle(publish_spy)
        self.extractor.install_hooks(model, handle)
        model()
        assert len(publish_spy.events) == 1
        assert publish_spy.events[0]["event_type"] == "inference"

    def test_install_hooks_tracks_error_and_reraises(self, publish_spy):
        model = _FailingKerasModel()
        handle = make_handle(publish_spy)
        self.extractor.install_hooks(model, handle)
        with pytest.raises(RuntimeError, match="gpu oom"):
            model()
        assert publish_spy.events[0]["event_type"] == "error"

    def test_install_hooks_does_not_affect_other_instances(self, publish_spy):
        model_a = _FakeKerasModel()
        model_b = _FakeKerasModel()
        handle = make_handle(publish_spy)
        self.extractor.install_hooks(model_a, handle)
        model_b()  # should NOT publish
        assert len(publish_spy.events) == 0
