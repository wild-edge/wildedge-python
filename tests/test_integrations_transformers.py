"""Inference tracking for transformers pipelines and models (#70)."""

from __future__ import annotations

import threading
import types

import pytest

import wildedge.integrations.transformers as tf_mod
from wildedge.integrations.transformers import TransformersExtractor
from wildedge.model import ModelHandle, ModelInfo


class FakeModel:
    """Enough of a torch module for install_hooks: forward hooks and a config."""

    config = types.SimpleNamespace(
        name_or_path="tiny-bert",
        model_type="bert",
        architectures=["BertForSequenceClassification"],
    )

    def __init__(self) -> None:
        self.pre_hooks: list = []
        self.post_hooks: list = []

    def register_forward_pre_hook(self, hook):
        self.pre_hooks.append(hook)

    def register_forward_hook(self, hook):
        self.post_hooks.append(hook)

    def __call__(self, input_ids):
        args = (input_ids,)
        for hook in self.pre_hooks:
            hook(self, args)
        output = [[0.1, 0.9]]
        for hook in self.post_hooks:
            hook(self, args, output)
        return output


class FakeTensor:
    shape = (1, 4)


def make_pipeline_class():
    class Pipeline:
        task = "text-classification"

        def __init__(self, model, fail=False):
            self.model = model
            self.fail = fail

        def __call__(self, inputs, *args, **kwargs):
            if self.fail:
                raise RuntimeError("tokenizer exploded")
            self.model(FakeTensor())
            return [{"label": "LABEL_1", "score": 0.9}]

    # is_pipeline() looks for a class named Pipeline from a transformers module.
    Pipeline.__module__ = "transformers.pipelines.base"
    return Pipeline


@pytest.fixture
def patched_pipeline(monkeypatch):
    pipeline_cls = make_pipeline_class()
    fake_transformers = types.SimpleNamespace(Pipeline=pipeline_cls)
    monkeypatch.setattr(tf_mod, "_transformers", fake_transformers)
    TransformersExtractor._patch_pipeline_call()
    return pipeline_cls


def make_handle(events: list) -> ModelHandle:
    info = ModelInfo(
        model_name="tiny-bert",
        model_version="1",
        model_source="local",
        model_format="transformers",
    )
    return ModelHandle(model_id="tiny-bert", info=info, publish=events.append)


def tracked_pipeline(pipeline_cls, events, fail=False):
    """A pipeline wrapping a model, both with hooks, as after pipeline() loads."""
    extractor = TransformersExtractor()
    handle = make_handle(events)
    model = FakeModel()
    extractor.install_hooks(model, handle)  # from_pretrained ran first
    pipe = pipeline_cls(model, fail=fail)
    extractor.install_hooks(pipe, handle)  # then the pipeline superseded it
    return pipe, model


def inference_events(events: list) -> list:
    return [e for e in events if e["event_type"] == "inference"]


def test_pipeline_call_is_tracked_once(patched_pipeline):
    events: list = []
    pipe, _ = tracked_pipeline(patched_pipeline, events)

    pipe("hello world")

    inferences = inference_events(events)
    assert len(inferences) == 1
    inference = inferences[0]["inference"]
    # The pipeline-level event, with what only the pipeline knows.
    assert inference["batch_size"] == 1
    assert inference["input_meta"]["word_count"] == 2
    assert inference["output_meta"]["task"] == "classification"


def test_direct_model_call_is_still_tracked(patched_pipeline):
    events: list = []
    _, model = tracked_pipeline(patched_pipeline, events)

    model(FakeTensor())

    inferences = inference_events(events)
    assert len(inferences) == 1
    assert inferences[0]["inference"]["input_meta"]["token_count"] == 4


def test_failed_pipeline_call_resets_suppression(patched_pipeline):
    events: list = []
    pipe, model = tracked_pipeline(patched_pipeline, events, fail=True)

    with pytest.raises(RuntimeError, match="tokenizer exploded"):
        pipe("hello")
    assert [e["event_type"] for e in events] == ["error"]

    model(FakeTensor())
    assert len(inference_events(events)) == 1


def test_untracked_pipeline_does_not_suppress_model_hooks(patched_pipeline):
    events: list = []
    model = FakeModel()
    TransformersExtractor().install_hooks(model, make_handle(events))
    pipe = patched_pipeline(model)  # no handle on the pipeline itself

    pipe("hello")

    assert len(inference_events(events)) == 1


class CallsFromAnotherThreadToo:
    """Pipeline model that also runs the hooked model on a second thread."""

    def __init__(self, model: FakeModel) -> None:
        self.model = model

    def __call__(self, input_ids):
        other = threading.Thread(target=self.model, args=(input_ids,))
        other.start()
        other.join()
        return self.model(input_ids)


def test_model_call_on_another_thread_during_pipeline_call_is_tracked(
    patched_pipeline,
):
    events: list = []
    extractor = TransformersExtractor()
    handle = make_handle(events)
    model = FakeModel()
    extractor.install_hooks(model, handle)
    pipe = patched_pipeline(CallsFromAnotherThreadToo(model))
    extractor.install_hooks(pipe, handle)

    pipe("hello")

    # The pipeline's own event, and the other thread's model call; the model
    # call on the pipeline's thread is part of the pipeline event.
    metas = [e["inference"]["input_meta"] for e in inference_events(events)]
    assert len(metas) == 2
    assert {"char_count": 5, "word_count": 1} in metas  # the pipeline call
    assert {"token_count": 4} in metas  # the other thread's model call
