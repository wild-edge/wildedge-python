"""Gemma inference view.

The Llama constructor is patched automatically by `wildedge run --integrations gguf`
via sitecustomize.py; load/unload/inference events are tracked without any
wildedge imports here.

On macOS, waitress (thread-pool, no fork) is used as the WSGI server.
Metal is initialised once at startup in the main process and shared safely
across request threads. gunicorn (fork-based) requires llama-cpp-python built
without Metal on macOS (CMAKE_ARGS="-DGGML_METAL=OFF").
"""

import json
import os
import threading

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from llama_cpp import Llama

REPO = "bartowski/gemma-2-2b-it-GGUF"
FILE = "gemma-2-2b-it-Q4_K_M.gguf"

_llm = Llama.from_pretrained(
    repo_id=REPO,
    filename=FILE,
    n_ctx=512,
    n_gpu_layers=int(os.environ.get("GPU_LAYERS", "-1")),
    verbose=False,
)

# Llama inference is not thread-safe on a single context; serialise requests.
_llm_lock = threading.Lock()


@csrf_exempt
@require_POST
def infer(request):
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "invalid JSON"}, status=400)

    prompt = body.get("prompt", "").strip()
    if not prompt:
        return JsonResponse({"error": "prompt is required"}, status=400)

    with _llm_lock:
        result = _llm(prompt, max_tokens=256, temperature=0.7)

    text = result["choices"][0]["text"].strip()
    return JsonResponse({"response": text})
