"""Gunicorn configuration — Linux only.

On macOS use waitress instead (demo.sh selects automatically).
Requires llama-cpp-python built without Metal:
  CMAKE_ARGS="-DGGML_METAL=OFF" pip install llama-cpp-python --no-binary llama-cpp-python

With CPU-only GGML, the model loaded via preload_app=True in the master is
inherited safely by forked workers via copy-on-write.
"""

workers = 2
bind = "0.0.0.0:8100"
timeout = 120
preload_app = True
control_socket_disable = True
