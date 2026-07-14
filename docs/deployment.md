# Deployment

How the SDK behaves in real process models: servers that fork, workers that
exec, build steps, and processes with no DSN configured. Everything on this
page is a contract the SDK keeps, not an implementation detail.

## How initialization happens

There are three ways a process gets its client, in order of coverage:

| Mode | Instrumentation coverage | Use when |
|---|---|---|
| `wildedge run -- <cmd>` | Guaranteed: patches before any user code runs | You control the start command |
| `wildedge.init(...)` at startup | Everything created after the call | You control application code |
| Lazy, from the first module-level call | Best effort: only what is created afterwards | Manual tracking only |

All three produce the same thing: one process-wide default client that
`wildedge.span()`, `wildedge.register_model()` and the rest delegate to.
`init()` without `dsn` reuses whatever client already exists, so code written
for in-process init runs unchanged under `wildedge run`.

The lazy mode configures itself from the environment: `WILDEDGE_DSN`,
`WILDEDGE_INTEGRATIONS`, `WILDEDGE_HUBS`. Unset integration variables mean
nothing is patched; auto-instrumentation is always opt-in.

## The no-DSN contract

Without `WILDEDGE_DSN`, the client is a no-op: no background threads, no
network, no patched frameworks, every event dropped. This is a supported mode
for development and CI, not an error; the SDK logs one INFO line per process
and stays quiet. Leave the SDK integrated and unset the variable wherever you
do not want telemetry (local dev, test runs, build and migration steps).

Under `wildedge run`, a missing DSN warns on stderr and your program runs
untracked. Telemetry failing must not take production down; that is the
default policy.

## Strict mode

`wildedge run --strict` inverts the failure policy for deployments where
running unobserved is worse than not running. Bootstrap failures then
terminate the process before your program starts:

| Exit code | Meaning |
|---|---|
| 120 | Configuration error (missing or invalid DSN) |
| 121 | A requested integration could not be instrumented (requires `--strict-integrations`) |
| 122 | Internal bootstrap error |

`--strict-integrations` is its own opt-in and exits with 121 on failure even
without `--strict`; asking for it is the request to fail.

## Forking and multi-process servers

`wildedge run` prepends `wildedge/autoload/` to `PYTHONPATH` and replaces
itself with your command. Every Python interpreter that starts under it runs
the bundled `sitecustomize.py`, which bootstraps the runtime before any user
code. Two mechanisms make this safe across worker models:

- Fresh interpreters (exec or spawn) bootstrap themselves via sitecustomize;
  a `sys.modules` marker prevents double initialization within one
  interpreter.
- Forked children inherit the parent's initialized runtime, and
  `os.register_at_fork` hooks stop the SDK's background threads before each
  fork and start fresh ones in parent and child afterward, so forked workers
  transmit normally instead of inheriting dead threads.

How that plays out on common servers:

| Server | Worker model | Behavior under `wildedge run` |
|---|---|---|
| gunicorn (sync/gthread, with or without `preload_app`) | fork from master | Master bootstraps once; each worker gets fresh SDK threads via the at-fork hooks |
| uvicorn (`--workers`, `--reload`) | spawn / exec | Every worker is a fresh interpreter and bootstraps itself |
| granian | spawned worker processes | Each worker bootstraps itself |
| daphne | single process | One bootstrap, nothing special |
| waitress | single process, thread pool | One bootstrap; the client is thread-safe |
| celery (prefork pool) | fork from master | Same as gunicorn: at-fork hooks restart threads per worker |

For managed platforms, wrap only the serving command. Build steps and
migrations run in separate processes that gain nothing from telemetry; leave
them unwrapped or unset `WILDEDGE_DSN` there.

## One client per process

- Framework patches are installed at most once per process; the first client
  to instrument owns the patch.
- `init()` reuses the existing default client unless you pass `dsn`
  explicitly, so the CLI-installed client and application code share one
  client instead of racing.
- Trace and span correlation lives in contextvars at module level, not on a
  client. Auto-instrumented events emitted inside `wildedge.trace(...)` /
  `wildedge.span(...)` blocks correlate into the same trace no matter which
  client emits them.

## Environment propagation

By default, `wildedge run` leaves its `WILDEDGE_*` variables in the
environment so that exec'd children (reload workers) can bootstrap. Pass
`--no-propagate` to have each bootstrapped process scrub the run-scoped
variables after initialization, keeping them away from nested subprocesses
you spawn yourself.
