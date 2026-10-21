"""Create an isolated, fictional SQL screenshot preview using the actual dashboard.

Usage: python prepare.py /path/to/wild-edge /private/tmp/wildedge-sql-preview
Requires duckdb in the Python environment and installed dashboard node_modules.
"""

import json
import math
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

import duckdb

source = Path(__file__).resolve().parent
repository = Path(sys.argv[1]).resolve()
destination = Path(sys.argv[2]).resolve()
if destination.exists():
    raise SystemExit(
        "Choose a new preview directory; existing files are never replaced."
    )
shutil.copytree(
    repository / "dashboard",
    destination,
    ignore=shutil.ignore_patterns("node_modules", "dist", "dist-demo", ".vite"),
)
(destination / "node_modules").symlink_to(
    repository / "dashboard" / "node_modules", target_is_directory=True
)
connection = duckdb.connect()
connection.execute(
    "CREATE TABLE events(event_timestamp TIMESTAMP, event_type VARCHAR, model_name VARCHAR)"
)
events = []
models = ["Gemma 3 4B · local", "Llama 3.2 3B · local", "GPT-4o mini · cloud"]
for hour in range(48):
    timestamp = datetime(2026, 10, 19) + timedelta(hours=hour)
    daytime = (1 + math.sin((hour % 24 - 7) * math.pi / 12)) / 2
    volumes = [
        round(45 + 175 * daytime + 20 * math.sin(hour * 0.7)),
        round(30 + 115 * daytime + 12 * math.cos(hour * 0.9)),
        round(15 + 55 * daytime + (35 if 31 <= hour <= 37 else 0)),
    ]
    for model, count in zip(models, volumes, strict=True):
        events.extend(
            (timestamp + timedelta(seconds=second * 7), "inference", model)
            for second in range(count)
        )
connection.executemany("INSERT INTO events VALUES (?, ?, ?)", events)
query = (source / "query.sql").read_text()
cursor = connection.execute(query)
rows = [
    [value.isoformat() if isinstance(value, datetime) else value for value in row]
    for row in cursor.fetchall()
]
assert len(rows) == 48 * 3
assert sum(row[2] for row in rows) == len(events)
fixture = {
    "columns": [description[0] for description in cursor.description],
    "rows": rows,
    "truncated": False,
    "elapsed_ms": 0,
}

(destination / "src/services/demo/screenshot-result.json").write_text(
    json.dumps(fixture)
)
shutil.copyfile(
    source / "query.sql", destination / "src/services/demo/screenshot-query.sql"
)
shutil.copyfile(
    source / "DemoAnalytics.vue", destination / "src/views/DemoAnalytics.vue"
)
print(f"Computed {len(rows)} grouped rows from {len(events)} fictional events.")
print(f"Preview prepared in {destination}")
