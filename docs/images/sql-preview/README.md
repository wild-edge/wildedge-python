# Reproduce the SQL screenshot

This is an illustrative local preview, not a deployed demo feature or production
telemetry. The SQL runs against deterministic fictional events in DuckDB, and
the existing dashboard SqlExplorer and QueryResult components render the result.
No chart pixels or query results are fabricated independently of the query.

Prerequisites: a wild-edge checkout with dashboard dependencies installed, Node,
and a Python environment containing DuckDB (the platform's `.venv` includes it).

```bash
/path/to/wild-edge/.venv/bin/python prepare.py /path/to/wild-edge /private/tmp/sql-preview
cd /private/tmp/sql-preview
npm run typecheck
npm run build:demo
MOCKED_BE=true npm run preview -- --outDir dist-demo --host 127.0.0.1 --port 4180
```

Open `http://127.0.0.1:4180/demo/wildedge-demolabs/news-reader/analytics?mode=sql`
at 1600 × 1100 and capture the full viewport. The preview uses the existing
compact chart layout to keep the query, axes, stacked bars, and legend visible.
The source checkout and production site are unchanged.

`prepare.py` creates 12,876 events spanning 48 hours and three fictional model
workloads. It executes `query.sql` and checks that its 144 grouped rows account
for every generated event. `DemoAnalytics.vue` is a screenshot-only composition
of existing dashboard components; the editable SQL field does not execute queries.
