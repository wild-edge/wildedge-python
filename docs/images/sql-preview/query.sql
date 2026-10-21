SELECT
  date_trunc('hour', event_timestamp) AS hour,
  model_name,
  count(*) AS inferences
FROM events
WHERE event_type = 'inference'
  AND event_timestamp >= TIMESTAMP '2026-10-19 00:00:00'
  AND event_timestamp <  TIMESTAMP '2026-10-21 00:00:00'
GROUP BY 1, 2
ORDER BY 1, 2;
