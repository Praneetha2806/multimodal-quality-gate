"""DuckDB insert/query helpers for the `events` table."""
import json
import os
from datetime import datetime

import duckdb

DB_PATH = os.getenv("GATE_DB", "gate.duckdb")

SCHEMA = """
CREATE SEQUENCE IF NOT EXISTS events_id_seq;
CREATE TABLE IF NOT EXISTS events (
    id            INTEGER DEFAULT nextval('events_id_seq') PRIMARY KEY,
    filename      VARCHAR,
    modality      VARCHAR,
    size_kb       DOUBLE,
    passed        BOOLEAN,
    reason        VARCHAR,
    metrics_json  VARCHAR,
    model_used    VARCHAR,
    latency_ms    DOUBLE,
    est_cost_usd  DOUBLE,
    created_at    TIMESTAMP
);
"""


def _connect():
    con = duckdb.connect(DB_PATH)
    con.execute(SCHEMA)
    return con


def insert_event(filename, modality, size_kb, passed=None, reason="", metrics=None,
                 model_used=None, latency_ms=None, est_cost_usd=None):
    with _connect() as con:
        row = con.execute(
            """INSERT INTO events (filename, modality, size_kb, passed, reason, metrics_json,
                                   model_used, latency_ms, est_cost_usd, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id""",
            [filename, modality, size_kb, passed, reason, json.dumps(metrics or {}, default=str),
             model_used, latency_ms, est_cost_usd, datetime.now()],
        ).fetchone()
    return row[0]


def query_events(where=None):
    sql = "SELECT * FROM events"
    if where:
        sql += f" WHERE {where}"
    sql += " ORDER BY created_at DESC"
    with _connect() as con:
        return con.execute(sql).df()
