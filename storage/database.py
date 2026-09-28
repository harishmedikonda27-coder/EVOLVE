import sqlite3
import os
from typing import Optional


class Database:
    def __init__(self, db_path: str = "storage/evolve.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self.get_connection() as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                mode TEXT,
                seed INTEGER,
                experiment_type TEXT,
                state TEXT,
                created_at TEXT
            );
            """)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                run_id TEXT,
                event_type TEXT,
                step INTEGER,
                phase TEXT,
                timestamp TEXT,
                payload_json TEXT
            );
            """)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                step INTEGER,
                episode INTEGER,
                phase TEXT,
                reward REAL,
                success INTEGER,
                need_score REAL,
                active_module_count INTEGER,
                parameter_count INTEGER
            );
            """)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS topology_history (
                history_id TEXT PRIMARY KEY,
                run_id TEXT,
                level_id TEXT,
                level_index INTEGER,
                event_id TEXT,
                event_type TEXT,
                mutation_type TEXT,
                mutation_status TEXT,
                label TEXT,
                timestamp TEXT,
                topology_revision INTEGER,
                performance REAL,
                parameter_count INTEGER,
                active_module_count INTEGER,
                edge_count INTEGER,
                topology_snapshot_json TEXT,
                parent_history_id TEXT
            );
            """)
            conn.commit()
