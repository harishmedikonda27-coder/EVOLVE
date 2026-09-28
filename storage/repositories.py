import json
from typing import Dict, Any, List, Optional
from storage.database import Database


class RunRepository:
    def __init__(self, db: Database):
        self.db = db

    def create_run(self, run_id: str, mode: str, seed: int, experiment_type: str, state: str, created_at: str):
        with self.db.get_connection() as conn:
            conn.execute(
                "INSERT INTO runs (run_id, mode, seed, experiment_type, state, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, mode, seed, experiment_type, state, created_at)
            )
            conn.commit()

    def update_state(self, run_id: str, state: str):
        with self.db.get_connection() as conn:
            conn.execute("UPDATE runs SET state = ? WHERE run_id = ?", (state, run_id))
            conn.commit()

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self.db.get_connection() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            return dict(row) if row else None


class EventRepository:
    def __init__(self, db: Database):
        self.db = db

    def save_event(self, event_id: str, run_id: str, event_type: str, step: int, phase: str, timestamp: str, payload: Dict[str, Any]):
        with self.db.get_connection() as conn:
            conn.execute(
                "INSERT INTO events (event_id, run_id, event_type, step, phase, timestamp, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (event_id, run_id, event_type, step, phase, timestamp, json.dumps(payload))
            )
            conn.commit()

    def get_events(self, run_id: str) -> List[Dict[str, Any]]:
        with self.db.get_connection() as conn:
            rows = conn.execute("SELECT * FROM events WHERE run_id = ? ORDER BY step ASC", (run_id,)).fetchall()
            result = []
            for r in rows:
                item = dict(r)
                item["payload"] = json.loads(item["payload_json"])
                del item["payload_json"]
                result.append(item)
            return result

class TopologyHistoryRepository:
    def __init__(self, db: Database):
        self.db = db

    def save_snapshot(self, record: Dict[str, Any]):
        with self.db.get_connection() as conn:
            # We assume topology_snapshot_json is already serialized, or we do it here
            snap_json = record.get("topology_snapshot_json")
            if not snap_json and "topology_snapshot" in record:
                snap_json = json.dumps(record["topology_snapshot"])
            
            conn.execute(
                """INSERT INTO topology_history 
                (history_id, run_id, level_id, level_index, event_id, event_type, 
                mutation_type, mutation_status, label, timestamp, topology_revision, 
                performance, parameter_count, active_module_count, edge_count, 
                topology_snapshot_json, parent_history_id) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record["history_id"],
                    record["run_id"],
                    record.get("level_id"),
                    record.get("level_index"),
                    record.get("event_id"),
                    record.get("event_type"),
                    record.get("mutation_type"),
                    record.get("mutation_status"),
                    record.get("label"),
                    record.get("timestamp"),
                    record.get("topology_revision"),
                    record.get("performance"),
                    record.get("parameter_count"),
                    record.get("active_module_count"),
                    record.get("edge_count"),
                    snap_json,
                    record.get("parent_history_id")
                )
            )
            conn.commit()

    def get_levels_for_run(self, run_id: str) -> List[Dict[str, Any]]:
        # Retrieve all distinct levels and count their events
        with self.db.get_connection() as conn:
            rows = conn.execute("""
                SELECT level_index, level_id, 
                       COUNT(history_id) as event_count,
                       MIN(timestamp) as initial_time,
                       MAX(timestamp) as final_time
                FROM topology_history
                WHERE run_id = ?
                GROUP BY level_index, level_id
                ORDER BY level_index ASC
            """, (run_id,)).fetchall()
            return [dict(r) for r in rows]

    def get_history_for_level(self, run_id: str, level_id: str) -> List[Dict[str, Any]]:
        with self.db.get_connection() as conn:
            rows = conn.execute("""
                SELECT history_id, run_id, level_id, level_index, event_id, event_type, 
                       mutation_type, mutation_status, label, timestamp, topology_revision, 
                       performance, parameter_count, active_module_count, edge_count, 
                       parent_history_id
                FROM topology_history 
                WHERE run_id = ? AND level_id = ?
                ORDER BY timestamp ASC
            """, (run_id, level_id)).fetchall()
            return [dict(r) for r in rows]
            
    def get_snapshot(self, run_id: str, history_id: str) -> Optional[Dict[str, Any]]:
        with self.db.get_connection() as conn:
            row = conn.execute("SELECT * FROM topology_history WHERE run_id = ? AND history_id = ?", (run_id, history_id)).fetchone()
            if not row:
                return None
            result = dict(row)
            if result.get("topology_snapshot_json"):
                result["topology_snapshot"] = json.loads(result["topology_snapshot_json"])
            del result["topology_snapshot_json"]
            return result
