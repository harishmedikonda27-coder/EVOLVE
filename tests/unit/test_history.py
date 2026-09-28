import pytest
import sqlite3
import json
from storage.database import Database
from storage.repositories import TopologyHistoryRepository

@pytest.fixture
def memory_db():
    class MemoryDatabase(Database):
        def __init__(self):
            # overriding to use memory
            self.db_path = ":memory:"
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            self._init_db()

        def get_connection(self):
            return self.conn

    return MemoryDatabase()

def test_topology_history_repository(memory_db):
    repo = TopologyHistoryRepository(memory_db)
    
    # save snapshot 1
    repo.save_snapshot({
        "history_id": "h1",
        "run_id": "r1",
        "level_id": "L1",
        "level_index": 0,
        "event_id": "e1",
        "event_type": "LEVEL_STARTED",
        "mutation_type": None,
        "mutation_status": None,
        "label": "INITIAL",
        "timestamp": "2026-09-24T12:00:00Z",
        "topology_revision": 1,
        "performance": None,
        "parameter_count": 100,
        "active_module_count": 1,
        "edge_count": 0,
        "topology_snapshot": {"nodes": [], "edges": []}
    })
    
    # save snapshot 2
    repo.save_snapshot({
        "history_id": "h2",
        "run_id": "r1",
        "level_id": "L1",
        "level_index": 0,
        "event_id": "e2",
        "event_type": "MUTATION_ACCEPTED",
        "mutation_type": "GROW",
        "mutation_status": "ACCEPTED",
        "label": "GROW \u00b7 ACCEPTED",
        "timestamp": "2026-09-24T12:01:00Z",
        "topology_revision": 2,
        "performance": 0.5,
        "parameter_count": 150,
        "active_module_count": 2,
        "edge_count": 1,
        "topology_snapshot": {"nodes": [{"id": "n1"}], "edges": []}
    })
    
    levels = repo.get_levels_for_run("r1")
    assert len(levels) == 1
    assert levels[0]["level_index"] == 0
    assert levels[0]["event_count"] == 2
    
    history = repo.get_history_for_level("r1", "L1")
    assert len(history) == 2
    assert history[0]["history_id"] == "h1"
    assert history[1]["history_id"] == "h2"
    
    snap = repo.get_snapshot("r1", "h2")
    assert snap is not None
    assert snap["topology_snapshot"] == {"nodes": [{"id": "n1"}], "edges": []}
    assert snap["parameter_count"] == 150
