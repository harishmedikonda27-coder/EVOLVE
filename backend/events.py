from dataclasses import dataclass, field
from typing import Dict, Any, List, Callable, Optional
import datetime
import uuid


@dataclass
class Event:
    event_type: str
    run_id: str
    step: int
    phase: str
    payload: Dict[str, Any]
    event_id: str = field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:8]}")
    timestamp: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "run_id": self.run_id,
            "step": self.step,
            "phase": self.phase,
            "timestamp": self.timestamp,
            "payload": self.payload,
        }


class EventBus:
    def __init__(self):
        self.subscribers: List[Callable[[Event], None]] = []
        self.history: List[Event] = []

    def subscribe(self, callback: Callable[[Event], None]):
        self.subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[Event], None]):
        if callback in self.subscribers:
            self.subscribers.remove(callback)

    def publish(self, event: Event):
        self.history.append(event)
        for callback in list(self.subscribers):
            try:
                callback(event)
            except Exception as e:
                print(f"[EventBus] Subscriber error: {e}")

    def get_history(self, run_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if run_id:
            return [e.to_dict() for e in self.history if e.run_id == run_id]
        return [e.to_dict() for e in self.history]
