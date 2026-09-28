from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional


class RunCreateRequest(BaseModel):
    mode: str = Field(default="research", description="Run mode: research or demo")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    experiment_type: str = Field(default="evolve", description="fixed, replay, always_grow, evolve")
    # task_sequence is deprecated in favor of families/levels for CLI
    task_sequence: List[str] = Field(default_factory=lambda: ["A", "B", "A"])
    families: List[str] = Field(default_factory=lambda: ["HierarchicalNavigation"])
    levels: int = Field(default=3)


class RunResponse(BaseModel):
    run_id: str
    state: str
    mode: str
    seed: int
    experiment_type: str
    created_at: str


class EventSchema(BaseModel):
    event_id: str
    event_type: str
    run_id: str
    step: int
    phase: str
    timestamp: str
    payload: Dict[str, Any]
    version: int = 1


class EnvironmentDefinition(BaseModel):
    id: str
    name: str
    family: str
    description: str
    config: Dict[str, Any]

