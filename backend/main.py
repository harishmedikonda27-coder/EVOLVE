from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import datetime
import uuid
import asyncio
import threading
from typing import Dict, Any, Optional

from backend.schemas import RunCreateRequest, RunResponse
from backend.state import RunState, RunStateMachine
from backend.events import EventBus, Event
from backend.websocket import ConnectionManager
from storage.database import Database
from storage.repositories import RunRepository, EventRepository, TopologyHistoryRepository
from backend.experiments.run import run_controls
from environment.base import TaskGenerationFailure

app = FastAPI(
    title="EVOLVE Research Platform API",
    description="API and event stream server for EVOLVE self-adaptive neural architecture framework",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db = Database()
run_repo = RunRepository(db)
event_repo = EventRepository(db)
topology_repo = TopologyHistoryRepository(db)
event_bus = EventBus()
connection_manager = ConnectionManager()

# We will store the main loop when startup happens
main_loop: Optional[asyncio.AbstractEventLoop] = None

@app.on_event("startup")
async def startup_event():
    global main_loop
    main_loop = asyncio.get_running_loop()

def _forward_event_to_ws(event: Event):
    if main_loop and main_loop.is_running():
        asyncio.run_coroutine_threadsafe(
            connection_manager.broadcast_event(event.run_id, event.to_dict()),
            main_loop
        )

event_bus.subscribe(_forward_event_to_ws)

def _save_event_to_db(event: Event):
    import uuid
    import datetime
    event_repo.save_event(
        event_id=uuid.uuid4().hex,
        run_id=event.run_id,
        event_type=event.event_type,
        step=event.step,
        phase=event.phase,
        timestamp=datetime.datetime.utcnow().isoformat(),
        payload=event.payload
    )

event_bus.subscribe(_save_event_to_db)

def _save_topology_history(event: Event):
    import uuid
    import datetime
    if event.event_type == "TOPOLOGY_HISTORY_ENTRY":
        payload = event.payload
        history_id = uuid.uuid4().hex
        
        record = {
            "history_id": history_id,
            "run_id": event.run_id,
            "level_id": payload.get("level_id"),
            "level_index": payload.get("level_index"),
            "event_id": payload.get("event_id"),
            "event_type": payload.get("event_type"),
            "mutation_type": payload.get("mutation_type"),
            "mutation_status": payload.get("mutation_status"),
            "label": payload.get("label"),
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "topology_revision": payload.get("topology_revision"),
            "performance": payload.get("performance"),
            "parameter_count": payload.get("parameter_count"),
            "active_module_count": payload.get("active_module_count"),
            "edge_count": payload.get("edge_count"),
            "topology_snapshot": payload.get("topology_snapshot"),
            "parent_history_id": payload.get("parent_history_id"),
        }
        topology_repo.save_snapshot(record)
        
        # Broadcast the metadata (without the massive snapshot) so frontend knows history updated
        meta = record.copy()
        meta.pop("topology_snapshot", None)
        meta.pop("topology_snapshot_json", None)
        
        if main_loop and main_loop.is_running():
            asyncio.run_coroutine_threadsafe(
                connection_manager.broadcast_event(event.run_id, {
                    "event_type": "LEVEL_HISTORY_UPDATED",
                    "run_id": event.run_id,
                    "timestamp": meta["timestamp"],
                    "payload": meta
                }),
                main_loop
            )

event_bus.subscribe(_save_topology_history)

active_runs: Dict[str, RunStateMachine] = {}
run_results: Dict[str, Dict[str, Any]] = {}
run_configs: Dict[str, RunCreateRequest] = {}


def _run_experiment_thread(run_id: str, experiment_type: str, seed: int, event_bus: EventBus, config: Optional[RunCreateRequest] = None):
    """Execute an experiment in a background thread."""
    import torch
    import numpy as np

    try:
        if experiment_type == "evolve":
            from backend.experiments.run import run_continuous_experiment
            levels = config.levels if config and hasattr(config, "levels") else 30
            result = run_continuous_experiment(
                run_id=run_id, 
                event_bus=event_bus, 
                num_levels=levels, 
                seed=seed, 
                max_steps=20000, 
                eval_freq=500
            )
        elif experiment_type == "fixed":
            from training.train_fixed import run_fixed_experiment
            result = run_fixed_experiment(steps_a=3000, steps_b=3000, seed=seed)
        elif experiment_type == "replay":
            from training.train_replay import run_replay_experiment
            result = run_replay_experiment(steps_a=3000, steps_b=3000, seed=seed)
        elif experiment_type == "always_grow":
            from training.train_always_grow import run_always_grow_experiment
            result = run_always_grow_experiment(steps_a=3000, steps_b=3000, seed=seed)
        elif experiment_type == "demo_hardened":
            from backend.experiments.run import run_continuous_experiment
            levels = config.levels if config and hasattr(config, "levels") else 3
            max_steps = 300
            eval_freq = 50
            if config and hasattr(config, "max_steps") and config.max_steps:
                max_steps = config.max_steps
            if config and hasattr(config, "eval_freq") and config.eval_freq:
                eval_freq = config.eval_freq
                
            result = run_continuous_experiment(
                run_id=run_id, 
                event_bus=event_bus, 
                num_levels=levels, 
                seed=seed, 
                max_steps=max_steps, 
                eval_freq=eval_freq
            )
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")

        run_results[run_id] = result

        if run_id in active_runs:
            current_state = active_runs[run_id].current_state
            if current_state == RunState.STOPPING:
                active_runs[run_id].transition_to(RunState.STOPPED)
                run_repo.update_state(run_id, RunState.STOPPED.value)
            elif current_state in [RunState.RUNNING, RunState.PAUSED, RunState.INITIALIZING]:
                event_bus.publish(Event(
                    event_type="EXPERIMENT_COMPLETED",
                    run_id=run_id,
                    step=-1,
                    phase="COMPLETE",
                    payload=result,
                ))
                active_runs[run_id].transition_to(RunState.COMPLETED)
                run_repo.update_state(run_id, RunState.COMPLETED.value)
    except TaskGenerationFailure as e:
        print(f"\n[CONTROLLED RUN FAILURE] Run {run_id} terminated due to generation exhaustion: {e}")
        event_bus.publish(Event(
            event_type="EXPERIMENT_FAILED",
            run_id=run_id,
            step=-1,
            phase="ERROR",
            payload={"error": str(e), "failure_type": "TASK_GENERATION_EXHAUSTED", "details": e.spec_dict},
        ))
        if run_id in active_runs:
            current_state = active_runs[run_id].current_state
            if current_state != RunState.FAILED:
                active_runs[run_id].transition_to(RunState.FAILED)
                run_repo.update_state(run_id, RunState.FAILED.value)
                
    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        print(f"\n[CRITICAL THREAD CRASH] Run {run_id} terminated with exception:\n{tb_str}")
        event_bus.publish(Event(
            event_type="EXPERIMENT_FAILED",
            run_id=run_id,
            step=-1,
            phase="ERROR",
            payload={"error": str(e), "traceback": tb_str},
        ))
        if run_id in active_runs:
            current_state = active_runs[run_id].current_state
            if current_state == RunState.STOPPING:
                # Thread crashed while stopping, settle gracefully
                active_runs[run_id].transition_to(RunState.STOPPED)
                run_repo.update_state(run_id, RunState.STOPPED.value)
            elif current_state != RunState.FAILED:
                active_runs[run_id].transition_to(RunState.FAILED)
                run_repo.update_state(run_id, RunState.FAILED.value)


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "EVOLVE Backend", "timestamp": datetime.datetime.utcnow().isoformat()}



@app.get("/brain/topology")
def get_brain_topology():
    from brain.brain import ModularBrain
    # Instantiate a fresh brain just to get the initial topology schema
    brain = ModularBrain(obs_dim=16, latent_dim=32, module_dim=64, action_dim=5, initial_module_id="module_A")
    return brain.topology_snapshot()


@app.get("/world/preview")
def get_world_preview():
    from environment.adaptive_frontier.world import AdaptiveFrontierWorld
    from environment.adaptive_frontier.difficulty import get_level_spec
    
    # Get the level 1 curriculum spec
    spec = get_level_spec(1, seed=42)
    
    # Generate the initial world state
    env = AdaptiveFrontierWorld(spec)
    
    return env.get_state()


@app.post("/runs", response_model=RunResponse)
def create_run(req: RunCreateRequest):
    print(f"[EVOLVE API] POST /runs called. mode={req.mode}")
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    created_at = datetime.datetime.utcnow().isoformat()
    sm = RunStateMachine(RunState.CREATED)
    active_runs[run_id] = sm
    run_configs[run_id] = req

    run_repo.create_run(
        run_id=run_id,
        mode=req.mode,
        seed=req.seed,
        experiment_type=req.experiment_type,
        state=sm.current_state.value,
        created_at=created_at,
    )

    return RunResponse(
        run_id=run_id,
        state=sm.current_state.value,
        mode=req.mode,
        seed=req.seed,
        experiment_type=req.experiment_type,
        created_at=created_at,
    )


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    run = run_repo.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    # Attach results if available
    if run_id in run_results:
        run["results"] = run_results[run_id]
    return run


@app.post("/runs/{run_id}/start")
def start_run(run_id: str):
    print(f"[EVOLVE API] POST /runs/{run_id}/start called.")
    if run_id not in active_runs:
        run_data = run_repo.get_run(run_id)
        if not run_data:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
        active_runs[run_id] = RunStateMachine(RunState(run_data["state"]))

    sm = active_runs[run_id]
    run_data = run_repo.get_run(run_id)

    if sm.current_state == RunState.CREATED:
        sm.transition_to(RunState.INITIALIZING)
        sm.transition_to(RunState.RUNNING)

        # Start experiment in background thread
        experiment_type = run_data.get("experiment_type", "evolve")
        seed = run_data.get("seed", 42)
        config = run_configs.get(run_id)
        
        # Initialize control synchronously
        from backend.experiments.run import RunControl, run_controls
        if run_id not in run_controls:
            run_controls[run_id] = RunControl()
            
        print(f"[EVOLVE RUN] start authorized for run_id: {run_id}")
        thread = threading.Thread(
            target=_run_experiment_thread,
            args=(run_id, experiment_type, seed, event_bus, config),
            daemon=True,
        )
        thread.start()

    elif sm.current_state == RunState.PAUSED:
        sm.transition_to(RunState.RUNNING)

    run_repo.update_state(run_id, sm.current_state.value)
    return {"run_id": run_id, "state": sm.current_state.value}


@app.post("/runs/{run_id}/stop")
def stop_run(run_id: str):
    if run_id not in active_runs:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    sm = active_runs[run_id]
    
    if run_id in run_controls:
        run_controls[run_id].stop_requested.set()

    if sm.current_state in [RunState.RUNNING, RunState.PAUSED]:
        sm.transition_to(RunState.STOPPING)
        run_repo.update_state(run_id, sm.current_state.value)
    return {"run_id": run_id, "state": sm.current_state.value}


@app.get("/runs/{run_id}/events")
def get_run_events(run_id: str):
    return event_repo.get_events(run_id)

@app.get("/runs/{run_id}/history")
def get_run_history(run_id: str):
    levels = topology_repo.get_levels_for_run(run_id)
    return {
        "run_id": run_id,
        "levels": levels
    }

@app.get("/runs/{run_id}/history/{level_id}")
def get_level_history(run_id: str, level_id: str):
    entries = topology_repo.get_history_for_level(run_id, level_id)
    return {
        "run_id": run_id,
        "level_id": level_id,
        "entries": entries
    }

@app.get("/runs/{run_id}/history/entry/{history_id}")
def get_history_entry(run_id: str, history_id: str):
    snapshot = topology_repo.get_snapshot(run_id, history_id)
    if not snapshot:
        raise HTTPException(status_code=404, detail=f"History entry '{history_id}' not found.")
    return snapshot



@app.get("/runs/{run_id}/results")
def get_run_results(run_id: str):
    if run_id not in run_results:
        raise HTTPException(status_code=404, detail=f"Results for run '{run_id}' not found (experiment may still be running).")
    return run_results[run_id]


@app.websocket("/runs/{run_id}/stream")
async def websocket_stream(websocket: WebSocket, run_id: str):
    await connection_manager.connect(run_id, websocket)
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        connection_manager.disconnect(run_id, websocket)
