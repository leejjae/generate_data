"""Generate VirtualHome trajectory data for TMoW training.

This script runs the ExpertPolicy in VirtualHome Unity simulator across all
(task_id, env_id) combinations and saves JSONL trajectory files compatible
with VirtualHomeDatasetGenerator._load_data().

Usage:
    # Start Unity simulator first (externally, from the simulation/ directory):
    #   DISPLAY=:1 nohup ./linux_exec.v2.3.0.x86_64 -batchmode -port 8080 -force-opengl > /tmp/vh_sim.log 2>&1 &
    #   # Wait for "Waiting for request" in Player.log before running this script
    python -m tmow.scripts.generate_virtualhome_data --output_dir /path/to/trajectories

JSONL format (one line per step):
    {
        "env_id": int,            # House ID (from VALID_HOUSE)
        "task_id": int,           # Task index in TASKS list
        "instruction": str,       # Task name e.g. "Turn on tv"
        "action": str,            # Action taken e.g. "walk tv"
        "position_graph": dict,   # Full position graph (nodes + edges) - same for all steps
        "visible_graph": dict,    # Visible objects graph before action
        "agent_graph": dict,      # Agent relation graph before action
        "next_visible_graph": dict,  # Visible objects graph after action
        "next_agent_graph": dict,    # Agent relation graph after action
    }
"""

import argparse
import json
import os
import subprocess
import time
import traceback
import types
from pathlib import Path

from virtualhome.environment.high_level_environment import (
    ExpertPolicy,
)
from virtualhome.environment.unity_environment import (
    UnityEnvironment,
)
from utils.virtualhome.const import TASKS, TASKS_SET

# All valid house IDs used in the TMoW paper (20 scenes: 10 seen + 10 unseen)
VALID_HOUSE = list(
    map(int, "0 1 5 6 7 8 9 12 13 15 18 20 22 24 26 28 29 31 32 34".split())
)

# Simulator binary and log paths (used for automatic restart on timeout)
_SIM_EXEC = (
    "/workspace/tmow/tmow/environments/virtualhome/virtualhome/simulation/"
    "linux_exec.v2.3.0.x86_64"
)
_SIM_LOG = "/tmp/vh_sim.log"
_PLAYER_LOG = "/root/.config/unity3d/VirtualHome/VirtualHome/Player.log"


def find_object_room(env: UnityEnvironment, obj_name: str) -> str | None:
    """Return the room name that obj_name is currently placed in (or None).

    Traverses the containment hierarchy (e.g. apple INSIDE cabinet INSIDE kitchen)
    using BFS to find the room even when the object is nested inside containers.
    """
    rooms = env.rooms  # {room_id: room_class_name}
    graph = env.get_graph()

    # Build an INSIDE parent map: child_id -> parent_id
    inside_parent: dict[int, int] = {}
    for edge in graph["edges"]:
        if edge["relation_type"] == "INSIDE":
            inside_parent[edge["from_id"]] = edge["to_id"]

    # Find object node IDs matching obj_name
    target_ids = [
        node["id"]
        for node in graph["nodes"]
        if node.get("class_name", "").lower() == obj_name.lower()
    ]

    # BFS up the containment tree from each candidate
    for start_id in target_ids:
        node_id = start_id
        visited: set[int] = set()
        while node_id not in visited:
            visited.add(node_id)
            if node_id in rooms:
                return rooms[node_id].lower()
            parent = inside_parent.get(node_id)
            if parent is None:
                break
            node_id = parent
    return None


def task_to_script(task_name: str) -> list[str]:
    """Convert a task name to a high-level ExpertPolicy script.

    Returns a list of instructions like ["walk tv", "switchon tv"].
    """
    if task_name.startswith("Turn on "):
        obj = task_name[8:].strip()
        return ["walk " + obj, "switchon " + obj]

    if task_name.startswith("Open "):
        obj = task_name[5:].strip()
        return ["walk " + obj, "open " + obj]

    if task_name.startswith("Put "):
        rest = task_name[4:].strip()
        for sep in [" on ", " to "]:
            if sep in rest:
                obj, target = rest.split(sep, 1)
                return [
                    "walk " + obj.strip(),
                    "grab " + obj.strip(),
                    "walk " + target.strip(),
                    "put " + target.strip(),
                ]
        raise ValueError(f"Unknown Put task format: {task_name!r}")

    if task_name.startswith("Place "):
        rest = task_name[6:].strip()
        if " in " not in rest:
            raise ValueError(f"Unknown Place task format: {task_name!r}")
        obj, container = rest.split(" in ", 1)
        obj, container = obj.strip(), container.strip()
        # Walk to container first to open it, then grab object, then put inside
        return [
            "walk " + container,
            "open " + container,
            "walk " + obj,
            "grab " + obj,
            "walk " + container,
            "putin " + container,
        ]

    raise ValueError(f"Unknown task format: {task_name!r}")


def _get_available_objects(env: UnityEnvironment) -> dict:
    """Return a name→id dict of all objects accessible to the agent.

    Combines:
    - Same-room objects + adjacent room IDs  (for 'walk' instructions)
    - Close objects                           (for interaction instructions)
    """
    name_to_id: dict = {}
    try:
        for obj_id in env.get_same_room_objects():
            if obj_id in env.id2node:
                name_to_id[env.id2node[obj_id]["class_name"]] = obj_id
    except Exception:
        pass
    try:
        for obj_id in env.get_close_objects():
            if obj_id in env.id2node:
                name_to_id[env.id2node[obj_id]["class_name"]] = obj_id
    except Exception:
        pass
    return name_to_id


_ROOMS = {"kitchen", "bedroom", "livingroom", "bathroom"}


def _make_script(task_name: str, env: UnityEnvironment) -> list[str]:
    """Build an ExpertPolicy script with room-navigation instructions inserted.

    ExpertPolicy.step() checks 'get_same_room_objects()' to decide if a walk
    target is reachable.  Objects in other rooms are invisible to this check,
    causing the agent to spin indefinitely.  This function inserts
    'walk <room>' steps before each 'walk <object>' instruction so the
    ExpertPolicy's walk_room logic navigates there first (transitively through
    adjacent rooms when needed).

    Example — "Place apple in fridge", apple in bedroom, fridge in kitchen:
        base:     walk fridge | open fridge | walk apple | grab apple | walk fridge | putin fridge
        enhanced: walk kitchen | walk fridge | open fridge |
                  walk bedroom | walk apple | grab apple |
                  walk kitchen | walk fridge | putin fridge
    """
    base_script = task_to_script(task_name)
    enhanced: list[str] = []
    current_room: str | None = None  # track where agent is headed

    for instruction in base_script:
        parts = instruction.split()
        action, target = parts[0], parts[1] if len(parts) > 1 else ""

        if action == "walk" and target not in _ROOMS:
            # Object navigation: look up the object's room and prepend if needed
            obj_room = find_object_room(env, target)
            if obj_room and obj_room != current_room:
                enhanced.append("walk " + obj_room)
                current_room = obj_room

        elif action == "walk" and target in _ROOMS:
            current_room = target

        enhanced.append(instruction)

    return enhanced


def run_episode(
    env: UnityEnvironment,
    task_name: str,
    task_id: int,
    env_id: int,
    position_graph: dict,
    initial_obs: dict,
    max_steps: int = 100,
) -> list[dict]:
    """Run one episode with ExpertPolicy; return trajectory steps."""
    script = _make_script(task_name, env)
    expert = ExpertPolicy()
    expert.reset(script)

    trajectory: list[dict] = []
    obs = initial_obs  # obs from env.reset() or previous env.step()

    for _step in range(max_steps):
        # Pre-action graphs in json mode (for JSONL storage)
        visible_graph_json = obs.get("visible_graph") or {"nodes": [], "edges": []}
        agent_graph_json = obs.get("agent_graph") or {"nodes": [], "edges": []}

        # Triples mode for ExpertPolicy (re-uses Unity's cached graph, no extra calls)
        visible_graph_triples = (
            env.get_visible_graph(mode="triples") or {"nodes": [], "edges": []}
        )
        agent_graph_triples = env.get_agent_graph(mode="triples")

        # Feed available objects to the expert
        name_to_id = _get_available_objects(env)
        expert.set_available_objects(name_to_id)

        # Expert decides action
        action = expert.step(visible_graph_triples, agent_graph_triples)
        if action is None:
            # Script finished
            break

        # Execute action in the simulator
        obs, _reward, done, _info = env.step(action)

        # Post-action graphs
        next_visible_graph = obs.get("visible_graph") or {"nodes": [], "edges": []}
        next_agent_graph = obs.get("agent_graph") or {"nodes": [], "edges": []}

        trajectory.append(
            {
                "env_id": env_id,
                "task_id": task_id,
                "instruction": task_name,
                "action": action,
                "position_graph": position_graph,
                "visible_graph": visible_graph_json,
                "agent_graph": agent_graph_json,
                "next_visible_graph": next_visible_graph,
                "next_agent_graph": next_agent_graph,
            }
        )

        if done:
            break

    return trajectory


def _is_timeout_error(e: Exception) -> bool:
    """Return True if the exception indicates the simulator is hung/unresponsive."""
    # UnityEngineException(408, ...) — simulator returned HTTP 408
    if hasattr(e, "args") and e.args and e.args[0] == 408:
        return True
    msg = str(e).lower()
    return "timed out" in msg or "timeout" in msg or "connectionerror" in msg


def _make_env(port: int, url: str) -> UnityEnvironment:
    """Create a UnityEnvironment with the standard timeout and no-image patch."""
    env = UnityEnvironment(url=url, base_port=port)
    env.comm.timeout_wait = 600

    def _obs_no_image(self):
        return {
            "image": None,
            "visible_graph": self.get_visible_graph(),
            "agent_graph": self.get_agent_graph(),
        }

    env.get_observations = types.MethodType(_obs_no_image, env)
    return env


def restart_simulator(
    env: UnityEnvironment | None,
    port: int = 8080,
    url: str = "localhost",
    verbose: bool = True,
) -> UnityEnvironment:
    """Kill the hung simulator, restart it, and return a fresh UnityEnvironment.

    Steps:
      1. Close the old env (best-effort).
      2. Kill the simulator process with pkill.
      3. Launch a new simulator instance.
      4. Wait up to 5 min for "Waiting for request" to appear in Player.log.
      5. Create a fresh env, apply patches, do a warm-up reset, and return it.
    """
    if verbose:
        print("    [restart] Killing stuck Unity simulator...")

    # Close old env gracefully (ignore errors — it may already be dead)
    if env is not None:
        try:
            env.close()
        except Exception:
            pass

    subprocess.run(["pkill", "-f", "linux_exec.v2.3.0.x86_64"], capture_output=True)
    time.sleep(5)  # Let the process die fully

    # Record current player log size so we can detect a *new* "Waiting for request"
    log_size_before = 0
    try:
        log_size_before = os.path.getsize(_PLAYER_LOG)
    except FileNotFoundError:
        pass

    # Launch new simulator instance
    if verbose:
        print("    [restart] Starting new simulator instance...")
    sim_env = {**os.environ, "DISPLAY": ":1"}
    with open(_SIM_LOG, "a") as log_f:
        subprocess.Popen(
            [_SIM_EXEC, "-batchmode", "-port", str(port), "-force-opengl"],
            stdout=log_f,
            stderr=log_f,
            env=sim_env,
            start_new_session=True,
        )

    # Poll Player.log for "Waiting for request" (up to 5 minutes)
    if verbose:
        print("    [restart] Waiting for simulator to be ready (up to 5 min)...")
    deadline = time.time() + 300
    ready = False
    while time.time() < deadline:
        time.sleep(3)
        try:
            with open(_PLAYER_LOG, "rb") as f:
                f.seek(log_size_before)
                new_content = f.read().decode("utf-8", errors="replace")
                if "Waiting for request" in new_content:
                    ready = True
                    break
        except FileNotFoundError:
            pass

    if not ready:
        raise RuntimeError("Simulator did not become ready within 5 min after restart")

    if verbose:
        print("    [restart] Simulator ready. Creating new environment...")

    new_env = _make_env(port, url)

    # Warm-up: the first scene reset loads GPU assets and can be slow
    try:
        new_env.set_task({"required_condition": [("tv", "is", "on")], "prohibited_condition": []})
        new_env.reset(environment_id=0)
        if verbose:
            print("    [restart] Warm-up complete.")
    except Exception as warmup_e:
        if verbose:
            print(f"    [restart] Warm-up failed (non-fatal): {warmup_e}")

    return new_env


def generate_data(
    output_dir: str,
    port: int = 8080,
    url: str = "localhost",
    max_steps: int = 100,
    task_ids: list[int] | None = None,
    env_ids: list[int] | None = None,
    verbose: bool = True,
) -> None:
    """Main loop: iterate over (task, env) pairs and save JSONL trajectories."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    env = _make_env(port, url)

    # Warm-up: the very first scene reset after simulator startup loads all GPU
    # assets and can take 5+ minutes with a software renderer.  Run one dummy
    # reset so that subsequent resets complete in a reasonable time.
    if verbose:
        print("Warming up simulator (first scene load may take several minutes)...")
    try:
        env.set_task({"required_condition": [("tv", "is", "on")], "prohibited_condition": []})
        env.reset(environment_id=0)
        if verbose:
            print("Simulator warm-up complete.")
    except Exception as e:
        if verbose:
            print(f"Warm-up failed (non-fatal): {e}")

    target_tasks = task_ids if task_ids is not None else list(range(len(TASKS)))
    target_envs = env_ids if env_ids is not None else VALID_HOUSE

    total = len(target_tasks) * len(target_envs)
    count = 0

    for task_id in target_tasks:
        task_name = TASKS[task_id]
        conditions = TASKS_SET[task_name]

        task_dir = os.path.join(output_dir, task_name.replace(" ", "_"))
        Path(task_dir).mkdir(parents=True, exist_ok=True)

        for env_id in target_envs:
            count += 1
            output_file = os.path.join(task_dir, f"env{env_id:02d}.jsonl")

            if os.path.exists(output_file):
                if verbose:
                    print(f"[{count}/{total}] Skip {task_name} / env{env_id} (exists)")
                continue

            if verbose:
                print(f"[{count}/{total}] Generating: {task_name} / env{env_id}")

            # --- Reset (with one restart-and-retry on timeout) ---
            obs = None
            position_graph = None
            reset_ok = False
            for attempt in range(2):
                try:
                    env.set_task(
                        {
                            "required_condition": conditions,
                            "prohibited_condition": [],
                        }
                    )
                    obs = env.reset(environment_id=env_id)
                    position_graph = env.get_position_graph()
                    reset_ok = True
                    break

                except NotImplementedError:
                    if verbose:
                        print(f"    Skipped: task objects not found in scene {env_id}")
                    break

                except Exception as e:
                    if verbose:
                        print(f"    Reset error (attempt {attempt + 1}): {e}")
                    if attempt == 0 and _is_timeout_error(e):
                        try:
                            env = restart_simulator(env, port=port, url=url, verbose=verbose)
                        except Exception as restart_e:
                            if verbose:
                                print(f"    Restart failed: {restart_e}")
                            break
                    else:
                        traceback.print_exc()
                        break

            if not reset_ok:
                continue

            # --- Episode (with one restart-and-retry on timeout) ---
            trajectory = None
            for attempt in range(2):
                try:
                    trajectory = run_episode(
                        env=env,
                        task_name=task_name,
                        task_id=task_id,
                        env_id=env_id,
                        position_graph=position_graph,
                        initial_obs=obs,
                        max_steps=max_steps,
                    )
                    break

                except Exception as e:
                    if verbose:
                        print(f"    Episode error (attempt {attempt + 1}): {e}")
                    if attempt == 0 and _is_timeout_error(e):
                        try:
                            env = restart_simulator(env, port=port, url=url, verbose=verbose)
                            # Re-setup the scene for the retry
                            env.set_task(
                                {
                                    "required_condition": conditions,
                                    "prohibited_condition": [],
                                }
                            )
                            obs = env.reset(environment_id=env_id)
                            position_graph = env.get_position_graph()
                        except Exception as re_e:
                            if verbose:
                                print(f"    Re-setup after restart failed: {re_e}")
                            trajectory = None
                            break
                    else:
                        traceback.print_exc()
                        break

            if trajectory is None:
                continue

            if not trajectory:
                if verbose:
                    print(f"    Empty trajectory, skipping")
                continue

            with open(output_file, "w") as f:
                for entry in trajectory:
                    f.write(json.dumps(entry) + "\n")

            if verbose:
                print(f"    Saved {len(trajectory)} steps → {output_file}")

    env.close()
    if verbose:
        print(f"\nDone. Trajectories saved to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate VirtualHome trajectory data using ExpertPolicy. "
            "The Unity simulator must already be running externally."
        )
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        default="trajectories",
        help="Directory to save trajectories (default: ./trajectories)",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8080,
        help="Port of the running VirtualHome simulator (default: 8080)",
    )
    parser.add_argument(
        "--url",
        default="localhost",
        help="IP/hostname of the simulator (default: localhost)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=100,
        help="Maximum number of steps per episode (default: 100)",
    )
    parser.add_argument(
        "--task_ids",
        nargs="*",
        type=int,
        default=None,
        help="Subset of task IDs to generate (default: all 78 tasks)",
    )
    parser.add_argument(
        "--env_ids",
        nargs="*",
        type=int,
        default=None,
        help="Subset of environment IDs to generate (default: all 20 VALID_HOUSE scenes)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    args = parser.parse_args()

    generate_data(
        output_dir=args.output_dir,
        port=args.port,
        url=args.url,
        max_steps=args.max_steps,
        task_ids=args.task_ids,
        env_ids=args.env_ids,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
