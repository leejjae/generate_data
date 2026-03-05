# Dataset Generation

This document describes how to generate and augment the datasets in `./data_alfworld/` (ALFWorld) and `./data_virtualhome/` (VirtualHome).

All scripts are located under `generation/`.

---

## ALFWorld (`./data_alfworld/`)

### Prerequisites

**Step 2 requires AI2-THOR**, which runs a Unity binary and needs an X display. On a headless server, start a virtual framebuffer first:

```bash
sudo apt-get install -y xvfb
Xvfb :1 -screen 0 1024x768x24 &
export DISPLAY=:1
```

ALFWorld data (JSON trajectories + AI2-THOR binary) is downloaded automatically on first run via `alfworld-download`. To download manually:

```bash
alfworld-download --extra
```

### Step 1: Collect expert trajectories

```bash
cd generation
python collect_alfworld_trajectories.py ../configs/base_config.yaml \
    --output_path ../data_alfworld/alfworld_trajectories.jsonl \
    --split train
```

Runs the ALFWorld text environment (no display required) with the built-in expert plan to collect (observation, action, next\_observation) step sequences. Only successfully terminated episodes are saved. Supports resuming from an existing output file.

### Step 2: Augment with environment graphs

```bash
cd generation
python generate_alfworld_data.py \
    --original_path ../data_alfworld/alfworld_trajectories.jsonl \
    --output_path ../data_alfworld/alfworld_augmented.jsonl
```

Replays each trajectory inside the AI2-THOR Unity simulator (requires `DISPLAY`) and injects an `observation_graph` field — a symbolic object-relation graph of the full environment — into every step.

---

## VirtualHome (`./data_virtualhome/`)

### Prerequisites

The VirtualHome Unity simulator must be run in headless mode. Set up a virtual framebuffer if not already running:

```bash
sudo apt-get install -y xvfb
Xvfb :1 -screen 0 1024x768x24 &
```

Download the VirtualHome Linux simulator binary (`linux_exec.v2.3.0.x86_64`) from the [VirtualHome releases](http://virtual-home.org/release/simulator/v2.0/v2.3.0/linux_exec.v2.3.0.zip) and place it under `simulation/`.

### Step 1: Start the Unity simulator

```bash
DISPLAY=:1 nohup ./simulation/linux_exec.v2.3.0.x86_64 \
    -batchmode -port 8080 -force-opengl \
    > /tmp/vh_sim.log 2>&1 &
# Wait until "Waiting for request" appears in:
#   ~/.config/unity3d/VirtualHome/VirtualHome/Player.log
```

### Step 2: Generate trajectories

```bash
cd generation
python generate_virtualhome_data.py \
    --output_dir ../data_virtualhome/ \
    --port 8080
```

Runs an `ExpertPolicy` agent across all 78 tasks x 20 house environments. For each (task, env) pair it resets the scene, executes the expert script (with automatic room-navigation steps inserted), and saves per-step records containing the action, position graph, visible graph, and agent graph. Skips already-completed files and auto-restarts the simulator on timeout.