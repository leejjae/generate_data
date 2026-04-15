# Dataset Generation

This document describes how to generate and augment the datasets in `./data_alfworld/` (ALFWorld) and `./data_virtualhome/` (VirtualHome).

All scripts are located under `generation/`.

---

## Environment Setup

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install dependencies

```bash
uv sync
```

`uv sync` reads `pyproject.toml`, creates `.venv/` with Python 3.12, and installs all
dependencies — including the PyTorch CUDA 12.8 build — in a single locked step.
The `virtualhome` package pins `opencv-python==4.5.1.48` and `tqdm==4.31.1` in its
metadata; the `[tool.uv] override-dependencies` entries in `pyproject.toml` replace
those pins with compatible versions automatically, so no `--no-deps` workaround is
needed. Additionally, `visdom==0.2.4` (via `alfworld[full]`) requires `setuptools`
at build time; the `[tool.uv.extra-build-dependencies]` section handles this.

---

## ALFWorld (`./data_alfworld/`)

### Prerequisites

**Step 2 requires AI2-THOR**, which runs a Unity binary and needs an X display.
On a headless server, start a virtual framebuffer first:

```bash
sudo apt-get install -y xvfb
Xvfb :1 -screen 0 1024x768x24 &
export DISPLAY=:1
```

ALFWorld data (JSON trajectories + AI2-THOR binary) is downloaded automatically on
first run via `alfworld-download`. To download manually:

```bash
uv run alfworld-download --extra
```

### Step 1: Collect expert trajectories

```bash
cd generation
uv run python collect_alfworld_trajectories.py \
    --output ../data_alfworld/alfworld_trajectories.jsonl \
    --split train
```

Runs the ALFWorld text environment (no display required) with the built-in
HandCodedTWAgent expert to collect `(observation, action, next_observation)` step
sequences. Only complete episodes are saved. Supports resuming from an existing output
file via the `--output` path.

Additional options:

| Flag | Default | Description |
|------|---------|-------------|
| `--alfworld_root PATH` | `$ALFWORLD_DATA` or `~/.cache/alfworld` | ALFWorld data root |
| `--max_episodes N` | all | Cap on episodes to process |
| `--no_resume` | — | Start fresh, ignoring existing output |

### Step 2: Augment with environment graphs

```bash
cd generation
uv run python generate_alfworld_data.py \
    --original_path ../data_alfworld/alfworld_trajectories.jsonl \
    --output_path ../data_alfworld/alfworld_augmented.jsonl
```

Replays each trajectory inside the AI2-THOR Unity simulator (requires `DISPLAY`) and
injects an `observation_graph` field — a symbolic object-relation graph of the full
environment — into every step.

> **Note:** This step is very slow (~75 s per episode) because each episode requires
> a full Unity scene reset plus per-step graph extraction. For the full 4027-episode
> training set, expect roughly **84 hours** of wall-clock time. The script supports
> resuming: it skips episodes already present in the output file.

---

## VirtualHome (`./data_virtualhome/`)

### Prerequisites

**Download the raw VirtualHome dataset** into `./raw_dataset/` (used for downstream
training and evaluation):

```bash
bash virtualhome/helper_scripts/download_dataset.sh
```

This downloads and unzips the VirtualHome program dataset
(`programs_processed_precond_nograb_morepreconds`) under `./raw_dataset/`.

The VirtualHome Unity simulator must be run in headless mode. Set up a virtual
framebuffer if not already running:

```bash
sudo apt-get install -y xvfb
Xvfb :1 -screen 0 1024x768x24 &
```

Download the VirtualHome Linux simulator binary (`linux_exec.v2.3.0.x86_64`) from the
[VirtualHome releases](http://virtual-home.org/release/simulator/v2.0/v2.3.0/linux_exec.zip)
and place it under `./simulation/` (project root).

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
uv run python generate_virtualhome_data.py \
    --output_dir ../data_virtualhome/ \
    --port 8080
```

Runs an `ExpertPolicy` agent across all 78 tasks × 20 house environments. For each
`(task, env)` pair it resets the scene, executes the expert script (with automatic
room-navigation steps inserted), and saves per-step records containing the action,
position graph, visible graph, and agent graph. Skips already-completed files and
auto-restarts the simulator on timeout.

---

## Split into Seen / Unseen Domains

After generating both ALFWorld and VirtualHome data, split each dataset into
seen/unseen task and scene domains:

```bash
cd generation
uv run python split_data.py
```

**ALFWorld** splits `data_alfworld/alfworld_trajectories.jsonl` and
`data_alfworld/alfworld_augmented.jsonl` (if it exists) by task type and scene:

| Domain | Task types | Scenes |
|--------|-----------|--------|
| Seen | pick_and_place_simple, look_at_obj_in_light, pick_clean_then_place_in_recep, pick_two_obj_and_place | bedroom, bathroom, living_room |
| Unseen | pick_heat_then_place_in_recep, pick_cool_then_place_in_recep | kitchen |

Output: `data_alfworld/alfworld_trajectories_seen_domain.jsonl`,
`data_alfworld/alfworld_trajectories_unseen_domain.jsonl` (and equivalents for
`alfworld_augmented`).

**VirtualHome** copies files from `data_virtualhome/<task>/env{id}.jsonl` into
`seen_domain/` or `unseen_domain/` subdirectories based on task index and environment
ID:

| Domain | Task indices (16 seen / 62 unseen) | Env IDs |
|--------|-----------------------------------|---------|
| Seen | every 5th task starting at 0: [0,4,9,...,74] | 18,20,22,24,26,28,29,31,32,34 |
| Unseen | remaining 62 tasks | 0,1,5,6,7,8,9,12,13,15 |

Output: `data_virtualhome/seen_domain/<task>/env{id}.jsonl`,
`data_virtualhome/unseen_domain/<task>/env{id}.jsonl`.

---

## Troubleshooting

### `networkx` / `fractions.gcd` ImportError

`textworld` (an ALFWorld dependency) pulls in `networkx==2.3`, which is incompatible
with Python 3.12+ (`fractions.gcd` was removed in Python 3.9). The
`[tool.uv] override-dependencies` section in `pyproject.toml` forces
`networkx>=3.0` to resolve this. If you see:

```
ImportError: cannot import name 'gcd' from 'fractions'
```

make sure `pyproject.toml` includes `"networkx>=3.0"` in `override-dependencies`
and re-run `uv sync`.

### Step 2 `FileNotFoundError` for `traj_data.json`

The expert trajectories collected in Step 1 may reference trials from multiple
`json_2.1.1` splits (`train`, `valid_seen`, `valid_train`, `valid_unseen`), not
just `train`. `generate_alfworld_data.py` searches all four splits when loading
trajectory data. If you encounter a `FileNotFoundError`, ensure `alfworld-download
--extra` has completed successfully and all splits are present under
`$ALFWORLD_DATA/json_2.1.1/`.
