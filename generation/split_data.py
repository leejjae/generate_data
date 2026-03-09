"""Split ALFWorld and VirtualHome datasets into seen/unseen domains.

Definitions
-----------
ALFWorld
  Seen tasks   (type 1,2,3,6): pick_and_place_simple, look_at_obj_in_light,
                                pick_clean_then_place_in_recep, pick_two_obj_and_place
  Unseen tasks (type 4,5):     pick_heat_then_place_in_recep, pick_cool_then_place_in_recep
  Seen scenes:   bedroom (301-330), bathroom (401-430), living_room (201-230)
  Unseen scenes: kitchen (1-30)

VirtualHome
  Seen tasks   (16): indices [0,4,9,14,19,24,29,34,39,44,49,54,59,64,69,74]
  Unseen tasks (62): remaining indices
  Seen domain  (10 envs): [18,20,22,24,26,28,29,31,32,34]
  Unseen domain(10 envs): [0,1,5,6,7,8,9,12,13,15]

Output
------
  data_alfworld/
    alfworld_trajectories_seen_domain.jsonl
    alfworld_trajectories_unseen_domain.jsonl
    alfworld_augmented_seen_domain.jsonl       (if source exists)
    alfworld_augmented_unseen_domain.jsonl     (if source exists)

  data_virtualhome/
    seen_domain/<task_dir>/env{id}.jsonl
    unseen_domain/<task_dir>/env{id}.jsonl

Usage
-----
  cd generation
  python split_data.py
"""

import json
import os
import re
import shutil
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))
from utils.virtualhome.const import TASKS, SEEN_TASKS, UNSEEN_TASKS, SEEN_DOMAIN, UNSEEN_DOMAIN


# ── ALFWorld constants ────────────────────────────────────────────────────────

ALFWORLD_SEEN_TASK_TYPES = {
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_two_obj_and_place",
}
ALFWORLD_UNSEEN_TASK_TYPES = {
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
}
ALFWORLD_SEEN_SCENES = {"bedroom", "bathroom", "living_room"}
ALFWORLD_UNSEEN_SCENES = {"kitchen"}


def _alfworld_task_type(trial_name: str) -> str:
    return trial_name.split("-")[0]


def _alfworld_scene(trial_name: str) -> str:
    m = re.search(r"-(\d+)_trial_", trial_name)
    if not m:
        return "unknown"
    scene = int(m.group(1))
    if 1 <= scene <= 30:
        return "kitchen"
    elif 201 <= scene <= 230:
        return "living_room"
    elif 301 <= scene <= 330:
        return "bedroom"
    elif 401 <= scene <= 430:
        return "bathroom"
    return "unknown"


def split_alfworld(data_dir: Path) -> None:
    source_files = [
        "alfworld_trajectories.jsonl",
        "alfworld_augmented.jsonl",
    ]

    for filename in source_files:
        src = data_dir / filename
        if not src.exists():
            print(f"  [skip] {filename} not found")
            continue

        stem = src.stem
        seen_path   = data_dir / f"{stem}_seen_domain.jsonl"
        unseen_path = data_dir / f"{stem}_unseen_domain.jsonl"

        seen_count = unseen_count = skip_count = 0
        with open(src) as f_in, \
             open(seen_path, "w") as f_seen, \
             open(unseen_path, "w") as f_unseen:
            for line in f_in:
                d = json.loads(line)
                task_type = _alfworld_task_type(d["trial_name"])
                scene     = _alfworld_scene(d["trial_name"])

                if task_type in ALFWORLD_SEEN_TASK_TYPES and scene in ALFWORLD_SEEN_SCENES:
                    f_seen.write(line)
                    seen_count += 1
                elif task_type in ALFWORLD_UNSEEN_TASK_TYPES and scene in ALFWORLD_UNSEEN_SCENES:
                    f_unseen.write(line)
                    unseen_count += 1
                else:
                    skip_count += 1

        print(f"  {filename}")
        print(f"    seen_domain:   {seen_count:4d} samples → {seen_path.name}")
        print(f"    unseen_domain: {unseen_count:4d} samples → {unseen_path.name}")
        print(f"    skipped:       {skip_count:4d} (mixed task/scene combinations)")


# ── VirtualHome constants ─────────────────────────────────────────────────────

SEEN_TASK_NAMES   = {TASKS[i].replace(" ", "_") for i in SEEN_TASKS}
UNSEEN_TASK_NAMES = {TASKS[i].replace(" ", "_") for i in UNSEEN_TASKS}
SEEN_DOMAIN_SET   = set(SEEN_DOMAIN)
UNSEEN_DOMAIN_SET = set(UNSEEN_DOMAIN)


def _env_id(fname: str) -> int:
    return int(re.search(r"\d+", fname).group())


def split_virtualhome(data_dir: Path) -> None:
    traj_dir  = data_dir
    seen_dir  = data_dir / "seen_domain"
    unseen_dir = data_dir / "unseen_domain"

    seen_dir.mkdir(exist_ok=True)
    unseen_dir.mkdir(exist_ok=True)

    seen_count = unseen_count = 0
    for task_dir_name in sorted(os.listdir(traj_dir)):
        task_path = traj_dir / task_dir_name
        if not task_path.is_dir():
            continue

        is_seen   = task_dir_name in SEEN_TASK_NAMES
        is_unseen = task_dir_name in UNSEEN_TASK_NAMES

        for fname in os.listdir(task_path):
            if not fname.endswith(".jsonl"):
                continue
            env_id = _env_id(fname)
            src    = task_path / fname

            if is_seen and env_id in SEEN_DOMAIN_SET:
                dst_dir = seen_dir / task_dir_name
                dst_dir.mkdir(exist_ok=True)
                shutil.copy2(src, dst_dir / fname)
                seen_count += 1
            elif is_unseen and env_id in UNSEEN_DOMAIN_SET:
                dst_dir = unseen_dir / task_dir_name
                dst_dir.mkdir(exist_ok=True)
                shutil.copy2(src, dst_dir / fname)
                unseen_count += 1

    print(f"  seen_domain:   {seen_count} files → {seen_dir}")
    print(f"  unseen_domain: {unseen_count} files → {unseen_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    root = Path(__file__).parent.parent

    print("=== ALFWorld ===")
    split_alfworld(root / "data_alfworld")

    print("\n=== VirtualHome ===")
    split_virtualhome(root / "data_virtualhome")

    print("\nDone.")


if __name__ == "__main__":
    main()