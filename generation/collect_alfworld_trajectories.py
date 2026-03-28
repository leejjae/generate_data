"""
ALFWorld Text Preprocessing  (Stage 1 of 2)
============================================
Raw ALFWorld seq2seq game files → JSONL with (observation, action, next_observation).

Uses ALFWorld's AlfredTWEnv (TextWorld-based) with the HandCodedTWAgent (expert)
to replay demonstrations and record plain-text observations.
No AI2-THOR / Unity is required for this stage.

Data layout expected after `alfworld-download --extra`
-------------------------------------------------------
$ALFWORLD_DATA/   (default: ~/.cache/alfworld/)
  json_2.1.1/
    train/
      <task_type>/<trial>/traj_data.json    ← PDDL trajectories
  seq2seq_data/
    train/
      <task_type>/<trial>/traj_data.json
      <task_type>/<trial>/game.tw-pddl      ← TextWorld game files ← needed here

The config key `dataset.data_path` must point to the directory that is walked
for `traj_data.json` + `game.tw-pddl` pairs (typically `seq2seq_data/train`).
`alfworld-download --extra` normally expands the downloaded archive so that
these files are already in `$ALFWORLD_DATA/`.  Check `alfworld-download --help`
and adjust `--alfworld_root` / `ALFWORLD_DATA` accordingly.

Output JSONL schema (one object per line)
-----------------------------------------
{
  "trial_name": "<task_folder>_<trial_folder>",
  "instruction": "<task goal, e.g. 'put a cool apple in garbage can'>",
  "history": [
    {
      "observation":      "<text obs at step t>",
      "action":           "<expert action>",
      "next_observation": "<text obs at step t+1>"
    },
    ...
  ]
}

Usage
-----
    # Activate the data venv first:
    source /workspace/dataset/data/bin/activate

    python -m alfworld.preprocess \\
        --output /workspace/dataset/output/alfworld_train.jsonl \\
        --split  train \\
        [--alfworld_root ~/.cache/alfworld] \\
        [--max_episodes 500]
"""

from __future__ import annotations

import json
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Minimal AlfredTWEnv config builder
# ---------------------------------------------------------------------------

_TASK_TYPE_IDS = [1, 2, 3, 4, 5, 6]   # all six task types

_MAX_STEPS_PER_EPISODE = 100            # HandCodedAgent default max_steps is 200


def _build_config(data_path: str) -> dict:
    """
    Build the minimal config dict required by AlfredTWEnv.

    Key decisions:
    - training_method="dagger"  → enables the AlfredExpert wrapper which adds
                                   "extra.expert_plan" to each step's infos.
    - train_eval="train"        → expert_plan is computed at every step.
    - task_types=[1..6]         → include all task categories.
    - data_path                 → directory walked for traj_data.json + game.tw-pddl
    """
    return {
        "general": {
            "random_seed": 42,
            "training_method": "dagger",   # required to enable expert plan
        },
        "dataset": {
            "data_path": data_path,
            "eval_id_data_path": data_path,
            "eval_ood_data_path": data_path,
            "num_train_games": -1,   # -1 = use all
            "num_eval_games": -1,
        },
        "env": {
            "type": "AlfredTWEnv",
            "domain_randomization": False,
            "expert_type": "handcoded",
            "goal_desc_human_anns_prob": 0.0,
            "task_types": _TASK_TYPE_IDS,
        },
        "dagger": {
            "training": {
                "max_nb_steps_per_episode": _MAX_STEPS_PER_EPISODE,
            }
        },
    }


# ---------------------------------------------------------------------------
# Trial-name extraction
# ---------------------------------------------------------------------------

def _extract_trial_name(gamefile: str) -> str:
    """
    Convert a game file path to the trial name used by json_2.1.1.

    Pattern:  .../seq2seq_data/train/<task_folder>/<trial_folder>/game.tw-pddl
    Result:   <task_folder>_<trial_folder>

    Downstream generate.py then converts  "task_X_trial_T"  →  "task_X/trial_T"
    for path lookup.
    """
    parts = Path(gamefile).parts
    for i in range(len(parts) - 1, 0, -1):
        if parts[i].startswith("trial_") or parts[i].startswith("Trial"):
            trial_folder = parts[i]
            task_folder = parts[i - 1]
            return f"{task_folder}_{trial_folder}"
    # Fallback: use last two directory segments
    task_folder = parts[-3] if len(parts) >= 3 else "unknown_task"
    trial_folder = parts[-2] if len(parts) >= 2 else "unknown_trial"
    return f"{task_folder}_{trial_folder}"


# ---------------------------------------------------------------------------
# Main preprocessing function
# ---------------------------------------------------------------------------

def preprocess_alfworld(
    output_path: str,
    split: str = "train",
    alfworld_root: str | None = None,
    max_episodes: int | None = None,
    resume: bool = True,
) -> None:
    """
    Preprocess ALFWorld text game data into JSONL format.

    Each episode is played using the built-in HandCodedTWAgent (expert).
    At each step, the current observation, expert action, and resulting
    next observation are recorded.

    Parameters
    ----------
    output_path : str
        Destination JSONL file.
    split : str
        'train', 'eval_in_distribution', or 'eval_out_of_distribution'.
        (Note: for eval splits, eval_id_data_path / eval_ood_data_path must
         point to the correct directories.)
    alfworld_root : str | None
        ALFWorld data root.  Defaults to $ALFWORLD_DATA or ~/.cache/alfworld.
        The data_path passed to AlfredTWEnv will be this directory directly;
        adjust if your seq2seq game files are in a subdirectory.
    max_episodes : int | None
        Cap on episodes to process.  None = process all available.
    resume : bool
        Skip trial names already present in output_path.
    """
    try:
        import alfworld.agents.environment as alf_env
        # alfworld 0.4.2: AlfredTWEnv is NOT exported from the package root.
        # It must be retrieved via get_environment() or imported directly from
        # alfworld.agents.environment.alfred_tw_env.
        # Using alf_env.AlfredTWEnv would raise AttributeError.
        AlfredTWEnv = alf_env.get_environment("AlfredTWEnv")
    except ImportError as e:
        raise ImportError(
            "alfworld package not found.\n"
            "Activate the venv and run:\n"
            "    pip install alfworld==0.4.2"
        ) from e

    from tqdm import tqdm

    # ---- resolve data root ----
    if alfworld_root is None:
        alfworld_root = os.environ.get(
            "ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld")
        )

    # AlfredTWEnv walks data_path recursively for (traj_data.json + game.tw-pddl)
    # pairs.  alfworld-download --extra extracts seq2seq_data/ under $ALFWORLD_DATA,
    # so passing alfworld_root is correct (the walker finds files in subdirectories).
    data_path = alfworld_root

    # ---- verify data ----
    if not os.path.isdir(data_path):
        raise FileNotFoundError(
            f"Data path not found: {data_path}\n"
            "Download ALFWorld data:\n"
            "    alfworld-download\n"
            "    alfworld-download --extra   # includes seq2seq game files"
        )

    # Quick sanity check: look for at least one game.tw-pddl file
    found_game = False
    for _root, _dirs, files in os.walk(data_path):
        if "game.tw-pddl" in files:
            found_game = True
            break
    if not found_game:
        raise FileNotFoundError(
            f"No game.tw-pddl files found under {data_path}\n"
            "Run 'alfworld-download --extra' to download seq2seq game files."
        )

    # ---- resume ----
    existing_trials: set[str] = set()
    if resume and os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                try:
                    existing_trials.add(json.loads(line).get("trial_name", ""))
                except json.JSONDecodeError:
                    pass
        if existing_trials:
            print(f"[resume] {len(existing_trials)} episodes already written – skipping.")

    # ---- build env ----
    config = _build_config(data_path)
    env = AlfredTWEnv(config, train_eval=split)
    env = env.init_env(batch_size=1)

    num_games: int = getattr(env, "num_games", 9999)
    target = min(num_games, max_episodes) if max_episodes is not None else num_games
    print(f"Split '{split}': {num_games} games total | processing: {target}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    processed = skipped = failed = 0

    with open(output_path, "a", encoding="utf-8") as f_out:
        for episode_idx in tqdm(range(target), desc=f"ALFWorld {split}"):
            try:
                obs, infos = env.reset()
            except Exception as e:
                print(f"\n[ep {episode_idx}] env.reset() failed: {e}")
                failed += 1
                continue

            # ---- trial name (from extra.gamefile injected by AlfredInfos) ----
            gamefile: str | None = None
            try:
                gf_list = infos.get("extra.gamefile", [None])
                gamefile = gf_list[0] if gf_list else None
            except (TypeError, IndexError):
                pass

            trial_name = (
                _extract_trial_name(gamefile)
                if gamefile
                else f"episode_{episode_idx:05d}"
            )

            if trial_name in existing_trials:
                skipped += 1
                continue

            # ---- instruction (task goal) ----
            # TextWorld stores the task objective in infos["objective"].
            # ALFWorld also exposes it via infos["extra.task_desc"] as a fallback.
            instruction: str = ""
            try:
                obj_list = infos.get("objective", [None])
                instruction = (obj_list[0] or "").strip()
            except (TypeError, IndexError):
                pass
            if not instruction:
                try:
                    td_list = infos.get("extra.task_desc", [None])
                    instruction = (td_list[0] or "").strip()
                except (TypeError, IndexError):
                    pass

            # ---- step through episode with expert ----
            history: list[dict] = []
            current_obs = obs[0].strip()
            done = False
            steps = 0

            while not done and steps < _MAX_STEPS_PER_EPISODE:
                # "extra.expert_plan" is a list-of-lists (one per batch item).
                # Each inner list contains exactly ONE suggested action.
                expert_plan_batch = infos.get("extra.expert_plan", [["look"]])
                expert_action_list = expert_plan_batch[0] if expert_plan_batch else ["look"]
                expert_action = expert_action_list[0] if expert_action_list else "look"

                try:
                    obs_next, _score, dones, infos = env.step([expert_action])
                except Exception as e:
                    print(f"\n[ep {episode_idx}] env.step() failed at step {steps}: {e}")
                    break

                next_obs = obs_next[0].strip()
                history.append(
                    {
                        "observation": current_obs,
                        "action": expert_action,
                        "next_observation": next_obs,
                    }
                )
                current_obs = next_obs
                done = dones[0]
                steps += 1

            if history:
                record = {"trial_name": trial_name, "instruction": instruction, "history": history}
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                existing_trials.add(trial_name)
                processed += 1
            else:
                failed += 1

    print(
        f"\nDone. Processed: {processed} | Skipped: {skipped} | Failed: {failed}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description=(
            "ALFWorld Stage 1 – preprocess text game data into\n"
            "(observation, action, next_observation) JSONL format.\n\n"
            "Prerequisites:\n"
            "  1. source /workspace/dataset/data/bin/activate\n"
            "  2. alfworld-download\n"
            "  3. alfworld-download --extra   (seq2seq game files)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output JSONL path (e.g. output/alfworld_train.jsonl)",
    )
    parser.add_argument(
        "--split",
        choices=["train", "eval_in_distribution", "eval_out_of_distribution"],
        default="train",
    )
    parser.add_argument(
        "--alfworld_root", default=None,
        help=(
            "ALFWorld data root (default: $ALFWORLD_DATA or ~/.cache/alfworld).\n"
            "Must contain game.tw-pddl files (seq2seq format)."
        ),
    )
    parser.add_argument(
        "--max_episodes", type=int, default=None,
        help="Cap on episodes to process (default: all)",
    )
    parser.add_argument(
        "--no_resume", action="store_true",
        help="Start fresh, ignoring any existing output file",
    )
    args = parser.parse_args()

    preprocess_alfworld(
        output_path=args.output,
        split=args.split,
        alfworld_root=args.alfworld_root,
        max_episodes=args.max_episodes,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
