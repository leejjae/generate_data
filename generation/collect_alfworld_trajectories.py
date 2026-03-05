import copy
import json
import os
from os.path import join as pjoin

import numpy as np

from alfworld.info import ALFWORLD_DATA
from alfworld.agents.environment import get_environment
import alfworld.agents.modules.generic as generic
from alfworld.agents.agent import TextDAggerAgent

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MAX_STEPS = 200


def collect_trajectories(output_path: str, split: str = "train", task_types: list = None):
    if task_types is None:
        task_types = [1, 2, 3, 4, 5, 6]

    config = generic.load_config()
    config['general']['training']['batch_size'] = 1
    config['general']['evaluate']['batch_size'] = 1
    config['general']['observation_pool_capacity'] = 1
    config['general']['training_method'] = 'dagger'
    config['env']['task_types'] = task_types

    if split == "train":
        config['dataset']['data_path'] = pjoin(ALFWORLD_DATA, "json_2.1.1", "train")
    else:
        config['dataset']['data_path'] = pjoin(ALFWORLD_DATA, "json_2.1.1", "valid_seen")

    agent = TextDAggerAgent(config)
    alfred_env = get_environment(config["env"]["type"])(config, train_eval="train")
    env = alfred_env.init_env(batch_size=1)
    num_game = alfred_env.num_games
    env.seed(42)
    np.random.seed(42)

    # Track already processed trials for resumability
    done_trials = set()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    done_trials.add(data["trial_name"])
        print(f"Resuming: {len(done_trials)} trials already processed.")

    episode_no = 0
    while episode_no < num_game:
        obs, infos = env.reset()
        game_names = infos["extra.gamefile"]

        # Build trial_name: "tasktype-...-scenenum_trial_T..."
        # game_names[0] is like: /path/to/pick_and_place_simple-Apple-None-Fridge-1/trial_T00000/game.tw-pddl
        # generate_alfworld_data.py expects: "pick_and_place_simple-Apple-None-Fridge-1_trial_T00000"
        # and converts it back with: trial_name.replace("_trial", os.sep + "trial")
        parts = game_names[0].replace("\\", "/").split("/")
        trial_name = parts[-3] + "_" + parts[-2]

        if trial_name in done_trials:
            agent.finish_of_episode(episode_no, 1)
            episode_no += 1
            continue

        agent.train()
        agent.init(1)

        # Parse initial observation and task description
        observation_strings = list(obs)
        task_desc_strings, observation_strings = agent.get_task_and_obs(observation_strings)
        task_desc_strings = agent.preprocess_task(task_desc_strings)
        observation_strings = agent.preprocess_observation(observation_strings)

        current_obs = observation_strings[0]
        history = []

        for step_no in range(MAX_STEPS):
            if "extra.expert_plan" in infos and len(infos["extra.expert_plan"][0]) > 0:
                action = infos["extra.expert_plan"][0][0]
            else:
                action = "look"

            obs_next, _, dones, infos = env.step([action])
            dones = [float(d) for d in dones]

            next_obs_strings = list(obs_next)
            next_obs_strings = agent.preprocess_observation(next_obs_strings)
            next_obs = next_obs_strings[0]

            history.append({
                "observation": current_obs,
                "action": action,
                "next_observation": next_obs,
            })

            current_obs = next_obs

            if dones[0]:
                break

        print(
            f"Episode: {episode_no:3d} | {trial_name} | steps: {len(history)}"
            + (" [SKIPPED: max steps]" if len(history) >= MAX_STEPS else "")
        )

        # Only save episodes that terminated successfully (did not hit max steps)
        if 0 < len(history) < MAX_STEPS:
            record = {
                "trial_name": trial_name,
                "task": task_desc_strings[0],
                "history": history,
            }
            with open(output_path, "a") as f:
                f.write(json.dumps(record) + "\n")

        agent.finish_of_episode(episode_no, 1)
        episode_no += 1

    print(f"Done. Saved to {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect ALFWorld expert trajectories in JSONL format "
                    "for use with generate_alfworld_data.py."
    )
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to alfworld config YAML file (e.g., configs/base_config.yaml).",
    )
    parser.add_argument(
        "--output_path", "-o",
        type=str,
        required=True,
        help="Path to output JSONL file (e.g., data/alfworld_trajectories.jsonl).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "valid_seen"],
        help="Dataset split to collect from.",
    )
    parser.add_argument(
        "--task_types",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6],
        help="Task type IDs to collect (1=pick_place, 2=pick_two, 3=look_at_light, "
             "4=pick_clean_place, 5=pick_heat_place, 6=pick_cool_place). Default: all.",
    )
    args = parser.parse_args()
    collect_trajectories(args.output_path, args.split, args.task_types)


if __name__ == "__main__":
    main()
