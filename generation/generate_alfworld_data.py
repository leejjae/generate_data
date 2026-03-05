import json
import os
from pathlib import Path

from utils.alfworld.env import CustomThorEnv
from utils.alfworld.load import load_trajectory

_alfworld_root = os.environ.get(
    "ALFWORLD_ROOT", os.path.expanduser("~/.cache/alfworld")
)


def _download_alfworld_data():
    import subprocess

    print("Downloading ALFWorld data...")
    subprocess.run(["alfworld-download", "--extra"])
    print("Downloaded ALFWorld data.")


def _maybe_download_alfworld_data():
    if not os.path.exists(os.path.join(_alfworld_root, "seq2seq_data")):
        _download_alfworld_data()


def _load_trajectory(traj_name: str):
    return load_trajectory(
        os.path.join(
            _alfworld_root, "json_2.1.1", "train", traj_name, "traj_data.json"
        )
    )


def _get_environment_graph(env: CustomThorEnv):
    graph = env.environment_graph(only_visible=False)

    id2sb = {}
    sbset = set()
    for node in graph["nodes"]:
        cat = node["category"].lower()
        i = 1
        while f"{cat} {i}" in sbset:
            i += 1
        sbset.add(f"{cat} {i}")
        id2sb[node["id"]] = f"{cat} {i}"

    return ", ".join(
        f"({id2sb[edge['from_id']] if edge['from_id'] in id2sb else edge['from_id']}, "
        f"{edge['relation'].value}, "
        f"{id2sb[edge['to_id']] if edge['to_id'] in id2sb else edge['to_id']})"
        for edge in graph["edges"]
    )


def generate_alfworld_data(original_path: str, output_path: str):
    _maybe_download_alfworld_data()

    env = CustomThorEnv(
        build_path="/root/.ai2thor/releases/thor-201909061227-Linux64/thor-201909061227-Linux64"
    )

    with open(original_path, "r") as f:
        original_data = [json.loads(line) for line in f.readlines()]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    trials = set()
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f.readlines():
                data = json.loads(line)
                trials.add(data["trial_name"])
    for data in original_data:
        if data["trial_name"] in trials:
            continue
        trial_name = data["trial_name"].replace("_trial", f"{os.sep}trial")
        traj = _load_trajectory(trial_name)
        env.reset(
            trajectory_root=os.path.join(_alfworld_root, "json_2.1.1", "train"),
            trajectory_data=traj,
        )
        for step in data["history"]:
            step["observation_graph"] = _get_environment_graph(env)
            env.render_script(step["action"])
        with open(output_path, "a") as f:
            f.write(json.dumps(data) + "\n")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original_path",
        "-i",
        type=str,
        help="Path to the original ALFWorld data.",
    )
    parser.add_argument(
        "--output_path",
        "-o",
        type=str,
        help="Path to save the generated ALFWorld data.",
    )
    args = parser.parse_args()
    generate_alfworld_data(args.original_path, args.output_path)


if __name__ == "__main__":
    main()
