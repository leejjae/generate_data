"""Delete loop trajectory files (walk-room-only or repeating pattern).

Run this before re-generating to let generate_virtualhome_data.py
recreate only the deleted files.
"""

import json
import os


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data_virtualhome")
    rooms = {"kitchen", "bedroom", "livingroom", "bathroom"}
    deleted = 0

    for root, _dirs, files in os.walk(data_dir):
        for fname in sorted(files):
            if not fname.endswith(".jsonl"):
                continue
            fpath = os.path.join(root, fname)
            actions = []
            with open(fpath) as f:
                for line in f:
                    actions.append(json.loads(line)["action"])

            all_walk_room = (
                all(
                    a.split()[0] == "walk" and a.split()[1] in rooms
                    for a in actions
                )
                if actions
                else False
            )

            has_loop = False
            if len(actions) >= 6:
                tail = actions[-6:]
                if (
                    tail[0] == tail[2] == tail[4]
                    and tail[1] == tail[3] == tail[5]
                ):
                    has_loop = True

            if all_walk_room or has_loop:
                os.remove(fpath)
                deleted += 1
                print(f"Deleted: {fpath}")

    print(f"\nTotal deleted: {deleted}")


if __name__ == "__main__":
    main()
