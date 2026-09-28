import json
import os
import re


JSON_FILE = "WLASL_v0.3.json"
VIDEO_FOLDER = "selected_videos"


with open(JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)


lookup = {}

for entry in data:

    sign = entry["gloss"].lower().strip()

    for instance in entry["instances"]:

        video_id = str(instance["video_id"])

        lookup[video_id] = {
            "sign": sign,
            "start": instance.get("frame_start"),
            "end": instance.get("frame_end")
        }


for sign_folder in sorted(os.listdir(VIDEO_FOLDER)):

    path = os.path.join(
        VIDEO_FOLDER,
        sign_folder
    )

    if not os.path.isdir(path):
        continue

    print()
    print("=" * 50)
    print(sign_folder)
    print("=" * 50)

    for filename in sorted(os.listdir(path)):

        if not filename.lower().endswith(
            (".mp4", ".avi", ".mov")
        ):
            continue

        # Pull first number from filename
        match = re.match(r"(\d+)", filename)

        if not match:
            continue

        video_id = match.group(1)

        info = lookup.get(video_id)

        if info:

            print(
                filename,
                "->",
                "start:",
                info["start"],
                "end:",
                info["end"]
            )

        else:

            print(
                filename,
                "-> NOT FOUND"
            )