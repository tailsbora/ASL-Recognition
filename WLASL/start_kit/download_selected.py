import json
import os
import subprocess


TARGET_SIGNS = {
    "hello",
    "yes",
    "no",
    "please",
    "thank you",
    "sorry",
    "help",
    "love",
    "you",
    "me"
}

JSON_FILE = "WLASL_v0.3.json"
OUTPUT_FOLDER = "selected_videos"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


with open(JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)


for entry in data:

    gloss = entry["gloss"].lower().strip()

    if gloss not in TARGET_SIGNS:
        continue

    sign_folder = os.path.join(
        OUTPUT_FOLDER,
        gloss.replace(" ", "_")
    )

    os.makedirs(sign_folder, exist_ok=True)

    print()
    print("=" * 50)
    print("SIGN:", gloss.upper())
    print("=" * 50)

    for instance in entry["instances"]:

        url = instance.get("url")
        video_id = instance.get("video_id")

        if not url or not video_id:
            continue

        output_file = os.path.join(
            sign_folder,
            f"{video_id}.mp4"
        )

        if os.path.exists(output_file):
            print("SKIP:", video_id)
            continue

        print("Downloading:", video_id)

        subprocess.run(
            [
                "yt-dlp",
                url,
                "-o",
                output_file,
                "--quiet",
                "--no-warnings"
            ],
            check=False
        )


print()
print("DONE")