import json
import os
import subprocess


TARGET_SIGNS = {
    "hello", "help", "love", "me", "no",
    "please", "sorry", "thank you", "yes", "you",

    "good", "bad", "want", "need", "like",
    "know", "think", "understand",

    "eat", "drink", "water", "food",

    "home", "school", "work",

    "friend", "family",
    "mother", "father",
    "brother", "sister",

    "man", "woman", "child",

    "happy", "sad", "angry", "tired",

    "morning", "night",
    "today", "tomorrow", "yesterday",

    "what", "where", "who",
    "why", "how", "when",

    "go", "come", "stop", "wait",

    "more", "again",

    "name", "time", "money",

    "bathroom",
    "car", "phone", "computer",

    "hot", "cold",
    "big", "small",

    "right", "wrong",
    "here", "there"
}


JSON_FILE = "WLASL_v0.3.json"
OUTPUT_FOLDER = "selected_videos"


with open(
    JSON_FILE,
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)


os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


for entry in data:

    gloss = entry["gloss"].lower().strip()

    if gloss not in TARGET_SIGNS:
        continue


    folder_name = gloss.replace(
        " ",
        "_"
    )


    sign_folder = os.path.join(
        OUTPUT_FOLDER,
        folder_name
    )


    os.makedirs(
        sign_folder,
        exist_ok=True
    )


    print()
    print("=" * 55)
    print("SIGN:", gloss.upper())
    print("=" * 55)


    for instance in entry["instances"]:

        url = instance.get("url")
        video_id = str(
            instance.get("video_id")
        )


        if not url or not video_id:
            continue


        # --------------------------------------------------
        # CHECK WHETHER THIS VIDEO ALREADY EXISTS
        # --------------------------------------------------

        existing = [
            f for f in os.listdir(sign_folder)
            if f.startswith(video_id)
        ]


        if existing:

            print(
                "SKIP:",
                video_id,
                "(already downloaded)"
            )

            continue


        output_template = os.path.join(
            sign_folder,
            video_id + ".%(ext)s"
        )


        print(
            "Downloading:",
            video_id
        )


        subprocess.run(
            [
                "yt-dlp",

                url,

                "-o",
                output_template,

                "--quiet",

                "--no-warnings",

                "--no-playlist"
            ],

            check=False
        )


print()
print("=" * 55)
print("DOWNLOAD COMPLETE")
print("=" * 55)