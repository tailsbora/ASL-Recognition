import os
import re
import json
import time
import shutil
import urllib.request
import urllib.parse

from pathlib import Path
from collections import Counter, defaultdict

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader


try:
    import yt_dlp

except ImportError:

    print()
    print("Missing yt-dlp.")
    print("Install it with:")
    print()
    print("pip install yt-dlp")
    print()

    raise SystemExit


# ============================================================
# ASL RECOGNITION
# MODEL UPDATER
#
# v0.2.0
#
# 70 words -> 100 words
#
# THIS FILE:
#
# 1. Checks the vocabulary
# 2. Downloads ONLY the 30 new words
# 3. Remembers dead download links
# 4. Counts usable videos
# 5. Extracts missing 30 x 189 sequences
# 6. Reuses existing extracted sequences
# 7. Warm-starts from the old model
# 8. Trains the 100-word BiGRU
# 9. Evaluates validation performance
# 10. Saves the new model
# ============================================================


VERSION = "0.2.0"


# ============================================================
# FILES
# ============================================================

JSON_FILE = Path(
    "WLASL_v0.3.json"
)

VIDEO_FOLDER = Path(
    "selected_videos"
)

SEQUENCE_FOLDER = Path(
    "wlasl_sequences_final"
)

MODEL_FILE = Path(
    "asl_bigru_model.pt"
)

FAILED_DOWNLOADS_FILE = Path(
    "failed_downloads.json"
)

VALIDATION_RESULTS_FILE = Path(
    "validation_results_v020.json"
)


HAND_MODEL_FILE = Path(
    "../../hand_landmarker.task"
)

POSE_MODEL_FILE = Path(
    "../../pose_landmarker.task"
)


# ============================================================
# MODEL SETTINGS
# ============================================================

SEQUENCE_LENGTH = 30

FEATURE_SIZE = 189


BATCH_SIZE = 16

EPOCHS = 300

LEARNING_RATE = 0.001

PATIENCE = 45

VALIDATION_SPLIT = 0.20

MIN_SEQUENCES_PER_CLASS = 4

RANDOM_SEED = 42


# ============================================================
# DOWNLOAD SETTINGS
# ============================================================

DOWNLOAD_TIMEOUT = 15


VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi"
}


# ============================================================
# ORIGINAL 70 WORDS
# ============================================================

ORIGINAL_WORDS = [

    "again",
    "angry",
    "bad",
    "bathroom",
    "big",

    "brother",
    "car",
    "child",
    "cold",
    "come",

    "computer",
    "drink",
    "eat",
    "family",
    "father",

    "food",
    "friend",
    "go",
    "good",
    "happy",

    "hello",
    "help",
    "here",
    "home",
    "hot",

    "how",
    "know",
    "like",
    "love",
    "man",

    "me",
    "money",
    "more",
    "morning",
    "mother",

    "name",
    "need",
    "night",
    "no",
    "phone",

    "please",
    "right",
    "sad",
    "school",
    "sister",

    "small",
    "sorry",
    "stop",
    "thank you",
    "there",

    "think",
    "time",
    "tired",
    "today",
    "tomorrow",

    "understand",
    "wait",
    "want",
    "water",
    "what",

    "when",
    "where",
    "who",
    "why",
    "woman",

    "work",
    "wrong",
    "yes",
    "yesterday",
    "you"
]


# ============================================================
# NEW 30 WORDS — v0.2.0
# ============================================================

NEW_WORDS = [

    "answer",
    "apple",
    "book",
    "boy",
    "girl",

    "day",
    "week",
    "month",
    "year",

    "learn",
    "teacher",
    "student",

    "read",
    "write",
    "play",

    "sleep",
    "remember",
    "forget",

    "run",
    "finish",

    "music",
    "coffee",

    "door",
    "room",
    "house",

    "chair",
    "dog",
    "cat",

    "language",
    "walk"
]


# ============================================================
# COMPLETE 100-WORD VOCABULARY
# ============================================================

WORDS = (
    ORIGINAL_WORDS
    + NEW_WORDS
)


if len(ORIGINAL_WORDS) != 70:

    raise RuntimeError(
        "Original vocabulary is not 70 words."
    )


if len(NEW_WORDS) != 30:

    raise RuntimeError(
        "New vocabulary is not 30 words."
    )


if len(WORDS) != 100:

    raise RuntimeError(
        "Total vocabulary is not 100 words."
    )


if len(set(WORDS)) != len(WORDS):

    raise RuntimeError(
        "Duplicate word found in vocabulary."
    )


# ============================================================
# RANDOM SEEDS
# ============================================================

np.random.seed(
    RANDOM_SEED
)

torch.manual_seed(
    RANDOM_SEED
)


# ============================================================
# HELPERS
# ============================================================

def folder_name(word):

    return (
        word
        .strip()
        .lower()
        .replace(
            " ",
            "_"
        )
    )


def get_video_id(filename):

    match = re.match(
        r"(\d+)",
        filename
    )

    if match:

        return match.group(
            1
        )

    return None


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

required_files = [

    JSON_FILE,
    HAND_MODEL_FILE,
    POSE_MODEL_FILE
]


for required_file in required_files:

    if not required_file.exists():

        print()
        print(
            "MISSING FILE:"
        )

        print(
            required_file.resolve()
        )

        print()

        raise SystemExit


# ============================================================
# START
# ============================================================

print()
print("=" * 72)
print(
    f"ASL RECOGNITION v{VERSION} MODEL UPDATER"
)
print("=" * 72)

print()
print(
    "Existing vocabulary:",
    len(ORIGINAL_WORDS)
)

print(
    "New vocabulary:",
    len(NEW_WORDS)
)

print(
    "Final vocabulary:",
    len(WORDS)
)


# ============================================================
# LOAD WLASL METADATA
# ============================================================

print()
print(
    "Loading WLASL metadata..."
)


with open(
    JSON_FILE,
    "r",
    encoding="utf-8"
) as file:

    metadata = json.load(
        file
    )


# ============================================================
# CREATE WLASL LOOKUP
# ============================================================

wlasl_lookup = {}


for entry in metadata:

    gloss = (
        entry
        .get(
            "gloss",
            ""
        )
        .strip()
        .lower()
    )


    if gloss:

        wlasl_lookup[
            gloss
        ] = entry.get(
            "instances",
            []
        )


# ============================================================
# CHECK ALL 100 WORDS
# ============================================================

print()
print("=" * 72)
print("CHECKING VOCABULARY")
print("=" * 72)


missing_words = []


for word in WORDS:

    if word not in wlasl_lookup:

        missing_words.append(
            word
        )


if missing_words:

    print()
    print(
        "These words are missing from WLASL:"
    )

    for word in missing_words:

        print(
            " -",
            word
        )

    raise SystemExit


print()
print(
    "All 100 words found ✅"
)


# ============================================================
# FAILED DOWNLOAD CACHE
# ============================================================

if FAILED_DOWNLOADS_FILE.exists():

    try:

        with open(
            FAILED_DOWNLOADS_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            failed_downloads = set(
                json.load(
                    file
                )
            )

    except Exception:

        failed_downloads = set()


else:

    failed_downloads = set()


def save_failed_downloads():

    with open(
        FAILED_DOWNLOADS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(

            sorted(
                failed_downloads
            ),

            file,

            indent=2
        )


# ============================================================
# SILENT YT-DLP LOGGER
#
# Stops giant ERROR messages from filling the terminal.
# We display our own concise status instead.
# ============================================================

class SilentYTDLPLogger:

    def debug(
        self,
        message
    ):
        pass


    def warning(
        self,
        message
    ):
        pass


    def error(
        self,
        message
    ):
        pass


# ============================================================
# VIDEO HELPERS
# ============================================================

def get_usable_videos(folder):

    if not folder.exists():

        return []


    return [

        file

        for file in folder.iterdir()

        if (

            file.is_file()

            and

            file.suffix.lower()
            in VIDEO_EXTENSIONS

        )
    ]


def video_exists(
    folder,
    video_id
):

    if not folder.exists():

        return False


    prefix = str(
        video_id
    )


    for file in folder.iterdir():

        if (

            file.is_file()

            and

            file.name.startswith(
                prefix
            )

        ):

            return True


    return False


# ============================================================
# CHECK WHETHER A DOWNLOADED VIDEO ACTUALLY OPENS
# ============================================================

def valid_video_file(
    path
):

    if not path.exists():

        return False


    if path.stat().st_size < 1024:

        return False


    capture = cv2.VideoCapture(
        str(
            path
        )
    )


    if not capture.isOpened():

        capture.release()

        return False


    success, frame = (
        capture.read()
    )


    capture.release()


    return (

        success

        and

        frame is not None
    )


# ============================================================
# DIRECT VIDEO DOWNLOAD
#
# Useful for URLs ending directly in .mp4 / .mov / .avi
# ============================================================

def direct_download(
    url,
    folder,
    video_id
):

    try:

        parsed = urllib.parse.urlparse(
            url
        )


        extension = Path(
            parsed.path
        ).suffix.lower()


        if extension not in VIDEO_EXTENSIONS:

            return False


        destination = (

            folder
            / f"{video_id}{extension}"
        )


        temp_destination = (

            folder
            / f"{video_id}{extension}.part"
        )


        request = urllib.request.Request(

            url,

            headers={

                "User-Agent":
                    "Mozilla/5.0"

            }
        )


        with urllib.request.urlopen(

            request,

            timeout=DOWNLOAD_TIMEOUT

        ) as response:

            with open(

                temp_destination,

                "wb"

            ) as file:

                while True:

                    chunk = response.read(
                        1024 * 1024
                    )


                    if not chunk:

                        break


                    file.write(
                        chunk
                    )


        temp_destination.replace(
            destination
        )


        if valid_video_file(
            destination
        ):

            return True


        try:

            destination.unlink()

        except Exception:

            pass


        return False


    except Exception:

        try:

            if temp_destination.exists():

                temp_destination.unlink()

        except Exception:

            pass


        return False


# ============================================================
# YT-DLP DOWNLOAD
# ============================================================

def ytdlp_download(
    url,
    folder,
    video_id
):

    output_template = str(

        folder
        / f"{video_id}.%(ext)s"
    )


    options = {

        "format":
            "best[ext=mp4]/best",

        "outtmpl":
            output_template,

        "quiet":
            True,

        "no_warnings":
            True,

        "noplaylist":
            True,

        "retries":
            1,

        "fragment_retries":
            1,

        "socket_timeout":
            DOWNLOAD_TIMEOUT,

        "ignoreerrors":
            True,

        "overwrites":
            False,

        "logger":
            SilentYTDLPLogger()
    }


    before = set(

        file.name

        for file in folder.iterdir()

        if file.is_file()
    )


    try:

        with yt_dlp.YoutubeDL(
            options
        ) as ydl:

            result = ydl.download(
                [url]
            )


        if result != 0:

            return False


    except Exception:

        return False


    after = [

        file

        for file in folder.iterdir()

        if (

            file.is_file()

            and

            file.name not in before

            and

            file.suffix.lower()
            in VIDEO_EXTENSIONS

        )
    ]


    for file in after:

        if valid_video_file(
            file
        ):

            return True


    # It may have created an existing filename
    # rather than a new one.
    for file in get_usable_videos(
        folder
    ):

        if (

            file.name.startswith(
                str(video_id)
            )

            and

            valid_video_file(
                file
            )

        ):

            return True


    return False


# ============================================================
# DOWNLOAD ONE URL
# ============================================================

def download_video(
    url,
    folder,
    video_id
):

    parsed = urllib.parse.urlparse(
        url
    )


    extension = Path(
        parsed.path
    ).suffix.lower()


    # Direct file URL
    if extension in VIDEO_EXTENSIONS:

        return direct_download(

            url,
            folder,
            video_id
        )


    # YouTube / other supported websites
    return ytdlp_download(

        url,
        folder,
        video_id
    )


# ============================================================
# STEP 1
# DOWNLOAD ONLY THE NEW 30 WORDS
# ============================================================

print()
print("=" * 72)
print("STEP 1/4 — DOWNLOADING NEW v0.2 WORDS")
print("=" * 72)

print()
print(
    "The original 70 words will NOT be downloaded again."
)

print(
    "Only the 30 new words will be checked."
)

print(
    "Remembered dead links:",
    len(
        failed_downloads
    )
)


VIDEO_FOLDER.mkdir(

    parents=True,

    exist_ok=True
)


downloaded = 0

already_have = 0

new_failures = 0

remembered_failures = 0


for word_number, word in enumerate(

    NEW_WORDS,

    start=1

):

    instances = wlasl_lookup[
        word
    ]


    destination = (

        VIDEO_FOLDER
        / folder_name(
            word
        )
    )


    destination.mkdir(

        parents=True,

        exist_ok=True
    )


    starting_count = len(

        get_usable_videos(
            destination
        )
    )


    print()
    print("=" * 72)

    print(

        f"[{word_number:02}/{len(NEW_WORDS)}] "

        f"{word.upper()}"
    )


    print(

        f"Existing usable videos: "
        f"{starting_count}"
    )


    print(

        f"WLASL metadata instances: "
        f"{len(instances)}"
    )


    word_downloaded = 0

    word_failed = 0

    word_cached = 0


    for instance in instances:

        video_id = str(

            instance.get(
                "video_id",
                ""
            )

        ).strip()


        url = str(

            instance.get(
                "url",
                ""
            )

        ).strip()


        if (

            not video_id

            or

            not url

        ):

            continue


        failure_key = (

            f"{folder_name(word)}"
            f":"
            f"{video_id}"
        )


        # ----------------------------------------------------
        # ALREADY DOWNLOADED
        # ----------------------------------------------------

        if video_exists(

            destination,

            video_id

        ):

            already_have += 1

            continue


        # ----------------------------------------------------
        # ALREADY KNOWN TO BE DEAD
        # ----------------------------------------------------

        if failure_key in failed_downloads:

            remembered_failures += 1

            word_cached += 1

            continue


        print(

            f"    Trying {video_id}...",

            end=" ",
            flush=True
        )


        success = download_video(

            url,
            destination,
            video_id
        )


        if success:

            downloaded += 1

            word_downloaded += 1


            print(
                "OK ✅"
            )


        else:

            failed_downloads.add(
                failure_key
            )


            save_failed_downloads()


            new_failures += 1

            word_failed += 1


            print(
                "FAILED ❌"
            )


    final_count = len(

        get_usable_videos(
            destination
        )
    )


    print()
    print(

        f"    {word}: "
        f"{final_count} usable videos"
    )


    print(

        f"    New: {word_downloaded} | "
        f"Failed: {word_failed} | "
        f"Previously known dead: {word_cached}"
    )


save_failed_downloads()


# ============================================================
# DOWNLOAD SUMMARY
# ============================================================

print()
print("=" * 72)
print("DOWNLOAD SUMMARY")
print("=" * 72)

print(
    "New usable videos:",
    downloaded
)

print(
    "Existing files skipped:",
    already_have
)

print(
    "Known dead links skipped:",
    remembered_failures
)

print(
    "New dead links remembered:",
    new_failures
)

print(
    "Failure cache:",
    FAILED_DOWNLOADS_FILE
)


# ============================================================
# COUNT VIDEOS FOR ALL 100 WORDS
# ============================================================

print()
print("=" * 72)
print("USABLE VIDEO COUNTS — ALL 100 WORDS")
print("=" * 72)


usable_counts = {}


for word in WORDS:

    folder = (

        VIDEO_FOLDER
        / folder_name(
            word
        )
    )


    count = len(

        get_usable_videos(
            folder
        )
    )


    usable_counts[
        word
    ] = count


    marker = ""


    if count < 4:

        marker = "  ⚠ LOW"


    print(

        f"{word:<15} "
        f"{count:>3}"
        f"{marker}"
    )


print()
print(

    "TOTAL USABLE VIDEOS:",

    sum(
        usable_counts.values()
    )
)


# ============================================================
# MEDIAPIPE EXTRACTION
# ============================================================

print()
print("=" * 72)
print("STEP 2/4 — EXTRACTING MISSING LANDMARK SEQUENCES")
print("=" * 72)


BaseOptions = (
    mp.tasks.BaseOptions
)


RunningMode = (
    mp.tasks.vision.RunningMode
)


HandLandmarker = (
    mp.tasks.vision.HandLandmarker
)


HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)


PoseLandmarker = (
    mp.tasks.vision.PoseLandmarker
)


PoseLandmarkerOptions = (
    mp.tasks.vision.PoseLandmarkerOptions
)


hand_options = HandLandmarkerOptions(

    base_options=BaseOptions(

        model_asset_path=str(
            HAND_MODEL_FILE
        )
    ),

    running_mode=(
        RunningMode.IMAGE
    ),

    num_hands=2
)


pose_options = PoseLandmarkerOptions(

    base_options=BaseOptions(

        model_asset_path=str(
            POSE_MODEL_FILE
        )
    ),

    running_mode=(
        RunningMode.IMAGE
    ),

    num_poses=1
)


hand_detector = (

    HandLandmarker
    .create_from_options(
        hand_options
    )
)


pose_detector = (

    PoseLandmarker
    .create_from_options(
        pose_options
    )
)


# ============================================================
# POSE POINTS USED BY MODEL
# ============================================================

POSE_USED_IDS = [

    0,

    1, 2, 3,

    4, 5, 6,

    7, 8,

    9, 10,

    11, 12,

    13, 14,

    15, 16,

    23, 24
]


# ============================================================
# HAND FEATURES
# ============================================================

def make_hand_vector(
    hand
):

    points = np.array(

        [
            [
                point.x,
                point.y,
                point.z
            ]

            for point in hand
        ],

        dtype=np.float32
    )


    wrist = points[
        0
    ].copy()


    relative = (

        points
        - wrist

    )


    scale = np.linalg.norm(

        relative[
            9
        ]
    )


    if scale < 0.0001:

        scale = 1.0


    relative /= scale


    return np.concatenate(

        [
            relative.flatten(),
            wrist
        ]
    )


# ============================================================
# POSE FEATURES
# ============================================================

def make_pose_vector(
    pose
):

    if pose is None:

        return np.zeros(

            57,

            dtype=np.float32
        )


    points = np.array(

        [
            [
                pose[index].x,
                pose[index].y,
                pose[index].z
            ]

            for index
            in POSE_USED_IDS
        ],

        dtype=np.float32
    )


    left_shoulder = np.array(

        [
            pose[11].x,
            pose[11].y,
            pose[11].z
        ],

        dtype=np.float32
    )


    right_shoulder = np.array(

        [
            pose[12].x,
            pose[12].y,
            pose[12].z
        ],

        dtype=np.float32
    )


    center = (

        left_shoulder
        + right_shoulder

    ) / 2


    shoulder_width = np.linalg.norm(

        left_shoulder
        - right_shoulder
    )


    if shoulder_width < 0.0001:

        shoulder_width = 1.0


    points = (

        points
        - center

    ) / shoulder_width


    return points.flatten()


# ============================================================
# FRAME FEATURES
# ============================================================

def make_frame_features(
    frame
):

    rgb = cv2.cvtColor(

        frame,

        cv2.COLOR_BGR2RGB
    )


    mp_image = mp.Image(

        image_format=(
            mp.ImageFormat.SRGB
        ),

        data=rgb
    )


    hand_result = (
        hand_detector.detect(
            mp_image
        )
    )


    pose_result = (
        pose_detector.detect(
            mp_image
        )
    )


    left_hand = np.zeros(

        66,

        dtype=np.float32
    )


    right_hand = np.zeros(

        66,

        dtype=np.float32
    )


    for hand, information in zip(

        hand_result.hand_landmarks,

        hand_result.handedness

    ):

        vector = make_hand_vector(
            hand
        )


        if not information:

            continue


        label = (

            information[0]
            .category_name
            .lower()
        )


        if label == "left":

            left_hand = vector


        elif label == "right":

            right_hand = vector


    pose = None


    if pose_result.pose_landmarks:

        pose = (
            pose_result
            .pose_landmarks[0]
        )


    pose_vector = make_pose_vector(
        pose
    )


    features = np.concatenate(

        [
            left_hand,
            right_hand,
            pose_vector
        ]
    )


    return features.astype(
        np.float32
    )


# ============================================================
# METADATA LOOKUP BY VIDEO ID
# ============================================================

instance_lookup = {}


for word in WORDS:

    instance_lookup[
        word
    ] = {}


    for instance in wlasl_lookup[
        word
    ]:

        video_id = str(

            instance.get(
                "video_id",
                ""
            )

        ).strip()


        if video_id:

            instance_lookup[
                word
            ][
                video_id
            ] = instance


# ============================================================
# READ FRAME SAFELY
# ============================================================

def read_frame(
    capture,
    frame_number
):

    attempts = [

        frame_number,

        frame_number - 1,

        frame_number + 1,

        frame_number - 2,

        frame_number + 2
    ]


    for attempt in attempts:

        if attempt < 0:

            continue


        capture.set(

            cv2.CAP_PROP_POS_FRAMES,

            int(
                attempt
            )
        )


        success, frame = (
            capture.read()
        )


        if (

            success

            and

            frame is not None

        ):

            return frame


    return None


# ============================================================
# EXTRACT ONE VIDEO
# ============================================================

def extract_video(
    video_path,
    instance
):

    capture = cv2.VideoCapture(

        str(
            video_path
        )
    )


    if not capture.isOpened():

        capture.release()

        return None


    total_frames = int(

        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    if total_frames <= 0:

        capture.release()

        return None


    try:

        frame_start = int(

            instance.get(
                "frame_start",
                1
            )

            or 1
        )

    except Exception:

        frame_start = 1


    try:

        frame_end = int(

            instance.get(
                "frame_end",
                -1
            )

            or -1
        )

    except Exception:

        frame_end = -1


    # ========================================================
    # ALREADY-TRIMMED / FULL CLIP
    # ========================================================

    if (

        frame_start <= 1

        and

        frame_end == -1

    ):

        start_index = 0

        end_index = (
            total_frames - 1
        )


    else:

        start_index = max(

            0,

            frame_start - 1
        )


        if frame_end == -1:

            end_index = (
                total_frames - 1
            )


        else:

            end_index = min(

                total_frames - 1,

                frame_end - 1
            )


        # Some downloaded WLASL videos are already trimmed.
        # Their original metadata frame numbers may therefore
        # be larger than the downloaded clip.
        if (

            start_index >= total_frames

            or

            end_index <= start_index

        ):

            start_index = 0

            end_index = (
                total_frames - 1
            )


    frame_numbers = np.linspace(

        start_index,

        end_index,

        SEQUENCE_LENGTH

    ).astype(
        int
    )


    sequence = []


    for frame_number in frame_numbers:

        frame = read_frame(

            capture,

            frame_number
        )


        if frame is None:

            capture.release()

            return None


        features = make_frame_features(
            frame
        )


        if features.shape[0] != FEATURE_SIZE:

            capture.release()

            return None


        sequence.append(
            features
        )


    capture.release()


    sequence = np.asarray(

        sequence,

        dtype=np.float32
    )


    if sequence.shape != (

        SEQUENCE_LENGTH,

        FEATURE_SIZE

    ):

        return None


    return sequence


# ============================================================
# EXTRACT ONLY MISSING SEQUENCES
# ============================================================

SEQUENCE_FOLDER.mkdir(

    parents=True,

    exist_ok=True
)


new_sequences = 0

existing_sequences = 0

failed_sequences = 0


for word_number, word in enumerate(

    WORDS,

    start=1

):

    video_folder = (

        VIDEO_FOLDER
        / folder_name(
            word
        )
    )


    sequence_folder = (

        SEQUENCE_FOLDER
        / folder_name(
            word
        )
    )


    sequence_folder.mkdir(

        parents=True,

        exist_ok=True
    )


    if not video_folder.exists():

        continue


    videos = sorted(

        get_usable_videos(
            video_folder
        )
    )


    print()
    print(

        f"[{word_number:03}/100] "

        f"{word.upper()} "

        f"({len(videos)} videos)"
    )


    for video_path in videos:

        video_id = get_video_id(
            video_path.name
        )


        if not video_id:

            continue


        output_file = (

            sequence_folder
            / f"{video_id}.npy"
        )


        # Existing old sequences are reused.
        if output_file.exists():

            existing_sequences += 1

            continue


        instance = (

            instance_lookup
            .get(
                word,
                {}
            )
            .get(
                video_id
            )
        )


        if instance is None:

            failed_sequences += 1

            continue


        print(

            f"    Extracting "
            f"{video_id}...",

            end=" ",
            flush=True
        )


        sequence = extract_video(

            video_path,
            instance
        )


        if sequence is None:

            failed_sequences += 1

            print(
                "FAILED ❌"
            )

            continue


        np.save(

            output_file,

            sequence
        )


        new_sequences += 1


        print(
            "OK ✅"
        )


hand_detector.close()

pose_detector.close()


print()
print("=" * 72)
print("EXTRACTION SUMMARY")
print("=" * 72)

print(
    "New sequences:",
    new_sequences
)

print(
    "Existing sequences reused:",
    existing_sequences
)

print(
    "Failed sequences:",
    failed_sequences
)


# ============================================================
# COUNT FINAL SEQUENCES
# ============================================================

print()
print("=" * 72)
print("FINAL SEQUENCE COUNTS")
print("=" * 72)


sequence_counts = {}


for word in WORDS:

    sequence_folder = (

        SEQUENCE_FOLDER
        / folder_name(
            word
        )
    )


    count = 0


    if sequence_folder.exists():

        count = len(

            list(

                sequence_folder.glob(
                    "*.npy"
                )
            )
        )


    sequence_counts[
        word
    ] = count


    marker = ""


    if count < MIN_SEQUENCES_PER_CLASS:

        marker = "  ❌ TOO LOW"


    elif count < 7:

        marker = "  ⚠ LOW"


    print(

        f"{word:<15} "
        f"{count:>3}"
        f"{marker}"
    )


total_sequences = sum(

    sequence_counts.values()
)


print()
print(
    "TOTAL SEQUENCES:",
    total_sequences
)


# ============================================================
# MAKE SURE ALL CLASSES ARE TRAINABLE
# ============================================================

too_small = [

    word

    for word, count
    in sequence_counts.items()

    if count < MIN_SEQUENCES_PER_CLASS
]


if too_small:

    print()
    print("=" * 72)
    print("TRAINING STOPPED")
    print("=" * 72)

    print()
    print(
        "These words have too few usable sequences:"
    )


    for word in too_small:

        print(

            f" - {word}: "
            f"{sequence_counts[word]}"
        )


    print()
    print(
        "We need at least",
        MIN_SEQUENCES_PER_CLASS,
        "per class."
    )

    print()
    print(
        "The existing model has NOT been changed."
    )

    raise SystemExit


# ============================================================
# STEP 3
# PREPARE TRAINING DATA
# ============================================================

print()
print("=" * 72)
print("STEP 3/4 — PREPARING TRAINING DATA")
print("=" * 72)


labels = [

    folder_name(
        word
    )

    for word in WORDS
]


label_to_index = {

    label: index

    for index, label
    in enumerate(
        labels
    )
}


samples_by_class = defaultdict(
    list
)


for label in labels:

    folder = (

        SEQUENCE_FOLDER
        / label
    )


    for file in sorted(

        folder.glob(
            "*.npy"
        )
    ):

        try:

            sequence = np.load(
                file
            )


            if sequence.shape == (

                SEQUENCE_LENGTH,

                FEATURE_SIZE

            ):

                samples_by_class[
                    label
                ].append(

                    sequence.astype(
                        np.float32
                    )
                )


        except Exception:

            pass


# ============================================================
# STRATIFIED TRAIN / VALIDATION SPLIT
# ============================================================

rng = np.random.default_rng(
    RANDOM_SEED
)


train_samples = []

validation_samples = []


for label in labels:

    sequences = samples_by_class[
        label
    ]


    indexes = np.arange(

        len(
            sequences
        )
    )


    rng.shuffle(
        indexes
    )


    validation_count = max(

        1,

        int(

            round(

                len(sequences)
                * VALIDATION_SPLIT

            )
        )
    )


    # Always leave at least 2 examples for training.
    validation_count = min(

        validation_count,

        len(sequences) - 2
    )


    validation_indexes = set(

        indexes[
            :validation_count
        ].tolist()
    )


    for index, sequence in enumerate(
        sequences
    ):

        sample = (

            sequence,

            label_to_index[
                label
            ]
        )


        if index in validation_indexes:

            validation_samples.append(
                sample
            )


        else:

            train_samples.append(
                sample
            )


rng.shuffle(
    train_samples
)

rng.shuffle(
    validation_samples
)


print()
print(
    "Classes:",
    len(labels)
)

print(
    "Training sequences:",
    len(train_samples)
)

print(
    "Validation sequences:",
    len(validation_samples)
)


# ============================================================
# NORMALIZATION
# ============================================================

training_array = np.stack(

    [
        sequence

        for sequence, _
        in train_samples
    ]
)


mean = training_array.mean(

    axis=(
        0,
        1
    )

).astype(
    np.float32
)


std = training_array.std(

    axis=(
        0,
        1
    )

).astype(
    np.float32
)


std[
    std < 1e-6
] = 1.0


# ============================================================
# DATASET
# ============================================================

class ASLDataset(Dataset):

    def __init__(
        self,
        samples,
        augment=False
    ):

        self.samples = samples

        self.augment = augment


    def __len__(
        self
    ):

        return len(
            self.samples
        )


    def __getitem__(
        self,
        index
    ):

        sequence, label = (
            self.samples[
                index
            ]
        )


        sequence = sequence.copy()


        # ====================================================
        # AUGMENTATION
        # ====================================================

        if self.augment:

            # -----------------------------------------------
            # SMALL LANDMARK NOISE
            # -----------------------------------------------

            if np.random.random() < 0.75:

                noise = np.random.normal(

                    0,

                    0.015,

                    sequence.shape

                ).astype(
                    np.float32
                )


                nonzero = (

                    np.abs(
                        sequence
                    )

                    > 1e-8
                )


                sequence[
                    nonzero
                ] += noise[
                    nonzero
                ]


            # -----------------------------------------------
            # TEMPORAL SHIFT
            # -----------------------------------------------

            if np.random.random() < 0.50:

                shift = np.random.randint(

                    -2,

                    3
                )


                original = (
                    sequence.copy()
                )


                if shift > 0:

                    sequence[
                        shift:
                    ] = original[
                        :-shift
                    ]


                    sequence[
                        :shift
                    ] = original[
                        0
                    ]


                elif shift < 0:

                    amount = abs(
                        shift
                    )


                    sequence[
                        :-amount
                    ] = original[
                        amount:
                    ]


                    sequence[
                        -amount:
                    ] = original[
                        -1
                    ]


            # -----------------------------------------------
            # SMOOTH RANDOM FRAME
            # -----------------------------------------------

            if np.random.random() < 0.35:

                frame_index = np.random.randint(

                    1,

                    SEQUENCE_LENGTH - 1
                )


                sequence[
                    frame_index
                ] = (

                    sequence[
                        frame_index - 1
                    ]

                    +

                    sequence[
                        frame_index + 1
                    ]

                ) / 2


            # -----------------------------------------------
            # RANDOM FEATURE DROPOUT
            # -----------------------------------------------

            if np.random.random() < 0.30:

                dropout_mask = (

                    np.random.random(
                        sequence.shape
                    )

                    < 0.02
                )


                sequence[
                    dropout_mask
                ] = 0.0


        # ====================================================
        # NORMALIZE
        # ====================================================

        sequence = (

            sequence
            - mean

        ) / std


        return (

            torch.tensor(

                sequence,

                dtype=torch.float32
            ),

            torch.tensor(

                label,

                dtype=torch.long
            )
        )


# ============================================================
# DATA LOADERS
# ============================================================

train_dataset = ASLDataset(

    train_samples,

    augment=True
)


validation_dataset = ASLDataset(

    validation_samples,

    augment=False
)


train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0
)


validation_loader = DataLoader(

    validation_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"
)


print()
print(
    "DEVICE:",
    device
)


if device.type == "cpu":

    cpu_count = (
        os.cpu_count()
        or 4
    )


    thread_count = max(

        1,

        min(
            4,
            cpu_count // 2
        )
    )


    torch.set_num_threads(
        thread_count
    )


    print(

        "PyTorch CPU threads:",
        thread_count
    )


# ============================================================
# NETWORK
# ============================================================

class SignLanguageModel(nn.Module):

    def __init__(
        self,
        input_size,
        num_classes
    ):

        super().__init__()


        self.input_projection = nn.Sequential(

            nn.Linear(
                input_size,
                192
            ),

            nn.LayerNorm(
                192
            ),

            nn.ReLU(),

            nn.Dropout(
                0.20
            )
        )


        self.gru = nn.GRU(

            input_size=192,

            hidden_size=128,

            num_layers=2,

            batch_first=True,

            bidirectional=True,

            dropout=0.25
        )


        self.attention = nn.Sequential(

            nn.Linear(
                256,
                128
            ),

            nn.Tanh(),

            nn.Linear(
                128,
                1
            )
        )


        self.classifier = nn.Sequential(

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.Dropout(
                0.35
            ),

            nn.Linear(
                128,
                num_classes
            )
        )


    def forward(
        self,
        x
    ):

        x = self.input_projection(
            x
        )


        output, _ = self.gru(
            x
        )


        attention_scores = (
            self.attention(
                output
            )
        )


        attention_weights = torch.softmax(

            attention_scores,

            dim=1
        )


        context = torch.sum(

            output
            * attention_weights,

            dim=1
        )


        return self.classifier(
            context
        )


# ============================================================
# LOAD OLD MODEL BEFORE REPLACING IT
# ============================================================

previous_checkpoint = None


if MODEL_FILE.exists():

    try:

        previous_checkpoint = torch.load(

            MODEL_FILE,

            map_location="cpu",

            weights_only=False
        )

    except Exception:

        previous_checkpoint = None


# ============================================================
# CREATE NEW 100-WORD MODEL
# ============================================================

model = SignLanguageModel(

    FEATURE_SIZE,

    len(labels)

).to(
    device
)


# ============================================================
# WARM START FROM EXISTING MODEL
#
# Reuse:
# - input projection
# - GRU
# - attention
# - first classifier layer
# - output weights for any labels already known
# ============================================================

transferred_labels = 0


if previous_checkpoint is not None:

    try:

        old_state = (
            previous_checkpoint[
                "model_state"
            ]
        )


        current_state = (
            model.state_dict()
        )


        # ----------------------------------------------------
        # LOAD ALL SAME-SHAPE LAYERS EXCEPT FINAL OUTPUT
        # ----------------------------------------------------

        for key, value in old_state.items():

            if key in {

                "classifier.3.weight",
                "classifier.3.bias"

            }:

                continue


            if (

                key in current_state

                and

                current_state[
                    key
                ].shape
                ==
                value.shape

            ):

                current_state[
                    key
                ] = value


        model.load_state_dict(
            current_state
        )


        # ----------------------------------------------------
        # COPY OLD OUTPUT ROWS BY WORD NAME
        # ----------------------------------------------------

        old_labels = (
            previous_checkpoint
            .get(
                "labels",
                []
            )
        )


        new_label_lookup = {

            label: index

            for index, label
            in enumerate(
                labels
            )
        }


        old_weight = old_state.get(
            "classifier.3.weight"
        )


        old_bias = old_state.get(
            "classifier.3.bias"
        )


        if (

            old_weight is not None

            and

            old_bias is not None

        ):

            with torch.no_grad():

                for old_index, old_label in enumerate(
                    old_labels
                ):

                    if old_label not in new_label_lookup:

                        continue


                    new_index = new_label_lookup[
                        old_label
                    ]


                    if (

                        old_index
                        >= old_weight.shape[0]

                    ):

                        continue


                    model.classifier[
                        3
                    ].weight[
                        new_index
                    ].copy_(

                        old_weight[
                            old_index
                        ].to(
                            device
                        )
                    )


                    model.classifier[
                        3
                    ].bias[
                        new_index
                    ].copy_(

                        old_bias[
                            old_index
                        ].to(
                            device
                        )
                    )


                    transferred_labels += 1


        print()
        print(
            "Warm-started from previous model ✅"
        )


        print(

            "Transferred existing output classes:",

            transferred_labels
        )


    except Exception as error:

        print()
        print(
            "Could not warm-start old model."
        )

        print(
            "Training from fresh weights instead."
        )

        print(
            "Reason:",
            error
        )


# ============================================================
# BACKUP CURRENT MODEL
# ============================================================

if MODEL_FILE.exists():

    timestamp_text = time.strftime(

        "%Y%m%d_%H%M%S"
    )


    backup_file = Path(

        f"asl_bigru_model_backup_"
        f"{timestamp_text}.pt"
    )


    shutil.copy2(

        MODEL_FILE,

        backup_file
    )


    print()
    print(
        "Old model backed up as:"
    )

    print(
        backup_file
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_train_counts = Counter(

    label

    for _, label
    in train_samples
)


weights = []


for class_index in range(
    len(labels)
):

    count = class_train_counts[
        class_index
    ]


    weights.append(

        1.0
        / max(
            count,
            1
        )
    )


weights = np.asarray(

    weights,

    dtype=np.float32
)


weights /= (
    weights.mean()
)


class_weights = torch.tensor(

    weights,

    dtype=torch.float32,

    device=device
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(

    weight=class_weights,

    label_smoothing=0.05
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=0.0005
)


# ============================================================
# LR SCHEDULER
# ============================================================

scheduler = (

    torch.optim.lr_scheduler
    .ReduceLROnPlateau(

        optimizer,

        mode="max",

        factor=0.5,

        patience=12
    )
)


# ============================================================
# EVALUATE
# ============================================================

def evaluate(
    network,
    loader
):

    network.eval()


    correct = 0

    total = 0

    total_loss = 0.0


    with torch.inference_mode():

        for sequences, targets in loader:

            sequences = sequences.to(
                device
            )


            targets = targets.to(
                device
            )


            outputs = network(
                sequences
            )


            loss = criterion(

                outputs,
                targets
            )


            total_loss += (

                loss.item()

                * targets.size(
                    0
                )
            )


            predictions = outputs.argmax(
                dim=1
            )


            correct += (

                predictions
                == targets

            ).sum().item()


            total += targets.size(
                0
            )


    if total == 0:

        return (
            0.0,
            0.0
        )


    return (

        total_loss / total,

        correct / total
    )


# ============================================================
# STEP 4
# TRAIN
# ============================================================

print()
print("=" * 72)
print("STEP 4/4 — TRAINING 100-WORD MODEL")
print("=" * 72)

print()
print(

    "Architecture: "
    "189 -> 192 -> BiGRU -> Attention -> 128 -> 100"
)

print()


best_accuracy = -1.0

epochs_without_improvement = 0


for epoch in range(

    1,

    EPOCHS + 1

):

    model.train()


    train_correct = 0

    train_total = 0

    train_loss_total = 0.0


    for sequences, targets in train_loader:

        sequences = sequences.to(
            device
        )


        targets = targets.to(
            device
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        outputs = model(
            sequences
        )


        loss = criterion(

            outputs,
            targets
        )


        loss.backward()


        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            1.0
        )


        optimizer.step()


        train_loss_total += (

            loss.item()

            * targets.size(
                0
            )
        )


        predictions = outputs.argmax(
            dim=1
        )


        train_correct += (

            predictions
            == targets

        ).sum().item()


        train_total += targets.size(
            0
        )


    train_accuracy = (

        train_correct
        / train_total
    )


    (
        validation_loss,
        validation_accuracy

    ) = evaluate(

        model,
        validation_loader
    )


    scheduler.step(
        validation_accuracy
    )


    current_lr = (

        optimizer
        .param_groups[0][
            "lr"
        ]
    )


    print(

        f"Epoch {epoch:03} | "

        f"Train "
        f"{train_accuracy * 100:5.1f}% | "

        f"Val "
        f"{validation_accuracy * 100:5.1f}% | "

        f"LR "
        f"{current_lr:.6f}"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if validation_accuracy > best_accuracy:

        best_accuracy = (
            validation_accuracy
        )


        epochs_without_improvement = 0


        cpu_state = {

            key:
                value
                .detach()
                .cpu()

            for key, value
            in model.state_dict().items()
        }


        torch.save(

            {

                "version":
                    VERSION,

                "model_state":
                    cpu_state,

                "labels":
                    labels,

                "mean":
                    mean,

                "std":
                    std,

                "feature_size":
                    FEATURE_SIZE,

                "sequence_length":
                    SEQUENCE_LENGTH,

                "validation_accuracy":
                    best_accuracy,

                "word_count":
                    len(labels),

                "total_sequences":
                    total_sequences

            },

            MODEL_FILE
        )


        print(
            "           BEST MODEL SAVED ✅"
        )


    else:

        epochs_without_improvement += 1


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    if (

        epochs_without_improvement
        >= PATIENCE

    ):

        print()
        print(
            "Early stopping."
        )

        break


# ============================================================
# LOAD BEST MODEL
# ============================================================

best_checkpoint = torch.load(

    MODEL_FILE,

    map_location=device,

    weights_only=False
)


model.load_state_dict(

    best_checkpoint[
        "model_state"
    ]
)


model.eval()


# ============================================================
# VALIDATION RESULTS
# ============================================================

print()
print("=" * 72)
print("VALIDATION RESULTS")
print("=" * 72)


correct = 0


per_class_correct = Counter()

per_class_total = Counter()


validation_results = []


with torch.inference_mode():

    for sequence, true_index in validation_samples:

        normalized = (

            sequence
            - mean

        ) / std


        tensor = torch.tensor(

            normalized,

            dtype=torch.float32

        ).unsqueeze(
            0

        ).to(
            device
        )


        output = model(
            tensor
        )


        probabilities = torch.softmax(

            output,

            dim=1

        )[0]


        top_values, top_indexes = torch.topk(

            probabilities,

            k=min(
                3,
                len(labels)
            )
        )


        predicted_index = (
            top_indexes[0]
            .item()
        )


        confidence = (
            top_values[0]
            .item()
        )


        second_confidence = (

            top_values[1]
            .item()

            if len(top_values) > 1

            else 0.0
        )


        true_label = labels[
            true_index
        ]


        predicted_label = labels[
            predicted_index
        ]


        per_class_total[
            true_label
        ] += 1


        is_correct = (

            predicted_index
            == true_index
        )


        if is_correct:

            correct += 1


            per_class_correct[
                true_label
            ] += 1


        print(

            f"{true_label:<15} -> "
            f"{predicted_label:<15} "
            f"{confidence * 100:5.1f}%"
        )


        validation_results.append(

            {

                "true":
                    true_label,

                "predicted":
                    predicted_label,

                "correct":
                    bool(
                        is_correct
                    ),

                "confidence":
                    round(
                        confidence,
                        6
                    ),

                "second_confidence":
                    round(
                        second_confidence,
                        6
                    ),

                "margin":
                    round(

                        confidence
                        - second_confidence,

                        6
                    ),

                "top3": [

                    {

                        "word":
                            labels[
                                index.item()
                            ],

                        "confidence":
                            round(
                                value.item(),
                                6
                            )

                    }

                    for value, index
                    in zip(
                        top_values,
                        top_indexes
                    )
                ]
            }
        )


# ============================================================
# SAVE VALIDATION RESULTS
#
# We'll use this later for per-word thresholds.
# ============================================================

with open(

    VALIDATION_RESULTS_FILE,

    "w",

    encoding="utf-8"

) as file:

    json.dump(

        validation_results,

        file,

        indent=2
    )


# ============================================================
# FINAL ACCURACY
# ============================================================

final_accuracy = (

    correct

    / len(
        validation_samples
    )
)


# ============================================================
# CLASS RESULTS
# ============================================================

class_results = []


for label in labels:

    total = per_class_total[
        label
    ]


    if total == 0:

        continue


    class_accuracy = (

        per_class_correct[
            label
        ]

        / total
    )


    class_results.append(

        (
            class_accuracy,
            label,
            total
        )
    )


class_results.sort()


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 72)
print("FINAL RESULTS")
print("=" * 72)

print()
print(
    "VERSION:",
    VERSION
)

print(
    "CLASSES:",
    len(labels)
)

print(
    "TOTAL SEQUENCES:",
    total_sequences
)

print(
    "TRAIN SEQUENCES:",
    len(train_samples)
)

print(
    "VALIDATION SEQUENCES:",
    len(validation_samples)
)

print()
print(

    "BEST VALIDATION ACCURACY:",

    f"{best_accuracy * 100:.1f}%"
)

print(

    "FINAL VALIDATION ACCURACY:",

    f"{final_accuracy * 100:.1f}%"
)

print()
print(
    "MODEL SAVED:"
)

print(
    MODEL_FILE
)

print()
print(
    "VALIDATION DATA SAVED:"
)

print(
    VALIDATION_RESULTS_FILE
)


# ============================================================
# WEAKEST CLASSES
# ============================================================

print()
print("=" * 72)
print("20 WEAKEST VALIDATION CLASSES")
print("=" * 72)


for (
    class_accuracy,
    label,
    total

) in class_results[
    :20
]:

    print(

        f"{label:<15} "

        f"{class_accuracy * 100:5.1f}% "

        f"({total} validation samples)"
    )


# ============================================================
# DONE
# ============================================================

print()
print("=" * 72)
print("UPDATE COMPLETE")
print("=" * 72)

print()
print(

    f"ASL Recognition v{VERSION}"
)

print(
    f"{len(labels)} words"
)

print()
print(
    "Your live recognizer will load "
    "asl_bigru_model.pt automatically."
)

print()