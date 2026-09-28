import cv2
import mediapipe as mp
import numpy as np
import os
import json
import re


# ==================================================
# SETTINGS
# ==================================================

INPUT_FOLDER = "selected_videos"
OUTPUT_FOLDER = "wlasl_sequences_final"

JSON_FILE = "WLASL_v0.3.json"

HAND_MODEL = "../../hand_landmarker.task"
POSE_MODEL = "../../pose_landmarker.task"

SEQUENCE_LENGTH = 30


# ==================================================
# LOAD WLASL METADATA
# ==================================================

with open(JSON_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)


video_metadata = {}

for entry in metadata:

    sign = entry["gloss"].lower().strip()

    for instance in entry["instances"]:

        video_id = str(
            instance["video_id"]
        )

        video_metadata[video_id] = {
            "sign": sign,
            "start": instance.get(
                "frame_start",
                1
            ),
            "end": instance.get(
                "frame_end",
                -1
            )
        }


# ==================================================
# MEDIAPIPE
# ==================================================

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions


hand_options = HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=HAND_MODEL
    ),

    running_mode=VisionRunningMode.IMAGE,

    num_hands=2,

    min_hand_detection_confidence=0.4,
    min_hand_presence_confidence=0.4,
    min_tracking_confidence=0.4
)


pose_options = PoseLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=POSE_MODEL
    ),

    running_mode=VisionRunningMode.IMAGE,

    num_poses=1,

    min_pose_detection_confidence=0.4,
    min_pose_presence_confidence=0.4,
    min_tracking_confidence=0.4
)


hand_detector = HandLandmarker.create_from_options(
    hand_options
)

pose_detector = PoseLandmarker.create_from_options(
    pose_options
)


# ==================================================
# HAND FEATURES
# ==================================================

def make_hand_vector(hand):

    points = np.array(
        [
            [p.x, p.y, p.z]
            for p in hand
        ],
        dtype=np.float32
    )

    wrist = points[0].copy()

    relative = (
        points - wrist
    )

    scale = np.linalg.norm(
        relative[9]
    )

    if scale < 0.0001:
        scale = 1.0

    relative /= scale

    # 63 hand-shape values
    # + 3 absolute wrist coordinates

    return np.concatenate(
        [
            relative.flatten(),
            wrist
        ]
    )


# ==================================================
# POSE FEATURES
# ==================================================

def make_pose_vector(pose):

    # Face + upper body
    #
    # 0 nose
    # 1-8 eyes / ears
    # 9-10 mouth
    # 11-12 shoulders
    # 13-14 elbows
    # 15-16 wrists
    # 23-24 hips

    landmark_ids = [
        0,
        1, 2,
        3, 4,
        5, 6,
        7, 8,
        9, 10,
        11, 12,
        13, 14,
        15, 16,
        23, 24
    ]

    # 19 landmarks × XYZ = 57

    if pose is None:

        return np.zeros(
            len(landmark_ids) * 3,
            dtype=np.float32
        )


    points = np.array(
        [
            [
                pose[i].x,
                pose[i].y,
                pose[i].z
            ]
            for i in landmark_ids
        ],
        dtype=np.float32
    )


    # ----------------------------------------------
    # Normalize relative to body
    # ----------------------------------------------

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
        points - center
    ) / shoulder_width


    return points.flatten()


# ==================================================
# FRAME FEATURES
# ==================================================

def make_frame_features(
    hand_result,
    pose_result
):

    HAND_SIZE = 66


    # --------------------------------------------------
    # HANDS
    # --------------------------------------------------

    left_hand = np.zeros(
        HAND_SIZE,
        dtype=np.float32
    )

    right_hand = np.zeros(
        HAND_SIZE,
        dtype=np.float32
    )


    hands = (
        hand_result.hand_landmarks
    )


    handedness = (
        hand_result.handedness
    )


    for hand, hand_info in zip(
        hands,
        handedness
    ):

        vector = make_hand_vector(
            hand
        )


        if len(hand_info) > 0:

            label = (
                hand_info[0]
                .category_name
                .lower()
            )


            if label == "left":
                left_hand = vector

            elif label == "right":
                right_hand = vector


    # --------------------------------------------------
    # POSE
    # --------------------------------------------------

    pose = None

    if pose_result.pose_landmarks:

        pose = (
            pose_result
            .pose_landmarks[0]
        )


    pose_vector = make_pose_vector(
        pose
    )


    return np.concatenate(
        [
            left_hand,
            right_hand,
            pose_vector
        ]
    )


# ==================================================
# GET VIDEO ID
# ==================================================

def get_video_id(filename):

    match = re.match(
        r"(\d+)",
        filename
    )

    if not match:
        return None

    return match.group(1)


# ==================================================
# READ CORRECT SIGN FRAMES
# ==================================================

def get_selected_frames(
    video_path,
    video_id
):

    cap = cv2.VideoCapture(
        video_path
    )


    if not cap.isOpened():

        print(
            "   FAILED TO OPEN"
        )

        return []


    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    info = video_metadata.get(
        video_id
    )


    start_frame = 1
    end_frame = -1


    if info:

        start_frame = (
            info["start"]
            if info["start"] is not None
            else 1
        )

        end_frame = (
            info["end"]
            if info["end"] is not None
            else -1
        )


    # ==================================================
    # ALREADY ISOLATED VIDEO
    # ==================================================

    if (
        start_frame <= 1
        and end_frame == -1
    ):

        first = 0

        last = max(
            0,
            total_frames - 1
        )


    # ==================================================
    # WLASL RANGE
    # ==================================================

    else:

        # WLASL frame numbers are 1-based

        first = max(
            0,
            int(start_frame) - 1
        )


        if end_frame == -1:

            last = max(
                first,
                total_frames - 1
            )

        else:

            last = min(
                total_frames - 1,
                int(end_frame) - 1
            )


    if last < first:

        last = max(
            first,
            total_frames - 1
        )


    # ==================================================
    # CHOOSE EXACTLY 30 FRAMES FROM SIGN WINDOW
    # ==================================================

    frame_indices = np.linspace(
        first,
        last,
        SEQUENCE_LENGTH
    ).astype(int)


    frames = []


    for index in frame_indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(index)
        )

        ret, frame = cap.read()


        if ret:

            frames.append(
                frame
            )


    cap.release()


    if len(frames) == 0:
        return []


    while (
        len(frames)
        < SEQUENCE_LENGTH
    ):

        frames.append(
            frames[-1].copy()
        )


    return frames[
        :SEQUENCE_LENGTH
    ]


# ==================================================
# PROCESS ONE VIDEO
# ==================================================

def process_video(
    video_path,
    video_id
):

    frames = get_selected_frames(
        video_path,
        video_id
    )


    if len(frames) == 0:
        return None


    sequence = []


    for frame in frames:

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
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


        features = (
            make_frame_features(
                hand_result,
                pose_result
            )
        )


        sequence.append(
            features
        )


    return np.array(
        sequence,
        dtype=np.float32
    )


# ==================================================
# MAIN
# ==================================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


for sign in sorted(
    os.listdir(
        INPUT_FOLDER
    )
):

    sign_folder = os.path.join(
        INPUT_FOLDER,
        sign
    )


    if not os.path.isdir(
        sign_folder
    ):
        continue


    output_folder = os.path.join(
        OUTPUT_FOLDER,
        sign
    )


    os.makedirs(
        output_folder,
        exist_ok=True
    )


    # Still skip WEBM for now

    videos = sorted(
        [
            filename
            for filename
            in os.listdir(
                sign_folder
            )
            if filename.lower().endswith(
                (
                    ".mp4",
                    ".mov",
                    ".avi"
                )
            )
        ]
    )


    print()
    print("=" * 60)
    print(
        "SIGN:",
        sign,
        "-",
        len(videos),
        "videos"
    )
    print("=" * 60)


    success = 0


    for i, filename in enumerate(
        videos
    ):

        video_id = get_video_id(
            filename
        )


        if video_id is None:

            print(
                "Skipping:",
                filename
            )

            continue


        video_path = os.path.join(
            sign_folder,
            filename
        )


        info = video_metadata.get(
            video_id
        )


        if info:

            print(
                f"{i + 1}/{len(videos)}",
                filename,
                f"[{info['start']} -> {info['end']}]"
            )

        else:

            print(
                f"{i + 1}/{len(videos)}",
                filename
            )


        try:

            sequence = process_video(
                video_path,
                video_id
            )


            if sequence is None:

                print(
                    "   FAILED"
                )

                continue


            output_path = os.path.join(
                output_folder,
                f"{video_id}.npy"
            )


            np.save(
                output_path,
                sequence
            )


            print(
                "   OK",
                sequence.shape
            )


            success += 1


        except Exception as error:

            print(
                "   ERROR:",
                error
            )


    print(
        f"{sign}: {success}/{len(videos)} successful"
    )


# ==================================================
# CLEANUP
# ==================================================

hand_detector.close()
pose_detector.close()


print()
print("=" * 60)
print("DONE")
print("=" * 60)