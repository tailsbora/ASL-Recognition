import cv2
import mediapipe as mp
import numpy as np
import os


# ==================================================
# SETTINGS
# ==================================================

INPUT_FOLDER = "selected_videos"
OUTPUT_FOLDER = "wlasl_sequences_max"

HAND_MODEL = "../../hand_landmarker.task"
POSE_MODEL = "../../pose_landmarker.task"

SEQUENCE_LENGTH = 30
MAX_VIDEOS_PER_CLASS = 5


# ==================================================
# MEDIAPIPE SETUP
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

def hand_features(hand):

    points = np.array(
        [[p.x, p.y, p.z] for p in hand],
        dtype=np.float32
    )

    # Absolute wrist position
    wrist_absolute = points[0].copy()

    # Relative hand shape
    relative = points - points[0]

    scale = np.linalg.norm(
        relative[9]
    )

    if scale < 0.0001:
        scale = 1.0

    relative = relative / scale

    return np.concatenate(
        [
            relative.flatten(),
            wrist_absolute
        ]
    )


# ==================================================
# POSE FEATURES
# ==================================================

def pose_features(pose):

    if pose is None:
        return np.zeros(
            24,
            dtype=np.float32
        )

    # Important upper-body landmarks:
    #
    # 0  nose
    # 11 left shoulder
    # 12 right shoulder
    # 13 left elbow
    # 14 right elbow
    # 15 left wrist
    # 16 right wrist
    # 23 left hip
    # 24 right hip

    landmark_ids = [
        0,
        11,
        12,
        13,
        14,
        15,
        16,
        23
    ]

    values = []

    for idx in landmark_ids:

        p = pose[idx]

        values.extend(
            [
                p.x,
                p.y,
                p.z
            ]
        )

    return np.array(
        values,
        dtype=np.float32
    )


# ==================================================
# BUILD ONE FRAME FEATURE VECTOR
# ==================================================

def make_frame_features(
    hand_result,
    pose_result
):

    # ----------------------------------------------
    # HANDS
    # ----------------------------------------------

    hands = hand_result.hand_landmarks

    hand_output = []


    if hands:

        hands = sorted(
            hands,
            key=lambda h: h[0].x
        )

        for hand in hands[:2]:

            features = hand_features(
                hand
            )

            hand_output.append(
                features
            )


    # Each hand =
    # 63 relative coords + 3 absolute wrist coords
    HAND_FEATURE_SIZE = 66


    while len(hand_output) < 2:

        hand_output.append(
            np.zeros(
                HAND_FEATURE_SIZE,
                dtype=np.float32
            )
        )


    both_hands = np.concatenate(
        hand_output[:2]
    )


    # ----------------------------------------------
    # POSE
    # ----------------------------------------------

    pose = None

    if pose_result.pose_landmarks:

        pose = pose_result.pose_landmarks[0]


    pose_vector = pose_features(
        pose
    )


    # ----------------------------------------------
    # FINAL FRAME VECTOR
    # ----------------------------------------------

    return np.concatenate(
        [
            both_hands,
            pose_vector
        ]
    )


# ==================================================
# RESAMPLE
# ==================================================

def resample_frames(
    frames,
    target_length
):

    if len(frames) == 0:
        return []

    indices = np.linspace(
        0,
        len(frames) - 1,
        target_length
    ).astype(int)

    return [
        frames[i]
        for i in indices
    ]


# ==================================================
# PROCESS VIDEO
# ==================================================

def process_video(
    video_path
):

    cap = cv2.VideoCapture(
        video_path
    )


    if not cap.isOpened():

        print(
            "   FAILED TO OPEN VIDEO"
        )

        return None


    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    selected_frames = []


    # ==================================================
    # KNOWN FRAME COUNT
    # ==================================================

    if total_frames > 0:

        frame_indices = np.linspace(
            0,
            total_frames - 1,
            SEQUENCE_LENGTH
        ).astype(int)


        for frame_index in frame_indices:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(frame_index)
            )

            ret, frame = cap.read()

            if ret:

                selected_frames.append(
                    frame
                )


        cap.release()


    # ==================================================
    # UNKNOWN FRAME COUNT
    # ==================================================

    else:

        print(
            "   Unknown frame count"
        )

        frame_number = 0

        KEEP_EVERY = 8
        MAX_CANDIDATES = 60


        while True:

            ret, frame = cap.read()

            if not ret:
                break


            if (
                frame_number
                % KEEP_EVERY
                == 0
            ):

                selected_frames.append(
                    frame.copy()
                )


            frame_number += 1


            if (
                len(selected_frames)
                >= MAX_CANDIDATES
            ):
                break


        cap.release()


        if len(
            selected_frames
        ) == 0:

            return None


        selected_frames = resample_frames(
            selected_frames,
            SEQUENCE_LENGTH
        )


    # ==================================================
    # VALIDATE
    # ==================================================

    if len(
        selected_frames
    ) == 0:

        return None


    while (
        len(selected_frames)
        < SEQUENCE_LENGTH
    ):

        selected_frames.append(
            selected_frames[-1].copy()
        )


    selected_frames = (
        selected_frames[
            :SEQUENCE_LENGTH
        ]
    )


    # ==================================================
    # EXTRACT FEATURES
    # ==================================================

    sequence = []


    for frame in selected_frames:

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

    sign_input = os.path.join(
        INPUT_FOLDER,
        sign
    )


    if not os.path.isdir(
        sign_input
    ):
        continue


    sign_output = os.path.join(
        OUTPUT_FOLDER,
        sign
    )


    os.makedirs(
        sign_output,
        exist_ok=True
    )


    # SKIP WEBM
    videos = [
        filename
        for filename
        in os.listdir(
            sign_input
        )
        if filename.lower().endswith(
            (
                ".mp4",
                ".mov",
                ".avi"
            )
        )
    ]


    videos = sorted(
        videos
    )


    videos = videos[
        :MAX_VIDEOS_PER_CLASS
    ]


    print()
    print(
        "=" * 50
    )
    print(
        "SIGN:",
        sign
    )
    print(
        "=" * 50
    )


    success = 0


    for i, filename in enumerate(
        videos
    ):

        path = os.path.join(
            sign_input,
            filename
        )


        print(
            f"Processing {i + 1}/{len(videos)}:",
            filename
        )


        try:

            sequence = (
                process_video(
                    path
                )
            )


            if sequence is None:

                print(
                    "   FAILED"
                )

                continue


            output_file = os.path.join(
                sign_output,
                f"{i:03d}.npy"
            )


            np.save(
                output_file,
                sequence
            )


            success += 1


            print(
                "   OK",
                sequence.shape
            )


        except Exception as error:

            print(
                "   ERROR:",
                error
            )


    print(
        f"{sign}: {success}/{len(videos)} successful"
    )


hand_detector.close()
pose_detector.close()


print()
print(
    "=" * 50
)
print(
    "DONE"
)
print(
    "=" * 50
)