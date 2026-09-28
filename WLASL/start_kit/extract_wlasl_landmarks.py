import cv2
import mediapipe as mp
import numpy as np
import os


# ==================================================
# SETTINGS
# ==================================================

INPUT_FOLDER = "selected_videos"
OUTPUT_FOLDER = "wlasl_sequences"

HAND_MODEL = "../../hand_landmarker.task"

SEQUENCE_LENGTH = 30
MAX_VIDEOS_PER_CLASS = 5


# ==================================================
# MEDIAPIPE
# ==================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=HAND_MODEL
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.4,
    min_hand_presence_confidence=0.4,
    min_tracking_confidence=0.4
)

detector = HandLandmarker.create_from_options(options)


# ==================================================
# NORMALIZE ONE HAND
# ==================================================

def normalize_hand(hand):

    points = np.array(
        [[p.x, p.y, p.z] for p in hand],
        dtype=np.float32
    )

    # Wrist = origin
    points = points - points[0]

    # Scale using wrist -> middle finger base
    scale = np.linalg.norm(points[9])

    if scale < 0.0001:
        return None

    points = points / scale

    return points.flatten()


# ==================================================
# CREATE FEATURES FOR ONE FRAME
# ==================================================

def make_frame_features(hands):

    # No hands detected
    if not hands:
        return np.zeros(
            126,
            dtype=np.float32
        )

    # Sort left-to-right
    hands = sorted(
        hands,
        key=lambda hand: hand[0].x
    )

    output = []

    for hand in hands[:2]:

        normalized = normalize_hand(hand)

        if normalized is not None:
            output.append(normalized)

    # No usable hands
    if len(output) == 0:

        return np.zeros(
            126,
            dtype=np.float32
        )

    # Only one hand
    if len(output) == 1:

        output.append(
            np.zeros(
                63,
                dtype=np.float32
            )
        )

    return np.concatenate(
        output[:2]
    )


# ==================================================
# RESAMPLE LIST TO EXACT NUMBER OF FRAMES
# ==================================================

def resample_frames(frames, target_length):

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
# PROCESS ONE VIDEO
# ==================================================

def process_video(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        print("   FAILED TO OPEN VIDEO")

        return None


    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    selected_frames = []


    # ==================================================
    # NORMAL CASE:
    # VIDEO REPORTS FRAME COUNT
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
    # UNKNOWN FRAME COUNT FALLBACK
    # ==================================================

    else:

        print(
            "   Unknown frame count - fast fallback"
        )

        frame_number = 0

        KEEP_EVERY = 8

        MAX_CANDIDATES = 60


        while True:

            ret, frame = cap.read()

            if not ret:
                break


            if frame_number % KEEP_EVERY == 0:

                selected_frames.append(
                    frame.copy()
                )


            frame_number += 1


            if len(selected_frames) >= MAX_CANDIDATES:
                break


        cap.release()


        if len(selected_frames) == 0:

            print("   NO FRAMES FOUND")

            return None


        selected_frames = resample_frames(
            selected_frames,
            SEQUENCE_LENGTH
        )


    # ==================================================
    # VALIDATE FRAMES
    # ==================================================

    if len(selected_frames) == 0:

        print("   NO VALID FRAMES")

        return None


    # If fewer than 30 frames were successfully read,
    # repeat the final frame.
    while len(selected_frames) < SEQUENCE_LENGTH:

        selected_frames.append(
            selected_frames[-1].copy()
        )


    selected_frames = selected_frames[
        :SEQUENCE_LENGTH
    ]


    # ==================================================
    # MEDIAPIPE
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


        # IMAGE MODE:
        # NO TIMESTAMPS REQUIRED
        result = detector.detect(
            mp_image
        )


        features = make_frame_features(
            result.hand_landmarks
        )


        sequence.append(
            features
        )


    sequence = np.array(
        sequence,
        dtype=np.float32
    )


    return sequence


# ==================================================
# CREATE OUTPUT FOLDER
# ==================================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ==================================================
# PROCESS EVERY SIGN
# ==================================================

for sign in sorted(
    os.listdir(INPUT_FOLDER)
):


    sign_input = os.path.join(
        INPUT_FOLDER,
        sign
    )


    if not os.path.isdir(sign_input):
        continue


    sign_output = os.path.join(
        OUTPUT_FOLDER,
        sign
    )


    os.makedirs(
        sign_output,
        exist_ok=True
    )


    # ==================================================
    # ONLY USE MP4 / MOV / AVI
    # SKIP WEBM
    # ==================================================

    videos = [
        filename
        for filename in os.listdir(sign_input)
        if filename.lower().endswith(
            (
                ".mp4",
                ".mov",
                ".avi"
            )
        )
    ]


    videos = sorted(videos)


    # Maximum 5 per sign for this first test
    videos = videos[
        :MAX_VIDEOS_PER_CLASS
    ]


    print()
    print("=" * 50)
    print("SIGN:", sign)
    print("=" * 50)


    if len(videos) == 0:

        print("NO SUPPORTED VIDEOS")

        continue


    success = 0


    # ==================================================
    # PROCESS VIDEOS
    # ==================================================

    for i, filename in enumerate(videos):


        video_path = os.path.join(
            sign_input,
            filename
        )


        print(
            f"Processing {i + 1}/{len(videos)}:",
            filename
        )


        try:

            sequence = process_video(
                video_path
            )


            if sequence is None:

                print("   FAILED")

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
                "   Skipping this video..."
            )


    print(
        f"{sign}: {success}/{len(videos)} successful"
    )


# ==================================================
# CLEANUP
# ==================================================

detector.close()


print()
print("=" * 50)
print("DONE")
print("=" * 50)