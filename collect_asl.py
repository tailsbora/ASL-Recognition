import cv2
import mediapipe as mp
import numpy as np
import os


# ==================================================
# SETTINGS
# ==================================================

HAND_MODEL = "hand_landmarker.task"
DATA_FOLDER = "asl_data"

# How many frames make one sign example
SEQUENCE_LENGTH = 30

# How many examples you want to collect for each sign
TARGET_SAMPLES = 40


# ==================================================
# ASK WHICH SIGN
# ==================================================

label = input("Which ASL sign are you recording? ").strip().upper()

if not label:
    print("You need to enter a sign name.")
    exit()

# Makes names like THANK YOU become THANK_YOU
label = label.replace(" ", "_")


save_folder = os.path.join(
    DATA_FOLDER,
    label
)

os.makedirs(
    save_folder,
    exist_ok=True
)


# ==================================================
# COUNT EXISTING SAMPLES
# ==================================================

existing_samples = [
    f for f in os.listdir(save_folder)
    if f.endswith(".npy")
]

sample_count = len(existing_samples)


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
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = HandLandmarker.create_from_options(options)


# ==================================================
# NORMALIZE ONE HAND
# ==================================================

def normalize_hand(hand):

    points = np.array(
        [
            [p.x, p.y, p.z]
            for p in hand
        ],
        dtype=np.float32
    )

    # Wrist becomes origin
    points = points - points[0]

    # Scale using wrist -> middle finger base
    scale = np.linalg.norm(
        points[9]
    )

    if scale < 0.0001:
        return None

    points = points / scale

    return points.flatten()


# ==================================================
# CREATE FEATURE VECTOR FOR FRAME
# ==================================================

def make_frame_features(hands):

    # No hands visible
    if not hands:
        return None

    # Sort left-to-right on screen
    hands = sorted(
        hands,
        key=lambda h: h[0].x
    )

    normalized = []

    for hand in hands[:2]:

        features = normalize_hand(hand)

        if features is not None:
            normalized.append(features)

    if len(normalized) == 0:
        return None

    # One hand visible -> second hand is zeros
    if len(normalized) == 1:

        normalized.append(
            np.zeros(
                63,
                dtype=np.float32
            )
        )

    # 63 + 63 = 126 features per frame
    return np.concatenate(
        normalized[:2]
    )


# ==================================================
# CAMERA
# ==================================================

cap = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)

if not cap.isOpened():
    print("Could not open camera.")
    exit()


frame_number = 0

recording = False
sequence = []


print()
print("Controls:")
print("SPACE = record one sign example")
print("Q / ESC = quit")
print()
print(f"Sign: {label}")
print(f"Already recorded: {sample_count}/{TARGET_SAMPLES}")
print()


# ==================================================
# MAIN LOOP
# ==================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(
        frame,
        1
    )

    height, width, _ = frame.shape


    # --------------------------------------------------
    # MEDIAPIPE IMAGE
    # --------------------------------------------------

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp = frame_number * 33
    frame_number += 1


    result = detector.detect_for_video(
        mp_image,
        timestamp
    )


    # --------------------------------------------------
    # DRAW HAND POINTS
    # --------------------------------------------------

    if result.hand_landmarks:

        for hand in result.hand_landmarks:

            for landmark in hand:

                x = int(
                    landmark.x * width
                )

                y = int(
                    landmark.y * height
                )

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 0, 255),
                    -1
                )


    # --------------------------------------------------
    # RECORDING
    # --------------------------------------------------

    if recording:

        features = make_frame_features(
            result.hand_landmarks
        )

        if features is not None:

            sequence.append(
                features
            )


        # Once we have enough frames
        if len(sequence) >= SEQUENCE_LENGTH:

            sequence_array = np.array(
                sequence[:SEQUENCE_LENGTH],
                dtype=np.float32
            )

            filename = os.path.join(
                save_folder,
                f"{sample_count:04d}.npy"
            )

            np.save(
                filename,
                sequence_array
            )

            sample_count += 1

            print(
                f"Saved {label} sample "
                f"{sample_count}/{TARGET_SAMPLES}"
            )

            sequence = []
            recording = False


    # ==================================================
    # UI
    # ==================================================

    BAR_HEIGHT = 100

    y_start = height - BAR_HEIGHT

    cv2.rectangle(
        frame,
        (0, y_start),
        (width, height),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        frame,
        f"{label}   Samples: {sample_count}/{TARGET_SAMPLES}",
        (15, y_start + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    if recording:

        status = (
            f"RECORDING "
            f"{len(sequence)}/{SEQUENCE_LENGTH}"
        )

        color = (
            0,
            255,
            0
        )

    else:

        status = "Press SPACE to record one example"

        color = (
            200,
            200,
            200
        )


    cv2.putText(
        frame,
        status,
        (15, y_start + 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )


    cv2.putText(
        frame,
        "Q: quit",
        (15, y_start + 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (150, 150, 150),
        1
    )


    cv2.imshow(
        "ASL Sign Collector",
        frame
    )


    # ==================================================
    # KEYS
    # ==================================================

    key = cv2.waitKey(10) & 0xFF


    # Quit
    if (
        key == ord("q")
        or key == ord("Q")
        or key == 27
    ):
        break


    # SPACE
    if key == 32:

        if not recording:

            sequence = []
            recording = True

            print(
                f"Recording {label}..."
            )


# ==================================================
# CLEANUP
# ==================================================

cap.release()

cv2.destroyAllWindows()

detector.close()