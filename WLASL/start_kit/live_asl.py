import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn

from collections import deque, Counter


# ============================================================
# SETTINGS
# ============================================================

MODEL_FILE = "asl_bigru_model.pt"

HAND_MODEL = "../../hand_landmarker.task"
POSE_MODEL = "../../pose_landmarker.task"

CAMERA_INDEX = 0

SEQUENCE_LENGTH = 30
FEATURE_SIZE = 189

CONFIDENCE_THRESHOLD = 0.65

# Predictions used for stabilization
STABILITY_WINDOW = 5
STABILITY_REQUIRED = 4


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("DEVICE:", device)


# ============================================================
# SAME NEURAL NETWORK USED DURING TRAINING
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

        attention_scores = self.attention(
            output
        )

        attention_weights = torch.softmax(
            attention_scores,
            dim=1
        )

        context = torch.sum(
            output * attention_weights,
            dim=1
        )

        logits = self.classifier(
            context
        )

        return logits


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_FILE,
    map_location=device,
    weights_only=False
)

labels = checkpoint["labels"]

mean = checkpoint["mean"]
std = checkpoint["std"]

model = SignLanguageModel(
    FEATURE_SIZE,
    len(labels)
).to(device)

model.load_state_dict(
    checkpoint["model_state"]
)

model.eval()


print()
print("Loaded signs:")

for label in labels:
    print(" ", label)

print()


# ============================================================
# MEDIAPIPE
# ============================================================

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


# ============================================================
# HAND FEATURES
# EXACT SAME FORMAT AS TRAINING DATA
# ============================================================

def make_hand_vector(hand):

    points = np.array(
        [
            [p.x, p.y, p.z]
            for p in hand
        ],
        dtype=np.float32
    )

    wrist = points[0].copy()

    relative = points - wrist

    scale = np.linalg.norm(
        relative[9]
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

def make_pose_vector(pose):

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


# ============================================================
# CREATE FEATURES FOR ONE CAMERA FRAME
# ============================================================

def make_frame_features(
    hand_result,
    pose_result
):

    HAND_SIZE = 66

    left_hand = np.zeros(
        HAND_SIZE,
        dtype=np.float32
    )

    right_hand = np.zeros(
        HAND_SIZE,
        dtype=np.float32
    )

    hands = hand_result.hand_landmarks

    handedness = hand_result.handedness


    for hand, hand_info in zip(
        hands,
        handedness
    ):

        vector = make_hand_vector(
            hand
        )

        if len(hand_info) == 0:
            continue

        label = (
            hand_info[0]
            .category_name
            .lower()
        )

        if label == "left":

            left_hand = vector

        elif label == "right":

            right_hand = vector


    pose = None

    if pose_result.pose_landmarks:

        pose = pose_result.pose_landmarks[0]


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


# ============================================================
# PREDICT SEQUENCE
# ============================================================

def predict(sequence):

    sequence = np.array(
        sequence,
        dtype=np.float32
    )

    sequence = (
        sequence - mean
    ) / std


    tensor = torch.tensor(
        sequence,
        dtype=torch.float32
    )

    tensor = tensor.unsqueeze(
        0
    ).to(device)


    with torch.no_grad():

        logits = model(
            tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]


    confidence, index = torch.max(
        probabilities,
        dim=0
    )


    prediction = labels[
        index.item()
    ]


    return (
        prediction,
        confidence.item(),
        probabilities.cpu().numpy()
    )


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)


if not cap.isOpened():

    print("Could not open camera.")

    hand_detector.close()
    pose_detector.close()

    raise SystemExit


# ============================================================
# BUFFERS
# ============================================================

sequence_buffer = deque(
    maxlen=SEQUENCE_LENGTH
)

prediction_buffer = deque(
    maxlen=STABILITY_WINDOW
)


current_prediction = "WAITING"
current_confidence = 0.0

stable_prediction = ""


frame_counter = 0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break


    # --------------------------------------------------------
    # IMPORTANT:
    # Analyze ORIGINAL frame.
    #
    # Only mirror the DISPLAY.
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


    hand_result = hand_detector.detect(
        mp_image
    )


    pose_result = pose_detector.detect(
        mp_image
    )


    features = make_frame_features(
        hand_result,
        pose_result
    )


    sequence_buffer.append(
        features
    )


    # --------------------------------------------------------
    # PREDICT ONCE BUFFER HAS 30 FRAMES
    # --------------------------------------------------------

    if len(sequence_buffer) == SEQUENCE_LENGTH:

        # Predict every 3 camera frames
        # instead of every frame
        if frame_counter % 3 == 0:

            (
                prediction,
                confidence,
                probabilities
            ) = predict(
                sequence_buffer
            )


            current_prediction = prediction

            current_confidence = confidence


            if confidence >= CONFIDENCE_THRESHOLD:

                prediction_buffer.append(
                    prediction
                )

            else:

                prediction_buffer.append(
                    "UNKNOWN"
                )


            # ------------------------------------------------
            # STABILITY CHECK
            # ------------------------------------------------

            if len(prediction_buffer) == STABILITY_WINDOW:

                counts = Counter(
                    prediction_buffer
                )

                best_word, count = (
                    counts.most_common(1)[0]
                )


                if (
                    best_word != "UNKNOWN"
                    and count >= STABILITY_REQUIRED
                ):

                    stable_prediction = best_word


    frame_counter += 1


    # ========================================================
    # DISPLAY
    # ========================================================

    display = cv2.flip(
        frame,
        1
    )


    height, width = display.shape[:2]


    # --------------------------------------------------------
    # BOTTOM UI
    # --------------------------------------------------------

    panel_height = 125


    overlay = display.copy()


    cv2.rectangle(

        overlay,

        (
            0,
            height - panel_height
        ),

        (
            width,
            height
        ),

        (
            20,
            20,
            20
        ),

        -1
    )


    cv2.addWeighted(
        overlay,
        0.75,
        display,
        0.25,
        0,
        display
    )


    # --------------------------------------------------------
    # BUFFER STATUS
    # --------------------------------------------------------

    cv2.putText(

        display,

        f"Frames: {len(sequence_buffer)}/{SEQUENCE_LENGTH}",

        (
            20,
            height - 90
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (
            180,
            180,
            180
        ),

        1,

        cv2.LINE_AA
    )


    # --------------------------------------------------------
    # RAW MODEL PREDICTION
    # --------------------------------------------------------

    if len(sequence_buffer) == SEQUENCE_LENGTH:

        text = (
            f"{current_prediction.upper()}  "
            f"{current_confidence * 100:.1f}%"
        )

    else:

        text = "WATCHING..."


    cv2.putText(

        display,

        text,

        (
            20,
            height - 55
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.9,

        (
            255,
            255,
            255
        ),

        2,

        cv2.LINE_AA
    )


    # --------------------------------------------------------
    # STABLE WORD
    # --------------------------------------------------------

    if stable_prediction:

        cv2.putText(

            display,

            "CONFIRMED: "
            + stable_prediction.upper(),

            (
                20,
                height - 18
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.75,

            (
                0,
                255,
                0
            ),

            2,

            cv2.LINE_AA
        )


    # --------------------------------------------------------
    # HAND DOTS
    # --------------------------------------------------------

    for hand in hand_result.hand_landmarks:

        for landmark in hand:

            x = int(
                (1 - landmark.x)
                * width
            )

            y = int(
                landmark.y
                * height
            )


            cv2.circle(

                display,

                (
                    x,
                    y
                ),

                3,

                (
                    0,
                    0,
                    255
                ),

                -1
            )


    cv2.imshow(
        "ASL Neural Network",
        display
    )


    key = cv2.waitKey(1) & 0xFF


    # Q = quit
    if key == ord("q"):
        break


    # R = reset current sequence
    if key == ord("r"):

        sequence_buffer.clear()

        prediction_buffer.clear()

        stable_prediction = ""

        current_prediction = "WAITING"

        current_confidence = 0.0


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

hand_detector.close()

pose_detector.close()

print("Closed.")