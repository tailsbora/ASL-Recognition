import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import pyttsx3

import threading
import queue
import time
import os

from collections import deque, Counter


# ============================================================
# SETTINGS
# ============================================================

MODEL_FILE = "WLASL/start_kit/asl_bigru_model.pt"

HAND_MODEL_FILE = "hand_landmarker.task"
POSE_MODEL_FILE = "pose_landmarker.task"

CAMERA_INDEX = 0

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# MediaPipe analyzes a smaller image for lower latency.
# Display still uses full camera resolution.
PROCESS_WIDTH = 512

SEQUENCE_LENGTH = 30
FEATURE_SIZE = 189

WINDOW_NAME = "ASL"


# ============================================================
# RECOGNITION
# ============================================================

# Normal acceptance
CONFIDENCE_THRESHOLD = 0.75

STABILITY_WINDOW = 3
STABILITY_REQUIRED = 2


# Fast acceptance
#
# >= 90%
# same prediction for 2 consecutive processed camera frames
# -> speak immediately
FAST_CONFIDENCE = 0.90
FAST_STREAK_REQUIRED = 2


# After hands return from a SHORT disappearance,
# only collect this many fresh hand frames before allowing
# recognition again.
HAND_RETURN_WARMUP_FRAMES = 5


# If hands stay gone longer than this,
# old gesture history is considered stale.
NO_HAND_BUFFER_RESET_SECONDS = 1.20


# Hands need to be genuinely absent this long
# before the same word can be spoken again.
NO_HAND_RELEASE_SECONDS = 0.20


# Custom Point A
POINT_A_OFFSET = 15


# ============================================================
# DISPLAY
# ============================================================

SHOW_POSE = True

fullscreen = False


# ============================================================
# CHECK FILES
# ============================================================

for path in [
    MODEL_FILE,
    HAND_MODEL_FILE,
    POSE_MODEL_FILE
]:

    if not os.path.exists(path):

        print(
            "Missing file:",
            path
        )

        raise SystemExit


# ============================================================
# OPENCV OPTIMIZATION
# ============================================================

cv2.setUseOptimized(True)


# ============================================================
# PYTORCH
# ============================================================

device = torch.device(

    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    "DEVICE:",
    device
)


# Avoid PyTorch taking every CPU thread while
# MediaPipe is also trying to run.
if device.type == "cpu":

    cpu_count = (
        os.cpu_count()
        or 4
    )

    torch_threads = max(
        1,
        min(
            4,
            cpu_count // 2
        )
    )

    torch.set_num_threads(
        torch_threads
    )

    print(
        "PyTorch CPU threads:",
        torch_threads
    )


# ============================================================
# NEURAL NETWORK
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

            output
            * attention_weights,

            dim=1
        )


        return self.classifier(
            context
        )


# ============================================================
# LOAD MODEL
# ============================================================

checkpoint = torch.load(

    MODEL_FILE,

    map_location=device,

    weights_only=False
)


labels = checkpoint[
    "labels"
]


mean = np.asarray(

    checkpoint["mean"],

    dtype=np.float32
)


std = np.asarray(

    checkpoint["std"],

    dtype=np.float32
)


model = SignLanguageModel(

    FEATURE_SIZE,

    len(labels)

).to(device)


model.load_state_dict(

    checkpoint[
        "model_state"
    ]
)


model.eval()


# Preload normalization directly onto device.
mean_tensor = torch.from_numpy(
    mean
).to(device)


std_tensor = torch.from_numpy(
    std
).to(device)


print(
    "Loaded signs:",
    len(labels)
)


# ============================================================
# TEXT TO SPEECH
# ============================================================

speech_queue = queue.Queue()


def speech_worker():

    while True:

        text = speech_queue.get()


        if text is None:

            speech_queue.task_done()

            break


        try:

            # Fresh engine avoids the Windows bug where
            # pyttsx3 sometimes works only once.
            engine = pyttsx3.init()


            engine.setProperty(
                "rate",
                190
            )


            engine.say(
                text
            )


            engine.runAndWait()


            engine.stop()


            del engine


        except Exception as error:

            print(
                "TTS error:",
                error
            )


        speech_queue.task_done()


speech_thread = threading.Thread(

    target=speech_worker,

    daemon=True
)


speech_thread.start()


def speak(
    text
):

    # Discard speech that is still waiting in queue.
    # The newest recognized word matters most.
    while not speech_queue.empty():

        try:

            speech_queue.get_nowait()

            speech_queue.task_done()

        except queue.Empty:

            break


    speech_queue.put(
        text
    )


# ============================================================
# MEDIAPIPE
# ============================================================

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


# VIDEO mode uses tracking between frames and is much
# faster than running completely independent IMAGE detections.
hand_options = HandLandmarkerOptions(

    base_options=BaseOptions(

        model_asset_path=(
            HAND_MODEL_FILE
        )
    ),

    running_mode=(
        RunningMode.VIDEO
    ),

    num_hands=2,

    min_hand_detection_confidence=0.4,

    min_hand_presence_confidence=0.4,

    min_tracking_confidence=0.4
)


pose_options = PoseLandmarkerOptions(

    base_options=BaseOptions(

        model_asset_path=(
            POSE_MODEL_FILE
        )
    ),

    running_mode=(
        RunningMode.VIDEO
    ),

    num_poses=1,

    min_pose_detection_confidence=0.4,

    min_pose_presence_confidence=0.4,

    min_tracking_confidence=0.4
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
# HAND CONNECTIONS
# ============================================================

HAND_CONNECTIONS = [

    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    # (0, 5) intentionally removed

    (5, 6),
    (6, 7),
    (7, 8),

    (5, 9),

    (9, 10),
    (10, 11),
    (11, 12),

    (9, 13),

    (13, 14),
    (14, 15),
    (15, 16),

    (13, 17),

    (17, 18),
    (18, 19),
    (19, 20),

    (0, 17)
]


# ============================================================
# POSE POINTS ACTUALLY USED BY MODEL
# ============================================================

POSE_USED_IDS = [

    # Face
    0,

    1, 2, 3,

    4, 5, 6,

    7, 8,

    9, 10,

    # Shoulders
    11, 12,

    # Elbows
    13, 14,

    # Wrists
    15, 16,

    # Hips
    23, 24
]


# ============================================================
# POSE VISUAL CONNECTIONS
# ============================================================

POSE_CONNECTIONS = [

    # Face — left side
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),

    # Face — right side
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),

    # Nose / mouth
    (0, 9),
    (0, 10),
    (9, 10),

    # Shoulders
    (11, 12),

    # Left arm
    (11, 13),
    (13, 15),

    # Right arm
    (12, 14),
    (14, 16),

    # Torso
    (11, 23),
    (12, 24),

    # Hips
    (23, 24)
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
# COMPLETE FRAME FEATURES
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


    return np.concatenate(

        [
            left_hand,

            right_hand,

            pose_vector
        ]
    )


# ============================================================
# PREDICTION
# ============================================================

def predict(
    sequence
):

    sequence_array = np.asarray(

        sequence,

        dtype=np.float32
    )


    tensor = torch.from_numpy(

        sequence_array

    ).unsqueeze(
        0

    ).to(
        device
    )


    # Normalize directly on device.
    tensor = (

        tensor
        - mean_tensor

    ) / std_tensor


    with torch.inference_mode():

        logits = model(
            tensor
        )


        probabilities = torch.softmax(

            logits,

            dim=1

        )[0]


    values, indices = torch.topk(

        probabilities,

        k=2
    )


    prediction = labels[
        indices[0].item()
    ]


    confidence = (
        values[0].item()
    )


    second_confidence = (
        values[1].item()
    )


    return (

        prediction,

        confidence,

        second_confidence
    )


# ============================================================
# LOW-LATENCY CAMERA
# ============================================================

class LatestFrameCamera:

    def __init__(
        self,
        index
    ):

        self.cap = cv2.VideoCapture(

            index,

            cv2.CAP_DSHOW
        )


        if not self.cap.isOpened():

            raise RuntimeError(
                "Could not open camera."
            )


        # MJPG often lowers USB webcam latency.
        try:

            self.cap.set(

                cv2.CAP_PROP_FOURCC,

                cv2.VideoWriter_fourcc(
                    *"MJPG"
                )
            )

        except Exception:

            pass


        self.cap.set(

            cv2.CAP_PROP_FRAME_WIDTH,

            CAMERA_WIDTH
        )


        self.cap.set(

            cv2.CAP_PROP_FRAME_HEIGHT,

            CAMERA_HEIGHT
        )


        self.cap.set(

            cv2.CAP_PROP_FPS,

            CAMERA_FPS
        )


        # May be ignored by DirectShow,
        # but helps on backends that support it.
        self.cap.set(

            cv2.CAP_PROP_BUFFERSIZE,

            1
        )


        self.frame = None

        self.frame_id = 0


        self.lock = threading.Lock()


        self.running = True


        self.thread = threading.Thread(

            target=self._capture_loop,

            daemon=True
        )


        self.thread.start()


    def _capture_loop(
        self
    ):

        while self.running:

            ret, frame = (
                self.cap.read()
            )


            if not ret:

                time.sleep(
                    0.002
                )

                continue


            with self.lock:

                # Replace old image.
                # Never create a frame queue.
                self.frame = frame

                self.frame_id += 1


    def read(
        self
    ):

        with self.lock:

            if self.frame is None:

                return (
                    None,
                    None
                )


            return (

                self.frame_id,

                self.frame.copy()
            )


    def release(
        self
    ):

        self.running = False


        self.thread.join(
            timeout=0.25
        )


        self.cap.release()


# ============================================================
# RESIZE ONLY THE ANALYSIS COPY
# ============================================================

def make_analysis_frame(
    frame
):

    height, width = (
        frame.shape[:2]
    )


    if width <= PROCESS_WIDTH:

        return frame


    scale = (

        PROCESS_WIDTH
        / width
    )


    new_height = max(

        1,

        int(
            height
            * scale
        )
    )


    return cv2.resize(

        frame,

        (
            PROCESS_WIDTH,
            new_height
        ),

        interpolation=cv2.INTER_AREA
    )


# ============================================================
# DRAW HAND
# ============================================================

def draw_hand(
    display,
    hand,
    hand_label
):

    height, width = (
        display.shape[:2]
    )


    points = []


    for landmark in hand:

        # Neural network sees unmirrored image.
        # User sees mirrored display.
        x = int(

            (
                1.0
                - landmark.x
            )

            * width
        )


        y = int(

            landmark.y
            * height
        )


        points.append(
            (x, y)
        )


    # ========================================================
    # HAND LINES
    # ========================================================

    for start, end in HAND_CONNECTIONS:

        cv2.line(

            display,

            points[start],

            points[end],

            (
                255,
                255,
                0
            ),

            2,

            cv2.LINE_AA
        )


    # ========================================================
    # HAND DOTS
    # ========================================================

    for point in points:

        cv2.circle(

            display,

            point,

            5,

            (
                0,
                0,
                255
            ),

            -1,

            cv2.LINE_AA
        )


    # ========================================================
    # CUSTOM POINT A
    # ========================================================

    p3 = np.array(

        points[3],

        dtype=np.float32
    )


    p5 = np.array(

        points[5],

        dtype=np.float32
    )


    point_a = (

        (
            p3
            + p5
        )

        / 2

    ).astype(
        int
    )


    point_a[1] += (
        POINT_A_OFFSET
    )


    # Display is mirrored.
    if hand_label == "left":

        point_a[0] -= (
            POINT_A_OFFSET
        )

    else:

        point_a[0] += (
            POINT_A_OFFSET
        )


    cv2.circle(

        display,

        (
            int(
                point_a[0]
            ),

            int(
                point_a[1]
            )
        ),

        5,

        (
            0,
            0,
            255
        ),

        -1,

        cv2.LINE_AA
    )


# ============================================================
# DRAW BODY + FACE
# ============================================================

def draw_pose_overlay(
    display,
    pose_result
):

    if not pose_result.pose_landmarks:

        return


    pose = (
        pose_result
        .pose_landmarks[0]
    )


    height, width = (
        display.shape[:2]
    )


    points = {}


    for index in POSE_USED_IDS:

        landmark = (
            pose[index]
        )


        x = int(

            (
                1.0
                - landmark.x
            )

            * width
        )


        y = int(

            landmark.y
            * height
        )


        points[index] = (
            x,
            y
        )


    # ========================================================
    # BODY / FACE LINES
    # ========================================================

    for start, end in POSE_CONNECTIONS:

        if (

            start in points

            and

            end in points

        ):

            cv2.line(

                display,

                points[start],

                points[end],

                (
                    255,
                    255,
                    0
                ),

                2,

                cv2.LINE_AA
            )


    # ========================================================
    # BODY / FACE DOTS
    # ========================================================

    for point in points.values():

        cv2.circle(

            display,

            point,

            4,

            (
                0,
                0,
                255
            ),

            -1,

            cv2.LINE_AA
        )


# ============================================================
# SCALE FINAL IMAGE TO WINDOW WHILE PRESERVING PROPORTIONS
# ============================================================

def fit_to_window(
    image
):

    try:

        _, _, window_width, window_height = (
            cv2.getWindowImageRect(
                WINDOW_NAME
            )
        )

    except cv2.error:

        return image


    if (

        window_width < 100

        or

        window_height < 100

    ):

        return image


    image_height, image_width = (
        image.shape[:2]
    )


    scale = min(

        window_width
        / image_width,

        window_height
        / image_height
    )


    new_width = max(

        1,

        int(
            image_width
            * scale
        )
    )


    new_height = max(

        1,

        int(
            image_height
            * scale
        )
    )


    interpolation = (

        cv2.INTER_LINEAR

        if scale > 1

        else cv2.INTER_AREA
    )


    resized = cv2.resize(

        image,

        (
            new_width,
            new_height
        ),

        interpolation=interpolation
    )


    # Black letterbox background.
    canvas = np.zeros(

        (
            window_height,
            window_width,
            3
        ),

        dtype=np.uint8
    )


    x = (
        window_width
        - new_width
    ) // 2


    y = (
        window_height
        - new_height
    ) // 2


    canvas[

        y:
        y + new_height,

        x:
        x + new_width

    ] = resized


    return canvas


# ============================================================
# CAMERA
# ============================================================

print(
    "Opening camera..."
)


try:

    camera = LatestFrameCamera(
        CAMERA_INDEX
    )


except RuntimeError as error:

    print(
        error
    )


    hand_detector.close()

    pose_detector.close()

    speech_queue.put(
        None
    )


    raise SystemExit


# ============================================================
# WINDOW
# ============================================================

cv2.namedWindow(

    WINDOW_NAME,

    cv2.WINDOW_NORMAL
    | cv2.WINDOW_FREERATIO
)


cv2.resizeWindow(

    WINDOW_NAME,

    960,

    720
)


# ============================================================
# STATE
# ============================================================

sequence_buffer = deque(

    maxlen=SEQUENCE_LENGTH
)


prediction_buffer = deque(

    maxlen=STABILITY_WINDOW
)


sentence = []


current_prediction = "WAITING"

current_confidence = 0.0

current_second_confidence = 0.0


last_raw_prediction = ""

raw_prediction_streak = 0


# A held gesture can speak only once.
# A different confirmed word can immediately replace it.
latched_word = None


no_hand_since = None


hand_return_frames = 0


last_timestamp = 0


last_processed_frame_id = -1


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    (
        captured_frame_id,
        raw_frame
    ) = camera.read()


    if raw_frame is None:

        time.sleep(
            0.001
        )

        continue


    # ========================================================
    # IMPORTANT PERFORMANCE FIX
    #
    # Do not analyze the exact same webcam frame twice.
    # ========================================================

    if (
        captured_frame_id
        == last_processed_frame_id
    ):

        time.sleep(
            0.001
        )

        continue


    last_processed_frame_id = (
        captured_frame_id
    )


    now = time.monotonic()


    # ========================================================
    # SMALLER COPY FOR MEDIAPIPE
    # ========================================================

    analysis_frame = (
        make_analysis_frame(
            raw_frame
        )
    )


    rgb = cv2.cvtColor(

        analysis_frame,

        cv2.COLOR_BGR2RGB
    )


    mp_image = mp.Image(

        image_format=(
            mp.ImageFormat.SRGB
        ),

        data=rgb
    )


    timestamp = int(

        now
        * 1000
    )


    if (
        timestamp
        <= last_timestamp
    ):

        timestamp = (
            last_timestamp
            + 1
        )


    last_timestamp = (
        timestamp
    )


    # ========================================================
    # MEDIAPIPE
    # ========================================================

    hand_result = (
        hand_detector
        .detect_for_video(
            mp_image,
            timestamp
        )
    )


    pose_result = (
        pose_detector
        .detect_for_video(
            mp_image,
            timestamp
        )
    )


    hand_present = (

        len(
            hand_result
            .hand_landmarks
        )

        > 0
    )


    # ========================================================
    # HAND PRESENT
    # ========================================================

    if hand_present:

        # ----------------------------------------------------
        # HANDS JUST RETURNED
        # ----------------------------------------------------

        if (
            no_hand_since
            is not None
        ):

            no_hand_duration = (

                now
                - no_hand_since
            )


            # Long disappearance:
            # old sequence is no longer trustworthy.
            if (

                no_hand_duration
                >=
                NO_HAND_BUFFER_RESET_SECONDS

            ):

                sequence_buffer.clear()


            # A genuine hand release permits the same
            # sign to be spoken again.
            if (

                no_hand_duration
                >=
                NO_HAND_RELEASE_SECONDS

            ):

                latched_word = None


            prediction_buffer.clear()


            last_raw_prediction = ""

            raw_prediction_streak = 0


            hand_return_frames = 0


            no_hand_since = None


        hand_return_frames += 1


        # ----------------------------------------------------
        # ONLY HAND-PRESENT FRAMES ENTER SIGN SEQUENCE
        # ----------------------------------------------------

        features = make_frame_features(

            hand_result,

            pose_result
        )


        sequence_buffer.append(
            features
        )


    # ========================================================
    # NO HAND
    # ========================================================

    else:

        if no_hand_since is None:

            no_hand_since = (
                now
            )


        no_hand_duration = (

            now
            - no_hand_since
        )


        hand_return_frames = 0


        current_prediction = (
            "NO HANDS"
        )


        current_confidence = 0.0

        current_second_confidence = 0.0


        prediction_buffer.clear()


        last_raw_prediction = ""

        raw_prediction_streak = 0


        # Do not keep feeding empty-hand frames
        # into the neural network.
        #
        # Short no-hand gaps therefore retain useful
        # previous gesture history.
        if (

            no_hand_duration
            >=
            NO_HAND_BUFFER_RESET_SECONDS

        ):

            sequence_buffer.clear()


        if (

            no_hand_duration
            >=
            NO_HAND_RELEASE_SECONDS

        ):

            latched_word = None


    # ========================================================
    # ALLOW PREDICTION?
    # ========================================================

    ready_to_predict = (

        hand_present

        and

        len(sequence_buffer)
        == SEQUENCE_LENGTH

        and

        hand_return_frames
        >= HAND_RETURN_WARMUP_FRAMES
    )


    if ready_to_predict:

        (
            prediction,
            confidence,
            second_confidence

        ) = predict(
            sequence_buffer
        )


        current_prediction = (

            prediction
            .replace(
                "_",
                " "
            )
            .upper()
        )


        current_confidence = (
            confidence
        )


        current_second_confidence = (
            second_confidence
        )


        # ====================================================
        # CONSECUTIVE PREDICTION STREAK
        # ====================================================

        if (
            prediction
            == last_raw_prediction
        ):

            raw_prediction_streak += 1


        else:

            last_raw_prediction = (
                prediction
            )

            raw_prediction_streak = 1


        # ====================================================
        # NORMAL VALID PREDICTION
        # ====================================================

        valid_prediction = (

            confidence
            >= CONFIDENCE_THRESHOLD
        )


        if valid_prediction:

            prediction_buffer.append(
                prediction
            )


        else:

            prediction_buffer.append(
                "UNKNOWN"
            )


        # ====================================================
        # FAST MODE
        #
        # 90%+
        # same word twice consecutively
        # ====================================================

        fast_confirm = (

            confidence
            >= FAST_CONFIDENCE

            and

            raw_prediction_streak
            >= FAST_STREAK_REQUIRED
        )


        # ====================================================
        # NORMAL MODE
        #
        # 75%+
        # same answer in at least 2 of last 3
        # ====================================================

        normal_confirm = False


        if (

            len(prediction_buffer)
            == STABILITY_WINDOW

        ):

            counts = Counter(
                prediction_buffer
            )


            best_word, count = (

                counts
                .most_common(1)[0]
            )


            normal_confirm = (

                best_word
                != "UNKNOWN"

                and

                best_word
                == prediction

                and

                count
                >= STABILITY_REQUIRED
            )


        confirmed = (

            fast_confirm
            or
            normal_confirm
        )


        # ====================================================
        # SPEAK
        # ====================================================

        if confirmed:

            # Same held sign should NOT repeatedly talk.
            #
            # But as soon as a different confirmed sign
            # appears, it can speak immediately.
            if (
                prediction
                != latched_word
            ):

                spoken_text = (

                    prediction
                    .replace(
                        "_",
                        " "
                    )
                )


                sentence.append(

                    spoken_text.upper()
                )


                print(

                    "Detected:",

                    spoken_text,

                    f"{confidence * 100:.1f}%"
                )


                speak(
                    spoken_text
                )


                latched_word = (
                    prediction
                )


                prediction_buffer.clear()


    # ========================================================
    # DISPLAY
    # ========================================================

    display = cv2.flip(

        raw_frame,

        1
    )


    # ========================================================
    # BODY / FACE VISUALIZATION
    # ========================================================

    if SHOW_POSE:

        draw_pose_overlay(

            display,

            pose_result
        )


    # ========================================================
    # HAND VISUALIZATION
    # ========================================================

    for hand, information in zip(

        hand_result.hand_landmarks,

        hand_result.handedness
    ):

        hand_label = "right"


        if information:

            hand_label = (

                information[0]
                .category_name
                .lower()
            )


        draw_hand(

            display,

            hand,

            hand_label
        )


    height, width = (
        display.shape[:2]
    )


    # ========================================================
    # BOTTOM UI
    # ========================================================

    PANEL_HEIGHT = 125


    panel_top = (

        height
        - PANEL_HEIGHT
    )


    overlay = (
        display.copy()
    )


    cv2.rectangle(

        overlay,

        (
            0,
            panel_top
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

        0.78,

        display,

        0.22,

        0,

        display
    )


    # ========================================================
    # STATUS
    # ========================================================

    if not hand_present:

        status = (
            "NO HANDS"
        )


    elif (
        len(sequence_buffer)
        < SEQUENCE_LENGTH
    ):

        status = (

            f"WATCHING  "
            f"{len(sequence_buffer)}"
            f"/{SEQUENCE_LENGTH}"
        )


    elif (
        hand_return_frames
        <
        HAND_RETURN_WARMUP_FRAMES
    ):

        status = (

            "REFOCUSING  "

            f"{hand_return_frames}"
            f"/{HAND_RETURN_WARMUP_FRAMES}"
        )


    else:

        status = (

            f"{current_prediction}  "
            f"{current_confidence * 100:.1f}%"
        )


    cv2.putText(

        display,

        status,

        (
            15,
            panel_top + 32
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.70,

        (
            0,
            255,
            0
        )
        if hand_present
        else
        (
            160,
            160,
            160
        ),

        2,

        cv2.LINE_AA
    )


    # ========================================================
    # SENTENCE
    # ========================================================

    sentence_text = " ".join(
        sentence
    )


    if len(
        sentence_text
    ) > 68:

        sentence_text = (

            "..."
            + sentence_text[-65:]
        )


    cv2.putText(

        display,

        sentence_text,

        (
            15,
            panel_top + 67
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.52,

        (
            255,
            255,
            255
        ),

        1,

        cv2.LINE_AA
    )


    # ========================================================
    # CONTROLS
    # ========================================================

    pose_text = (

        "ON"
        if SHOW_POSE
        else "OFF"
    )


    controls_text = (

        "C clear   "
        "R reset   "
        f"P pose:{pose_text}   "
        "F fullscreen   "
        "Q quit"
    )


    cv2.putText(

        display,

        controls_text,

        (
            15,
            panel_top + 103
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.36,

        (
            175,
            175,
            175
        ),

        1,

        cv2.LINE_AA
    )


    # ========================================================
    # FIT WHOLE RENDERED IMAGE TO WINDOW
    # ========================================================

    window_image = (
        fit_to_window(
            display
        )
    )


    cv2.imshow(

        WINDOW_NAME,

        window_image
    )


    # ========================================================
    # KEYS
    # ========================================================

    key = (

        cv2.waitKey(1)
        & 0xFF
    )


    # --------------------------------------------------------
    # WINDOW X BUTTON
    # --------------------------------------------------------

    try:

        if (

            cv2.getWindowProperty(

                WINDOW_NAME,

                cv2.WND_PROP_VISIBLE

            )

            < 1

        ):

            break

    except cv2.error:

        break


    # ========================================================
    # Q / ESC
    # ========================================================

    if (

        key == ord("q")

        or

        key == ord("Q")

        or

        key == 27

    ):

        break


    # ========================================================
    # C = CLEAR
    # ========================================================

    if (

        key == ord("c")

        or

        key == ord("C")

    ):

        sentence.clear()

        latched_word = None


    # ========================================================
    # R = RESET RECOGNITION
    # ========================================================

    if (

        key == ord("r")

        or

        key == ord("R")

    ):

        sequence_buffer.clear()

        prediction_buffer.clear()


        current_prediction = (
            "WAITING"
        )


        current_confidence = 0.0

        current_second_confidence = 0.0


        last_raw_prediction = ""

        raw_prediction_streak = 0


        latched_word = None


        no_hand_since = None


        hand_return_frames = 0


    # ========================================================
    # P = POSE VISUALIZATION
    # ========================================================

    if (

        key == ord("p")

        or

        key == ord("P")

    ):

        SHOW_POSE = (
            not SHOW_POSE
        )


    # ========================================================
    # F = FULLSCREEN
    # ========================================================

    if (

        key == ord("f")

        or

        key == ord("F")

    ):

        fullscreen = (
            not fullscreen
        )


        if fullscreen:

            cv2.setWindowProperty(

                WINDOW_NAME,

                cv2.WND_PROP_FULLSCREEN,

                cv2.WINDOW_FULLSCREEN
            )


        else:

            cv2.setWindowProperty(

                WINDOW_NAME,

                cv2.WND_PROP_FULLSCREEN,

                cv2.WINDOW_NORMAL
            )


            cv2.resizeWindow(

                WINDOW_NAME,

                960,

                720
            )


# ============================================================
# CLEANUP
# ============================================================

camera.release()


cv2.destroyAllWindows()


hand_detector.close()


pose_detector.close()


speech_queue.put(
    None
)


print(
    "Closed."
)