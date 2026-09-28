# ASL Recognition

> *Real-time American Sign Language word recognition using computer vision and deep learning.*

A webcam-based ASL recognition system built with **Python, MediaPipe, OpenCV, and PyTorch**.

The program tracks both hands and upper-body landmarks, analyzes movement across multiple frames using a **Bidirectional GRU neural network**, predicts one of **70 supported ASL words**, and automatically speaks the recognized word using text-to-speech.

---

## 🌟 Highlights

- 🎥 Real-time webcam recognition
- 🤲 Tracks up to **two hands**
- 🧍 Uses hand, face, and upper-body landmarks
- 🧠 **2-layer Bidirectional GRU** neural network
- ⏱️ Uses **30-frame temporal sequences**
- 🔊 Automatic text-to-speech
- 💬 Builds a live sentence from recognized words
- ⚡ Low-latency camera capture
- 🚫 Prevents speech when no hands are detected
- 🖥️ Fullscreen and maximized-window support
- 👁️ Toggleable landmark visualization
- 📚 Trained using the **WLASL word-level ASL dataset**
- 🧩 Currently supports **70 ASL words**

---

## ℹ️ Overview

Most simple gesture-recognition systems classify a single image.

This project instead analyzes how the hands and body **move over time**.

Each webcam frame is converted into numerical landmark coordinates using MediaPipe:

```text
Left Hand
      \
       \
Right Hand ──► 189 features per frame
       /
      /
Body + Face
```

The neural network then processes:

```text
30 frames × 189 features
```

to recognize the complete motion of a sign.

This allows the system to distinguish signs that may have similar hand shapes but different movements.

---

## 🚀 Quick Start

### 1. Clone the repository

```powershell
git clone https://github.com/tailsbora/Asl-Recognition.git
cd Asl-Recognition
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install the required libraries

```powershell
pip install opencv-python mediapipe numpy torch pyttsx3
```

If you also want to download additional WLASL training videos:

```powershell
pip install yt-dlp
```

### 4. Run the recognizer

```powershell
python first.py
```

> If `first.py` has been renamed, run the new filename instead.

---

## 📦 Requirements

The project is currently designed and tested primarily on **Windows**.

You will need:

| Requirement | Purpose |
|---|---|
| Python 3 | Main programming language |
| OpenCV | Camera capture and display |
| MediaPipe | Hand and pose landmark detection |
| NumPy | Numerical processing |
| PyTorch | Neural-network inference and training |
| pyttsx3 | Text-to-speech |
| Webcam | Live ASL input |
| yt-dlp | Optional WLASL video downloading |

---

## 🎮 Controls

| Key | Action |
|---|---|
| `P` | Show / hide body and face landmarks |
| `F` | Toggle fullscreen |
| `C` | Clear recognized sentence |
| `R` | Reset recognition |
| `Q` | Quit |
| `Esc` | Quit |

The normal application window can also be maximized while preserving the camera's proportions.

---

## 🧠 How It Works

The recognition pipeline is:

```text
Webcam
   │
   ▼
MediaPipe
   │
   ├── Left hand landmarks
   ├── Right hand landmarks
   └── Body / face landmarks
   │
   ▼
189 features per frame
   │
   ▼
30-frame sequence
   │
   ▼
Linear Projection
189 → 192
   │
   ▼
2-Layer Bidirectional GRU
   │
   ▼
Attention Layer
   │
   ▼
Dense Layer
   │
   ▼
70 ASL Classes
   │
   ├── On-screen prediction
   └── Text-to-speech
```

---

## ✋ Hand Tracking

MediaPipe detects **21 landmarks per hand**.

For each hand:

```text
21 landmarks × 3 coordinates = 63
Absolute wrist position       =  3
----------------------------------
Total                         = 66
```

Two hands:

```text
66 × 2 = 132 features
```

Hand coordinates are normalized relative to the wrist to make recognition less dependent on hand size and position.

---

## 🧍 Body & Face Tracking

The model also uses **19 pose landmarks** from areas such as:

- Face
- Shoulders
- Elbows
- Wrists
- Hips

Each landmark contains:

```text
X
Y
Z
```

Therefore:

```text
19 × 3 = 57 pose features
```

Combined with the hand data:

```text
Hands = 132
Pose  =  57
------------
Total = 189 features per frame
```

Press `P` while the program is running to show or hide these landmarks.

The red dots and cyan lines are only a visualization.

The neural network continues using the landmarks even when the visualization is hidden.

---

## 🧠 Neural Network

The model is implemented using **PyTorch**.

```text
Input
30 × 189
   │
   ▼
Linear Projection
189 → 192
   │
   ├── LayerNorm
   ├── ReLU
   └── Dropout
   │
   ▼
Bidirectional GRU
Layer 1
   │
   ▼
Bidirectional GRU
Layer 2
   │
   ▼
256 features per frame
   │
   ▼
Attention
   │
   ▼
256-feature context vector
   │
   ▼
Dense Layer
256 → 128
   │
   ▼
Output
128 → 70 classes
```

The GRU uses a hidden size of **128 in each direction**.

Because it is bidirectional, the model can learn information from the full movement sequence rather than treating each frame independently.

---

## ⏱️ Why 30 Frames?

ASL is based heavily on movement.

A single image is often not enough to distinguish two signs.

The model therefore analyzes:

```text
30 consecutive frames
```

After the first 30 frames have been collected, the recognizer uses a sliding window:

```text
Frames 1–30
Frames 2–31
Frames 3–32
Frames 4–33
...
```

This allows continuous real-time predictions.

Short moments where the hands disappear do **not immediately erase the entire sequence**, which helps reduce unnecessary recognition delay.

Longer periods without hands reset the sequence to prevent old movements from affecting future predictions.

---

## ⚡ Recognition Logic

The neural network produces a confidence value for every possible word.

The program adds extra rules before speaking a prediction.

### Fast recognition

A word can be accepted quickly when:

```text
Confidence ≥ 90%
```

and the same prediction appears on multiple consecutive processed frames.

### Normal recognition

Predictions above approximately:

```text
75%
```

must remain consistent across several recent predictions.

This helps prevent random single-frame predictions from immediately triggering speech.

> Neural-network confidence is not the same as certainty. A high-confidence prediction can still be incorrect.

---

## 🚫 No-Hand Protection

The recognizer will **not speak while no hand is detected**.

When no hands are visible, the program displays:

```text
NO HANDS
```

This greatly reduces false predictions while the user is standing in front of the camera without signing.

---

## 🔊 Text-to-Speech

Once a word is confirmed:

1. The recognized word is displayed.
2. It is added to the sentence history.
3. It is printed in the terminal.
4. It is spoken automatically using `pyttsx3`.

The speech engine runs in a separate thread so it does not freeze the camera.

---

## ⚡ Low-Latency Camera

Several optimizations are included to reduce camera delay.

### Latest-frame capture

The webcam runs on a separate thread.

Instead of processing a queue of old frames:

```text
Frame 1
Frame 2
Frame 3
Frame 4
...
```

the program constantly replaces them with the **newest available frame**.

This prevents the display from slowly falling behind real time.

### Duplicate-frame protection

The same camera frame is never processed more than once.

### Smaller processing resolution

MediaPipe analyzes a smaller copy of the camera image while the full camera image is still used for display.

This reduces CPU usage and improves responsiveness.

---

## 🪞 Mirrored Display

The image displayed to the user is mirrored like a normal webcam:

```text
Display → Mirrored
```

However, the neural network receives the original unmirrored image:

```text
Camera
   │
   ├── Original → MediaPipe → Neural Network
   │
   └── Mirrored → User Display
```

This is important because the WLASL training videos were processed without mirroring.

Mirroring the neural-network input could reverse left/right hand information and reduce recognition accuracy.

---

## 📖 Supported Words

The current model recognizes **70 ASL words**:

```text
again       angry       bad         bathroom      big
brother     car         child       cold          come
computer    drink       eat         family        father
food        friend      go          good          happy
hello       help        here        home          hot
how         know        like        love          man
me          money       more        morning       mother
name        need        night       no            phone
please      right       sad         school        sister
small       sorry       stop        thank you     there
think       time        tired       today         tomorrow
understand  wait        want        water         what
when        where       who         why           woman
work        wrong       yes         yesterday     you
```

---

## 📊 Current Performance

The current 70-word neural network achieved:

```text
Validation Accuracy: 66.7%
```

on the validation split used during training.

Some examples of strong correct validation predictions were:

| Word | Confidence |
|---|---:|
| Please | 98.4% |
| When | 98.3% |
| Today | 98.2% |
| Man | 98.1% |
| Name | 98.0% |
| Thank You | 97.9% |

Performance varies significantly between signs.

Some words are visually similar, and some classes currently have fewer training examples than others.

---

## 📚 Training Data

The project uses the **WLASL — Word-Level American Sign Language dataset**.

The current dataset used for this model contains:

```text
70 classes
594 usable video sequences
30 frames per sequence
189 features per frame
```

The raw downloaded videos are not stored in this GitHub repository.

Large generated training files are excluded using `.gitignore`.

---

## 🏋️ Training the Model

Training scripts are located in:

```text
WLASL/start_kit/
```

The main files are:

| File | Purpose |
|---|---|
| `download_more_signs.py` | Downloads selected WLASL videos |
| `extract_wlasl_final.py` | Extracts hand and pose features |
| `train_asl_torch.py` | Trains the BiGRU neural network |
| `WLASL_v0.3.json` | WLASL metadata |
| `asl_bigru_model.pt` | Trained neural network |

### Step 1 — Download videos

```powershell
cd WLASL\start_kit
python download_more_signs.py
```

### Step 2 — Extract landmarks

```powershell
python extract_wlasl_final.py
```

The extractor converts every usable video into:

```text
30 × 189
```

NumPy sequences.

### Step 3 — Train

```powershell
python train_asl_torch.py
```

The best model is saved as:

```text
asl_bigru_model.pt
```

---

## 🧪 Training Techniques

The training pipeline includes:

- AdamW optimizer
- Cross-entropy loss
- Label smoothing
- Class weighting
- Gradient clipping
- Early stopping
- Learning-rate reduction
- Landmark noise augmentation
- Temporal shifting
- Frame replacement
- Random feature dropout

Normalization statistics are calculated using the training data and stored inside the final model file.

---

## 📁 Project Structure

```text
Asl-Recognition/
│
├── first.py
│   └── Main real-time recognizer
│
├── hand_landmarker.task
│   └── MediaPipe hand model
│
├── pose_landmarker.task
│   └── MediaPipe pose model
│
├── .gitignore
│
└── WLASL/
    │
    └── start_kit/
        │
        ├── asl_bigru_model.pt
        │   └── Trained 70-word model
        │
        ├── WLASL_v0.3.json
        │
        ├── download_more_signs.py
        │
        ├── extract_wlasl_final.py
        │
        └── train_asl_torch.py
```

---

## ⚠️ Limitations

> *This is currently a word-level recognition project, not a complete ASL translation system.*

The system currently recognizes only its **70 trained words**.

It does not yet support:

- Arbitrary ASL vocabulary
- Full ASL grammar
- Complete fingerspelling
- Continuous unrestricted signing
- Full sentence-level translation
- All facial grammar used in ASL

Recognition can also be affected by:

- Lighting
- Camera angle
- Distance from the camera
- Hand visibility
- Signing speed
- Individual signing style
- Similar-looking signs

---

## 🔮 Future Improvements

Some possible future upgrades:

- Expand beyond 70 words
- Collect more training examples
- Balance weak classes
- Add webcam-recorded training samples
- Add a dedicated `no_sign` class
- Add sign-specific confidence thresholds
- Improve continuous-sign segmentation
- Add fingerspelling
- Add hand velocity and acceleration features
- Improve facial landmark features
- Perform signer-independent evaluation
- Add GPU-accelerated training
- Support full sentence-level recognition

---

## 💭 Feedback & Contributions

Suggestions, bug reports, and improvements are welcome.

If you find a problem or have an idea for a new feature, feel free to open an **Issue** on the repository.

Repository:

[github.com/tailsbora/Asl-Recognition](https://github.com/tailsbora/Asl-Recognition)

---

## 📜 Dataset Notice

This project uses data and metadata from the **WLASL dataset**.

Raw WLASL videos are not included in this repository.

WLASL and other third-party projects such as MediaPipe, PyTorch, and OpenCV have their own licenses and usage requirements.

Please review the original projects before redistributing their data, models, or source code.

---

## ✍️ Author

Created by [tailsbora](https://github.com/tailsbora).

> *Built to explore how computer vision and temporal neural networks can be used for real-time sign-language recognition.*
