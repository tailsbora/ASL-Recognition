# 🤟 ASL Recognition

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
git clone https://github.com/tailsbora/ASL-Recognition.git
cd ASL-Recognition
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

The **red dots and cyan lines** are only a visualization.

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

### What happens when the hands disappear?

A short period of `NO HANDS` does **not** immediately erase the entire 30-frame sequence.

If the hands return quickly, the recognizer keeps the useful recent sequence and waits for only a few fresh hand frames before continuing.

If the hands remain absent for approximately:

```text
1.2 seconds
```

the old sequence is cleared because it is considered stale.

This improves responsiveness while preventing old movements from affecting future signs.

---

## ⚡ Recognition Logic

The neural network produces a probability for every supported word.

The program adds additional temporal confirmation before a prediction is spoken.

### Fast recognition

A sign can take the fast path when:

```text
Confidence ≥ 90%
```

and the same prediction appears on:

```text
2 consecutive processed frames
```

This allows strong predictions to trigger text-to-speech very quickly.

### Normal recognition

Predictions at:

```text
Confidence ≥ 75%
```

use the normal confirmation system.

The same word must appear in at least:

```text
2 of the last 3 predictions
```

before being accepted.

This prevents one unstable frame from immediately triggering speech.

> Neural-network confidence is not the same as certainty. A high-confidence prediction can still be incorrect.

---

## 🚫 No-Hand Protection

The recognizer will **not speak while no hand is detected**.

When no hands are visible, the program displays:

```text
NO HANDS
```

No prediction is allowed to trigger text-to-speech during this state.

Short hand disappearances are handled without unnecessarily restarting the entire recognition sequence.

This greatly reduces false predictions while the user is not signing.

---

## 🔊 Text-to-Speech

Once a word is confirmed:

1. The recognized word is displayed.
2. It is added to the sentence history.
3. It is printed in the terminal.
4. It is spoken automatically using `pyttsx3`.

The speech engine runs on a separate thread so speaking does not freeze the webcam or recognition loop.

A held sign is also prevented from repeatedly speaking the same word over and over.

When a genuinely different sign is recognized, the new word can be spoken immediately without requiring the user to completely remove their hands first.

---

## ⚡ Low-Latency Camera

Several optimizations are included to reduce camera and recognition delay.

### Latest-frame capture

The webcam runs on a separate thread.

Instead of processing a growing queue of old camera frames:

```text
Frame 1
Frame 2
Frame 3
Frame 4
...
```

the program continuously replaces the previous frame with the **newest available frame**.

This helps keep the camera view close to real time.

### Duplicate-frame protection

The same captured webcam frame is not processed multiple times.

### Smaller MediaPipe processing resolution

MediaPipe analyzes a reduced-resolution copy of the webcam image.

The full camera image is still used for display.

This reduces CPU usage while preserving the appearance of the final video.

### MediaPipe Video Mode

MediaPipe runs in video-tracking mode rather than treating every frame as a completely unrelated image.

This allows it to use tracking information across frames and improves real-time performance.

---

## 🪞 Mirrored Display

The image displayed to the user is mirrored like a normal webcam:

```text
Display → Mirrored
```

However, the image sent into MediaPipe and the neural network is **not mirrored**:

```text
Camera
   │
   ├── Original → MediaPipe → Neural Network
   │
   └── Mirrored → User Display
```

This is important because the WLASL training videos were processed without first mirroring them.

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

on the held-out validation split used during training.

The model correctly classified:

```text
80 / 120 validation sequences
```

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

Some signs are visually similar, and some classes currently have fewer usable training examples than others.

A high confidence value also does not guarantee that the prediction is correct, which is why the real-time recognizer uses temporal confirmation in addition to raw neural-network confidence.

---

## 📚 Training Data

The project uses the **WLASL — Word-Level American Sign Language dataset**.

The current training dataset contains:

```text
70 classes
594 usable video sequences
30 frames per sequence
189 features per frame
```

The raw downloaded videos are **not stored in this GitHub repository**.

Large generated training files are excluded using `.gitignore`.

They are not required to run the included trained model.

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

Downloaded videos are stored under:

```text
selected_videos/
```

### Step 2 — Extract landmarks

```powershell
python extract_wlasl_final.py
```

The extractor:

- Reads the WLASL metadata
- Uses the relevant sign frame range
- Samples exactly 30 frames
- Detects up to two hands
- Detects body and face reference landmarks
- Produces 189 features per frame
- Saves the extracted sequence as NumPy data

Each extracted sequence has the shape:

```text
30 × 189
```

The output is stored under:

```text
wlasl_sequences_final/
```

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

Normalization statistics are calculated using the **training split only** and are saved inside the final model checkpoint.

The model can therefore use the exact same normalization during live inference.

---

## 📁 Project Structure

```text
ASL-Recognition/
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
        │   └── WLASL metadata
        │
        ├── download_more_signs.py
        │   └── Dataset downloader
        │
        ├── extract_wlasl_final.py
        │   └── Landmark / sequence extractor
        │
        └── train_asl_torch.py
            └── BiGRU training script
```

Large downloaded videos and generated NumPy training sequences are intentionally excluded from GitHub.

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

The WLASL training clips also come from different videos and signers than a typical live webcam environment, so real-world webcam performance can differ from validation performance.

---

## 🔮 Future Improvements

Some possible future upgrades:

- Expand beyond 70 words
- Collect more training examples
- Balance weak classes
- Add webcam-recorded training samples
- Add a dedicated `no_sign` / transition class
- Add sign-specific confidence thresholds
- Improve continuous-sign segmentation
- Add fingerspelling recognition
- Add hand velocity and acceleration features
- Improve facial landmark features
- Perform signer-independent evaluation
- Improve temporal augmentation
- Add GPU-accelerated training
- Support full sentence-level recognition

---

## 💭 Feedback & Contributions

Suggestions, bug reports, and improvements are welcome.

If you find a problem or have an idea for a new feature, feel free to open an **Issue** on the repository.

Repository:

[github.com/tailsbora/ASL-Recognition](https://github.com/tailsbora/ASL-Recognition)

---

## 📜 Dataset Notice

This project uses data and metadata from the **WLASL dataset**.

Raw WLASL source videos are not included in this repository.

WLASL and other third-party projects such as MediaPipe, PyTorch, OpenCV, and PyTorch have their own licenses and usage requirements.

Please review the original projects before redistributing their data, models, or source code.

---

## ✍️ Author

Created by [tailsbora](https://github.com/tailsbora).

> *Built to explore how computer vision and temporal neural networks can be used for real-time sign-language recognition.*

---

## ⚡ TL;DR

- 🤟 Real-time **ASL word recognition** from a webcam
- 🧠 Uses a **2-layer Bidirectional GRU + attention**
- 🤲 Tracks **two hands plus body and face landmarks**
- ⏱️ Processes **30-frame × 189-feature** motion sequences
- 📖 Recognizes **70 ASL words**
- 🔊 Automatically speaks confirmed predictions
- 🚫 Does not speak when no hands are detected
- ⚡ Includes low-latency camera and fast high-confidence recognition
- 📊 Current validation accuracy: **66.7%**
- 🛠️ Built with **Python, MediaPipe, OpenCV, NumPy, PyTorch, and pyttsx3**

Run it with:

```powershell
python first.py
```

> **In short:** point the webcam at yourself, perform one of the supported ASL signs, and the model tracks your movement, recognizes the word, displays it, and says it aloud.
