import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# SETTINGS
# ============================================================

DATA_FOLDER = "wlasl_sequences_final"
MODEL_FILE = "asl_bigru_model.pt"

SEQUENCE_LENGTH = 30
FEATURE_SIZE = 189

BATCH_SIZE = 16
EPOCHS = 300
LEARNING_RATE = 0.001
PATIENCE = 45

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print()
print("DEVICE:", device)
print()


# ============================================================
# LOAD DATA
# ============================================================

samples = []


for sign in sorted(os.listdir(DATA_FOLDER)):

    sign_folder = os.path.join(
        DATA_FOLDER,
        sign
    )

    if not os.path.isdir(sign_folder):
        continue

    for filename in sorted(os.listdir(sign_folder)):

        if not filename.endswith(".npy"):
            continue

        path = os.path.join(
            sign_folder,
            filename
        )

        sequence = np.load(path).astype(
            np.float32
        )

        if sequence.shape != (
            SEQUENCE_LENGTH,
            FEATURE_SIZE
        ):

            print(
                "Skipping bad shape:",
                filename,
                sequence.shape
            )

            continue

        samples.append(
            (
                sequence,
                sign,
                filename
            )
        )


labels = sorted(
    set(sign for _, sign, _ in samples)
)

label_to_index = {
    label: i
    for i, label in enumerate(labels)
}


print("SIGNS:")

for label in labels:

    count = sum(
        1
        for _, sign, _ in samples
        if sign == label
    )

    print(
        f"  {label}: {count}"
    )


print()
print("TOTAL:", len(samples))


# ============================================================
# STRATIFIED TRAIN / VALIDATION SPLIT
# ============================================================

train_samples = []
val_samples = []


for label in labels:

    class_samples = [
        item
        for item in samples
        if item[1] == label
    ]

    random.shuffle(
        class_samples
    )

    # At least one validation sample per class
    val_count = max(
        1,
        round(
            len(class_samples) * 0.2
        )
    )

    # Keep at least 2 training samples
    val_count = min(
        val_count,
        len(class_samples) - 2
    )

    val_samples.extend(
        class_samples[:val_count]
    )

    train_samples.extend(
        class_samples[val_count:]
    )


random.shuffle(
    train_samples
)

random.shuffle(
    val_samples
)


print()
print(
    "TRAIN:",
    len(train_samples)
)

print(
    "VALIDATION:",
    len(val_samples)
)


# ============================================================
# CALCULATE NORMALIZATION USING TRAINING SET ONLY
# ============================================================

train_array = np.concatenate(
    [
        sequence
        for sequence, _, _
        in train_samples
    ],
    axis=0
)


mean = train_array.mean(
    axis=0
).astype(np.float32)


std = train_array.std(
    axis=0
).astype(np.float32)


std[
    std < 1e-6
] = 1.0


# ============================================================
# DATA AUGMENTATION
# ============================================================

def augment(sequence):

    sequence = sequence.copy()

    # --------------------------------------------------------
    # SMALL LANDMARK NOISE
    # --------------------------------------------------------

    if random.random() < 0.75:

        nonzero = (
            np.abs(sequence) > 1e-8
        )

        noise = np.random.normal(
            0,
            0.015,
            sequence.shape
        ).astype(np.float32)

        sequence = (
            sequence
            + noise * nonzero
        )


    # --------------------------------------------------------
    # RANDOM TEMPORAL SHIFT
    # --------------------------------------------------------

    if random.random() < 0.5:

        shift = random.randint(
            -2,
            2
        )

        if shift != 0:

            sequence = np.roll(
                sequence,
                shift,
                axis=0
            )


    # --------------------------------------------------------
    # RANDOM FRAME REPEAT
    # Simulates slightly slower/faster signing
    # --------------------------------------------------------

    if random.random() < 0.35:

        index = random.randint(
            1,
            SEQUENCE_LENGTH - 2
        )

        sequence[index] = (
            sequence[index - 1]
            + sequence[index + 1]
        ) / 2


    # --------------------------------------------------------
    # SMALL FEATURE DROPOUT
    # --------------------------------------------------------

    if random.random() < 0.3:

        mask = (
            np.random.random(
                sequence.shape
            ) > 0.02
        )

        sequence = (
            sequence
            * mask.astype(np.float32)
        )


    return sequence


# ============================================================
# DATASET
# ============================================================

class ASLDataset(Dataset):

    def __init__(
        self,
        data,
        training=False
    ):

        self.data = data
        self.training = training


    def __len__(self):

        return len(
            self.data
        )


    def __getitem__(
        self,
        index
    ):

        sequence, label, _ = (
            self.data[index]
        )

        sequence = (
            sequence.copy()
        )


        if self.training:

            sequence = augment(
                sequence
            )


        sequence = (
            sequence - mean
        ) / std


        target = (
            label_to_index[
                label
            ]
        )


        return (
            torch.tensor(
                sequence,
                dtype=torch.float32
            ),
            torch.tensor(
                target,
                dtype=torch.long
            )
        )


train_dataset = ASLDataset(
    train_samples,
    training=True
)

val_dataset = ASLDataset(
    val_samples,
    training=False
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_counts = []


for label in labels:

    count = sum(
        1
        for _, sign, _
        in train_samples
        if sign == label
    )

    class_counts.append(
        count
    )


class_counts = np.array(
    class_counts,
    dtype=np.float32
)


weights = (
    len(train_samples)
    /
    (
        len(labels)
        * class_counts
    )
)


class_weights = torch.tensor(
    weights,
    dtype=torch.float32,
    device=device
)


# ============================================================
# TEMPORAL ATTENTION MODEL
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


        logits = self.classifier(
            context
        )


        return logits


# ============================================================
# MODEL
# ============================================================

model = SignLanguageModel(
    input_size=FEATURE_SIZE,
    num_classes=len(labels)
).to(device)


criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.05
)


optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=0.0005
)


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="max",

    factor=0.5,

    patience=12
)


# ============================================================
# EVALUATION
# ============================================================

def evaluate():

    model.eval()

    correct = 0
    total = 0
    total_loss = 0.0


    with torch.no_grad():

        for x, y in val_loader:

            x = x.to(
                device
            )

            y = y.to(
                device
            )


            logits = model(
                x
            )


            loss = criterion(
                logits,
                y
            )


            total_loss += (
                loss.item()
                * len(y)
            )


            prediction = torch.argmax(
                logits,
                dim=1
            )


            correct += (
                prediction == y
            ).sum().item()


            total += len(y)


    accuracy = (
        correct / total
        if total > 0
        else 0
    )


    loss = (
        total_loss / total
        if total > 0
        else 0
    )


    return loss, accuracy


# ============================================================
# TRAIN
# ============================================================

best_accuracy = 0.0
best_state = None
epochs_without_improvement = 0


print()
print("=" * 60)
print("TRAINING")
print("=" * 60)


for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()


    running_loss = 0.0
    correct = 0
    total = 0


    for x, y in train_loader:

        x = x.to(
            device
        )

        y = y.to(
            device
        )


        optimizer.zero_grad()


        logits = model(
            x
        )


        loss = criterion(
            logits,
            y
        )


        loss.backward()


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )


        optimizer.step()


        running_loss += (
            loss.item()
            * len(y)
        )


        prediction = torch.argmax(
            logits,
            dim=1
        )


        correct += (
            prediction == y
        ).sum().item()


        total += len(y)


    train_loss = (
        running_loss
        / total
    )

    train_accuracy = (
        correct
        / total
    )


    val_loss, val_accuracy = (
        evaluate()
    )


    scheduler.step(
        val_accuracy
    )


    if (
        epoch == 1
        or epoch % 5 == 0
    ):

        print(
            f"Epoch {epoch:03d} | "
            f"Train {train_accuracy * 100:5.1f}% | "
            f"Val {val_accuracy * 100:5.1f}% | "
            f"Loss {val_loss:.4f}"
        )


    if val_accuracy > best_accuracy:

        best_accuracy = (
            val_accuracy
        )

        best_state = copy.deepcopy(
            model.state_dict()
        )

        epochs_without_improvement = 0


    else:

        epochs_without_improvement += 1


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
# RESTORE BEST MODEL
# ============================================================

if best_state is not None:

    model.load_state_dict(
        best_state
    )


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

print()
print("=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)


model.eval()


correct = 0


with torch.no_grad():

    for sequence, actual, filename in val_samples:

        normalized = (
            sequence - mean
        ) / std


        x = torch.tensor(
            normalized,
            dtype=torch.float32
        )

        x = x.unsqueeze(
            0
        ).to(
            device
        )


        logits = model(
            x
        )


        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]


        confidence, prediction_index = (
            torch.max(
                probabilities,
                dim=0
            )
        )


        predicted = labels[
            prediction_index.item()
        ]


        if predicted == actual:

            correct += 1


        print(
            f"{actual:12} -> "
            f"{predicted:12} "
            f"{confidence.item() * 100:5.1f}%"
        )


final_accuracy = (
    correct
    / len(val_samples)
)


print()
print(
    f"BEST VALIDATION ACCURACY: "
    f"{best_accuracy * 100:.1f}%"
)

print(
    f"FINAL VALIDATION ACCURACY: "
    f"{final_accuracy * 100:.1f}%"
)


# ============================================================
# SAVE MODEL
# ============================================================

checkpoint = {

    "model_state": model.state_dict(),

    "labels": labels,

    "mean": mean,

    "std": std,

    "feature_size": FEATURE_SIZE,

    "sequence_length": SEQUENCE_LENGTH
}


torch.save(
    checkpoint,
    MODEL_FILE
)


print()
print(
    "MODEL SAVED:",
    MODEL_FILE
)

print("DONE")