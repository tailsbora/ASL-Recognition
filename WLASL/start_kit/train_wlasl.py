import os
import numpy as np


# ==================================================
# SETTINGS
# ==================================================

DATA_FOLDER = "wlasl_sequences"
MODEL_FILE = "wlasl_model.npz"


# ==================================================
# LOAD DATA
# ==================================================

X = []
y = []


print("Loading sequences...")


for sign in sorted(os.listdir(DATA_FOLDER)):

    sign_folder = os.path.join(
        DATA_FOLDER,
        sign
    )

    if not os.path.isdir(sign_folder):
        continue


    files = sorted(
        f for f in os.listdir(sign_folder)
        if f.endswith(".npy")
    )


    for filename in files:

        path = os.path.join(
            sign_folder,
            filename
        )


        sequence = np.load(path)


        if sequence.shape != (30, 126):

            print(
                "Skipping bad shape:",
                path,
                sequence.shape
            )

            continue


        # Flatten 30 frames × 126 values
        features = sequence.flatten()

        X.append(features)
        y.append(sign)


# ==================================================
# CONVERT TO NUMPY
# ==================================================

X = np.array(
    X,
    dtype=np.float32
)

y = np.array(y)


print()
print("Samples:", len(X))
print("Feature size:", X.shape[1])
print("Signs:")


for sign in sorted(set(y)):

    count = np.sum(y == sign)

    print(
        f"  {sign}: {count}"
    )


# ==================================================
# NORMALIZE FEATURES
# ==================================================

mean = X.mean(axis=0)

std = X.std(axis=0)


# Prevent division by zero
std[std < 0.000001] = 1.0


X_normalized = (
    X - mean
) / std


# ==================================================
# LEAVE-ONE-OUT TEST
# ==================================================

print()
print("Testing model...")


correct = 0


for i in range(len(X_normalized)):

    test_sample = X_normalized[i]


    # Calculate distance to every sample
    distances = np.linalg.norm(
        X_normalized - test_sample,
        axis=1
    )


    # Do not compare sample with itself
    distances[i] = np.inf


    # Closest training example
    nearest_index = np.argmin(
        distances
    )


    prediction = y[
        nearest_index
    ]


    actual = y[i]


    if prediction == actual:
        correct += 1


    print(
        f"{actual:12} -> {prediction:12}"
    )


accuracy = (
    correct / len(X_normalized)
) * 100


print()
print(
    f"Leave-one-out accuracy: {accuracy:.1f}%"
)


# ==================================================
# SAVE MODEL
# ==================================================

np.savez_compressed(
    MODEL_FILE,
    X=X_normalized,
    y=y,
    mean=mean,
    std=std
)


print()
print(
    "MODEL SAVED:",
    MODEL_FILE
)

print("DONE")