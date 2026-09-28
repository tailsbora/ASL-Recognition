import os
import numpy as np


DATA_FOLDER = "wlasl_sequences_max"
MODEL_FILE = "wlasl_model_max.npz"


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

    for filename in sorted(os.listdir(sign_folder)):

        if not filename.endswith(".npy"):
            continue

        path = os.path.join(
            sign_folder,
            filename
        )

        sequence = np.load(path)

        if sequence.shape != (30, 156):

            print(
                "Skipping bad shape:",
                path,
                sequence.shape
            )

            continue

        X.append(
            sequence.flatten()
        )

        y.append(sign)


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

    print(
        f"  {sign}: {np.sum(y == sign)}"
    )


# ==================================================
# STANDARDIZE
# ==================================================

mean = X.mean(axis=0)

std = X.std(axis=0)

std[std < 0.000001] = 1.0


Xn = (
    X - mean
) / std


# ==================================================
# LEAVE-ONE-OUT NEAREST NEIGHBOR
# ==================================================

print()
print("Testing...")


correct = 0


for i in range(len(Xn)):

    test = Xn[i]

    distances = np.linalg.norm(
        Xn - test,
        axis=1
    )

    distances[i] = np.inf

    nearest = np.argmin(
        distances
    )

    prediction = y[
        nearest
    ]

    actual = y[i]

    if prediction == actual:
        correct += 1

    print(
        f"{actual:12} -> {prediction:12}"
    )


accuracy = (
    correct / len(Xn)
) * 100


print()
print(
    f"Leave-one-out accuracy: {accuracy:.1f}%"
)


# ==================================================
# SAVE
# ==================================================

np.savez_compressed(
    MODEL_FILE,
    X=Xn,
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