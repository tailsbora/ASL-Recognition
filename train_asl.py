import os
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


DATA_FOLDER = "asl_data"

X = []
y = []


print("Loading ASL data...")


for folder_name in os.listdir(DATA_FOLDER):

    folder_path = os.path.join(
        DATA_FOLDER,
        folder_name
    )

    if not os.path.isdir(folder_path):
        continue


    # Supports either:
    # A/
    # or A_left/ A_right/

    label = folder_name.split("_")[0].upper()


    files = [
        f for f in os.listdir(folder_path)
        if f.endswith(".npy")
    ]


    print(
        f"{folder_name}: {len(files)} samples -> {label}"
    )


    for filename in files:

        path = os.path.join(
            folder_path,
            filename
        )

        sample = np.load(path)

        X.append(sample)
        y.append(label)


X = np.array(
    X,
    dtype=np.float32
)

y = np.array(y)


print()
print("Total samples:", len(X))
print("Letters:", sorted(set(y)))


if len(X) == 0:
    print("No training data found.")
    exit()


# --------------------------------------------------
# MODEL
# --------------------------------------------------

model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "classifier",
        KNeighborsClassifier(
            n_neighbors=5,
            weights="distance"
        )
    )
])


print("Training...")

model.fit(
    X,
    y
)


# --------------------------------------------------
# SAVE
# --------------------------------------------------

with open("asl_model.pkl", "wb") as f:
    pickle.dump(model, f)


print()
print("DONE")
print("Saved as asl_model.pkl")