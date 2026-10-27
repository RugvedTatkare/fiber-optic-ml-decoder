
from pathlib import Path
import json
import shutil

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "ml" / "models"
DATA_FILE = ROOT / "final_data_enclosed" / "final_dataset_20k.csv"
LIVE_FILE = ROOT / "live_training_data" / "bent_live.csv"
OUTPUT_DIR = MODEL_DIR / "live_experiment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 531
SEED = 42
tf.keras.utils.set_random_seed(SEED)

# Load the original dataset and the new live dataset.
old = pd.read_csv(DATA_FILE)
live = pd.read_csv(LIVE_FILE)

required = {"trial", "original_bit", "sensor_reading"}
for name, df in [("original", old), ("live", live)]:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} dataset missing columns: {missing}")

# Original data: use trials 1-7 for training and trial 8 for validation.
# Live data: trials 1-7 for training, 8 for validation, 9-10 for testing.
old_fit = old[old["trial"].between(1, 7)].copy()
old_val = old[old["trial"] == 8].copy()
live_fit = live[live["trial"].between(1, 7)].copy()
live_val = live[live["trial"] == 8].copy()
live_test = live[live["trial"].isin([9, 10])].copy()

if any(part.empty for part in [
    old_fit, old_val, live_fit, live_val, live_test
]):
    raise ValueError("A training, validation, or test split is empty.")

def xy(df):
    X = df[["sensor_reading"]].to_numpy(dtype=np.float32)
    y = df["original_bit"].to_numpy(dtype=int)
    return X, y

X_old_fit, y_old_fit = xy(old_fit)
X_old_val, y_old_val = xy(old_val)
X_live_fit, y_live_fit = xy(live_fit)
X_live_val, y_live_val = xy(live_val)
X_test, y_test = xy(live_test)

# Combine original training samples with live bent-fiber training samples.
X_train = np.vstack([X_old_fit, X_live_fit])
y_train = np.concatenate([y_old_fit, y_live_fit])

# Fit the candidate scaler on training data only.
candidate_scaler = StandardScaler()
X_train_scaled = candidate_scaler.fit_transform(X_train)
X_val_scaled = candidate_scaler.transform(
    np.vstack([X_old_val, X_live_val])
)
y_val = np.concatenate([y_old_val, y_live_val])
X_test_scaled = candidate_scaler.transform(X_test)

# Create the candidate ANN using the existing architecture.
candidate = Sequential([
    Input(shape=(1,)),
    Dense(16, activation="relu"),
    Dropout(0.1),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid"),
])

candidate.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True,
)

history = candidate.fit(
    X_train_scaled,
    y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=300,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1,
)

# Load the existing production model and its original scaler.
existing = load_model(MODEL_DIR / "bit_decoder.keras")
existing_scaler = joblib.load(MODEL_DIR / "bit_scaler.joblib")

def evaluate(name, model, scaler, X, y):
    probabilities = model.predict(scaler.transform(X), verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype(int)
    errors = int(np.sum(predictions != y))
    accuracy = accuracy_score(y, predictions) * 100

    print(f"\n{name}")
    print(f"Accuracy: {accuracy:.3f}%")
    print(f"Errors: {errors}/{len(y)}")
    print("Confusion matrix [true rows, predicted columns]:")
    print(confusion_matrix(y, predictions, labels=[0, 1]))
    return {
        "accuracy_percent": float(accuracy),
        "errors": errors,
        "samples": int(len(y)),
    }

print("\n=== HELD-OUT LIVE BENT-FIBER TRIALS 9-10 ===")
results = {}

results["existing_ann"] = evaluate(
    "Existing ANN", existing, existing_scaler, X_test, y_test
)
results["candidate_ann"] = evaluate(
    "Candidate ANN", candidate, candidate_scaler, X_test, y_test
)

threshold_predictions = (X_test.reshape(-1) > THRESHOLD).astype(int)
threshold_errors = int(np.sum(threshold_predictions != y_test))
threshold_accuracy = accuracy_score(y_test, threshold_predictions) * 100

print("\nThreshold baseline (531 ADC)")
print(f"Accuracy: {threshold_accuracy:.3f}%")
print(f"Errors: {threshold_errors}/{len(y_test)}")
print(confusion_matrix(y_test, threshold_predictions, labels=[0, 1]))

results["threshold_531"] = {
    "accuracy_percent": float(threshold_accuracy),
    "errors": threshold_errors,
    "samples": int(len(y_test)),
}

# Save candidate separately; do not overwrite the production model.
candidate.save(OUTPUT_DIR / "candidate_decoder.keras")
joblib.dump(candidate_scaler, OUTPUT_DIR / "candidate_scaler.joblib")

with (OUTPUT_DIR / "evaluation.json").open("w") as f:
    json.dump(results, f, indent=2)

print("\nCandidate saved to:", OUTPUT_DIR)
print("Production model was not changed.")
