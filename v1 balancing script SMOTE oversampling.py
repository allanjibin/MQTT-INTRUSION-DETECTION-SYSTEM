import pandas as pd
import numpy as np
import os
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import joblib

# ── Paths ──────────────────────────────────────────────────────────────────────
input_file  = r"C:\mqttdataset\BCCC-IoT-MQTT-IDS-2025\New folder\strat\Dataset_138_Features.csv"
output_dir  = r"C:\mqttdataset\BCCC-IoT-MQTT-IDS-2025\New folder\strat"
output_file = os.path.join(output_dir, "Dataset_138_Balanced.csv")

os.makedirs(output_dir, exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading 138-feature dataset...")
df = pd.read_csv(input_file, low_memory=False)
print(f"Shape: {df.shape}")

# ── Detect label column ────────────────────────────────────────────────────────
label_col = next(
    (c for c in df.columns if c.lower() in ['label', 'attack', 'class', 'category']),
    None
)
if label_col is None:
    raise ValueError("Could not auto-detect label column.")
print(f"\nLabel column: '{label_col}'")
print("Class distribution (before balancing):")
print(df[label_col].value_counts())

# ── Separate features and label ────────────────────────────────────────────────
X = df.drop(columns=[label_col])
y = df[label_col]

# ── Convert to numeric ─────────────────────────────────────────────────────────
print("\nConverting features to numeric...")
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
print("Done.")

# ── Encode labels ──────────────────────────────────────────────────────────────
le = LabelEncoder()
y_enc = le.fit_transform(y)
print(f"\nClasses: {list(le.classes_)}")

# ── SMOTE oversampling (v1) ────────────────────────────────────────────────────
print("\nApplying SMOTE oversampling (v1)...")
print("This may take several minutes on large datasets...")

# k_neighbors set to min(5, smallest_class_size - 1) to avoid errors
min_class_size = df[label_col].value_counts().min()
k_neighbors = min(5, min_class_size - 1)
print(f"Using k_neighbors={k_neighbors}")

smote = SMOTE(
    sampling_strategy='not majority',  # oversample all minority classes
    k_neighbors=k_neighbors,
    random_state=42,
    n_jobs=-1
)

X_balanced, y_balanced = smote.fit_resample(X, y_enc)

print(f"\nShape after SMOTE: {X_balanced.shape}")

# ── Decode labels back to original strings ─────────────────────────────────────
y_decoded = le.inverse_transform(y_balanced)

# ── Build balanced dataframe ───────────────────────────────────────────────────
df_balanced = pd.DataFrame(X_balanced, columns=X.columns)
df_balanced[label_col] = y_decoded

print("\nClass distribution (after SMOTE):")
print(df_balanced[label_col].value_counts())
print(f"\nTotal rows: {df_balanced.shape[0]:,}")

# ── Save ───────────────────────────────────────────────────────────────────────
print(f"\nSaving to {output_file} ...")
df_balanced.to_csv(output_file, index=False)
print("Done.")

# ── Save label encoder ────────────────────────────────────────────────────────
joblib.dump(le, os.path.join(output_dir, "label_encoder_v1.pkl"))
print("Label encoder saved.")

print("\n========== SUMMARY ==========")
print(f"Original rows  : {df.shape[0]:,}")
print(f"Balanced rows  : {df_balanced.shape[0]:,}")
print(f"Features       : {X.shape[1]}")
print(f"Output file    : {output_file}")
