import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# ── Paths ──────────────────────────────────────────────────────────────────────
v1_file    = r"C:\mqttdataset\BCCC-IoT-MQTT-IDS-2025\New folder\strat\Dataset_138_Balanced.csv"
output_dir = r"C:\mqttdataset\BCCC-IoT-MQTT-IDS-2025\New folder\strat"

os.makedirs(output_dir, exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading v1 balanced dataset...")
df = pd.read_csv(v1_file, low_memory=False)
print(f"Loaded shape: {df.shape}")

# ── Detect label column ────────────────────────────────────────────────────────
label_col = next(
    (c for c in df.columns if c.lower() in ['label', 'attack', 'class', 'category']),
    None
)
if label_col is None:
    raise ValueError("Could not detect label column.")
print(f"Label column: '{label_col}'")
print("\nClass distribution:")
print(df[label_col].value_counts())

X = df.drop(columns=[label_col])
y = df[label_col]

# ══════════════════════════════════════════════════════════════════════════════
# SPLIT 1 — 60:40
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*50)
print("Split 1: 60:40")
print("="*50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.40, random_state=42, stratify=y
)

train_df = X_train.copy()
train_df[label_col] = y_train.values
test_df  = X_test.copy()
test_df[label_col]  = y_test.values

train_path = os.path.join(output_dir, "Train_Balanced_60.csv")
test_path  = os.path.join(output_dir, "Test_Balanced_40.csv")

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path,   index=False)

print(f"Train shape : {train_df.shape}  → Train_Balanced_60.csv")
print(f"Test shape  : {test_df.shape}   → Test_Balanced_40.csv")
print("Train class distribution:")
print(train_df[label_col].value_counts())
print("Test class distribution:")
print(test_df[label_col].value_counts())

# ══════════════════════════════════════════════════════════════════════════════
# SPLIT 2 — 70:30
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*50)
print("Split 2: 70:30")
print("="*50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

train_df = X_train.copy()
train_df[label_col] = y_train.values
test_df  = X_test.copy()
test_df[label_col]  = y_test.values

train_path = os.path.join(output_dir, "Train_Balanced_70.csv")
test_path  = os.path.join(output_dir, "Test_Balanced_30.csv")

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path,   index=False)

print(f"Train shape : {train_df.shape}  → Train_Balanced_70.csv")
print(f"Test shape  : {test_df.shape}   → Test_Balanced_30.csv")
print("Train class distribution:")
print(train_df[label_col].value_counts())
print("Test class distribution:")
print(test_df[label_col].value_counts())

# ══════════════════════════════════════════════════════════════════════════════
# SPLIT 3 — 80:20
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*50)
print("Split 3: 80:20")
print("="*50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

train_df = X_train.copy()
train_df[label_col] = y_train.values
test_df  = X_test.copy()
test_df[label_col]  = y_test.values

train_path = os.path.join(output_dir, "Train_Balanced_80.csv")
test_path  = os.path.join(output_dir, "Test_Balanced_20.csv")

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path,   index=False)

print(f"Train shape : {train_df.shape}  → Train_Balanced_80.csv")
print(f"Test shape  : {test_df.shape}   → Test_Balanced_20.csv")
print("Train class distribution:")
print(train_df[label_col].value_counts())
print("Test class distribution:")
print(test_df[label_col].value_counts())

print("\n========== v1 ALL SPLITS COMPLETE ==========")
print("Files generated:")
print("  Train_Balanced_60.csv  /  Test_Balanced_40.csv")
print("  Train_Balanced_70.csv  /  Test_Balanced_30.csv")
print("  Train_Balanced_80.csv  /  Test_Balanced_20.csv")
