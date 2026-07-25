import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import joblib

# ── Paths ──────────────────────────────────────────────────────────────────────
input_file  = r"C:\mqttdataset\BCCC-IoT-MQTT-IDS-2025\New folder\strat\Dataset_138_Features.csv"
output_dir  = r"C:\mqttdataset\BCCC-IoT-MQTT-IDS-2025\New folder\strat"
output_file = os.path.join(output_dir, "Dataset_138_Balanced_v2.csv")

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

# ── Downsampling strategy (v2) ─────────────────────────────────────────────────
# Target: downsample dominant class(es) to a reasonable cap
# Keep all minority classes as-is
# Cap = second largest class size (so we don't lose too much minority data)

class_counts = df[label_col].value_counts()
print(f"\nLargest class  : {class_counts.index[0]} ({class_counts.iloc[0]:,} rows)")
print(f"2nd largest    : {class_counts.index[1]} ({class_counts.iloc[1]:,} rows)")

# Downsample cap = size of second largest class
downsample_cap = int(class_counts.iloc[1])
print(f"\nDownsampling cap set to: {downsample_cap:,} rows per class")

# ── Apply downsampling ─────────────────────────────────────────────────────────
print("\nApplying dominant-class downsampling (v2)...")

balanced_dfs = []
for class_name, group in df.groupby(label_col):
    if len(group) > downsample_cap:
        sampled = group.sample(n=downsample_cap, random_state=42)
        print(f"  {class_name:<45} {len(group):>10,} → {len(sampled):>10,} (downsampled)")
    else:
        sampled = group
        print(f"  {class_name:<45} {len(group):>10,} → {len(sampled):>10,} (kept as-is)")
    balanced_dfs.append(sampled)

df_balanced = pd.concat(balanced_dfs, ignore_index=True)

# ── Shuffle ────────────────────────────────────────────────────────────────────
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nClass distribution (after downsampling):")
print(df_balanced[label_col].value_counts())
print(f"\nTotal rows: {df_balanced.shape[0]:,}")

# ── Save label encoder ────────────────────────────────────────────────────────
le = LabelEncoder()
le.fit(df_balanced[label_col])
joblib.dump(le, os.path.join(output_dir, "label_encoder_v2.pkl"))
print("Label encoder saved.")

# ── Save ───────────────────────────────────────────────────────────────────────
print(f"\nSaving to {output_file} ...")
df_balanced.to_csv(output_file, index=False)
print("Done.")

print("\n========== SUMMARY ==========")
print(f"Original rows  : {df.shape[0]:,}")
print(f"Balanced rows  : {df_balanced.shape[0]:,}")
print(f"Features       : {df_balanced.shape[1] - 1}")
print(f"Output file    : {output_file}")
