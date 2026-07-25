import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import StratifiedShuffleSplit

# ── Paths ──────────────────────────────────────────────────────────────────────
input_dir  = r"C:\mqttdataset\BCCC-IoT-MQTT-IDS-2025"
output_dir = r"C:\mqttdataset\BCCC-IoT-MQTT-IDS-2025\New folder\strat"
output_file = os.path.join(output_dir, "Dataset_Stratified_10pct.csv")

os.makedirs(output_dir, exist_ok=True)

# ── Find all CSV files ─────────────────────────────────────────────────────────
csv_files = glob.glob(os.path.join(input_dir, "**", "*.csv"), recursive=True)
print(f"Found {len(csv_files)} CSV files")

# ── Load and concatenate in chunks ─────────────────────────────────────────────
CHUNK_SIZE = 100_000
all_chunks = []

for file in csv_files:
    print(f"Reading: {os.path.basename(file)}")
    try:
        for chunk in pd.read_csv(file, chunksize=CHUNK_SIZE, low_memory=False):
            all_chunks.append(chunk)
    except Exception as e:
        print(f"  Skipped {os.path.basename(file)}: {e}")

print("\nConcatenating all chunks...")
df = pd.concat(all_chunks, ignore_index=True)
print(f"Total rows loaded: {df.shape[0]:,}")
print(f"Total columns    : {df.shape[1]}")

# ── Detect label column ────────────────────────────────────────────────────────
label_col = next(
    (c for c in df.columns if c.lower() in ['label', 'attack', 'class', 'category']),
    None
)
if label_col is None:
    raise ValueError("Could not auto-detect label column. Check column names.")

print(f"\nLabel column: '{label_col}'")
print("Class distribution (full dataset):")
print(df[label_col].value_counts())

# ── Drop rows where label is missing ──────────────────────────────────────────
df = df.dropna(subset=[label_col])

# ── Stratified 10% sample ─────────────────────────────────────────────────────
# Ensure at least 1 row per class even for tiny classes
print("\nPerforming stratified 10% sampling...")

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.9, random_state=42)

for sample_idx, _ in sss.split(df, df[label_col]):
    df_sampled = df.iloc[sample_idx].reset_index(drop=True)

print(f"\nSampled rows : {df_sampled.shape[0]:,}")
print("Class distribution (sampled):")
print(df_sampled[label_col].value_counts())

# ── Save ───────────────────────────────────────────────────────────────────────
print(f"\nSaving to {output_file} ...")
df_sampled.to_csv(output_file, index=False)
print("Done.")
