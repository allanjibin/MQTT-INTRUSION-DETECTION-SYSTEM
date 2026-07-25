import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ── Paths ──────────────────────────────────────────────────────────────────────
input_file  = r"C:\mqttdataset\BCCC-IoT-MQTT-IDS-2025\New folder\strat\Dataset_Stratified_10pct.csv"
output_dir  = r"C:\mqttdataset\BCCC-IoT-MQTT-IDS-2025\New folder\strat"
output_file = os.path.join(output_dir, "Dataset_138_Features.csv")

os.makedirs(output_dir, exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading stratified dataset...")
df = pd.read_csv(input_file, low_memory=False)
print(f"Shape: {df.shape}")

# ── Detect label column ────────────────────────────────────────────────────────
label_col = next(
    (c for c in df.columns if c.lower() in ['label', 'attack', 'class', 'category']),
    None
)
if label_col is None:
    raise ValueError("Could not auto-detect label column.")
print(f"Label column: '{label_col}'")
print("Class distribution:")
print(df[label_col].value_counts())

# ── Separate features and label ────────────────────────────────────────────────
X = df.drop(columns=[label_col])
y = df[label_col]

# ── Convert all features to numeric ───────────────────────────────────────────
print("\nConverting features to numeric...")
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
print("Done.")

# ── Encode labels ──────────────────────────────────────────────────────────────
le = LabelEncoder()
y_enc = le.fit_transform(y)
print(f"\nClasses: {list(le.classes_)}")

# ── Train/test split for feature selection ────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# ── One-vs-Rest XGBoost feature importance ────────────────────────────────────
print("\nRunning One-vs-Rest XGBoost for feature importance...")
n_classes = len(le.classes_)
feature_importance_sum = np.zeros(X.shape[1])

for i, class_name in enumerate(le.classes_):
    print(f"  Training OvR for class: {class_name} ({i+1}/{n_classes})")
    y_binary = (y_train == i).astype(int)

    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    model.fit(X_train, y_binary)
    feature_importance_sum += model.feature_importances_

# ── Rank and select top 138 features ──────────────────────────────────────────
print("\nRanking features by cumulative OvR importance...")
feature_scores = pd.Series(feature_importance_sum, index=X.columns)
feature_scores = feature_scores.sort_values(ascending=False)

TOP_N = 138
top_features = feature_scores.head(TOP_N).index.tolist()

print(f"\nTop {TOP_N} features selected:")
for rank, (feat, score) in enumerate(feature_scores.head(TOP_N).items(), 1):
    print(f"  {rank:>3}. {feat:<50} score: {score:.6f}")

# ── Save feature list ──────────────────────────────────────────────────────────
feature_list_path = os.path.join(output_dir, "selected_138_features.txt")
with open(feature_list_path, 'w') as f:
    for feat in top_features:
        f.write(feat + '\n')
print(f"\nFeature list saved to: {feature_list_path}")

# ── Save importance plot ───────────────────────────────────────────────────────
plt.figure(figsize=(12, 8))
feature_scores.head(TOP_N).plot(kind='bar')
plt.title(f'Top {TOP_N} Feature Importances (One-vs-Rest XGBoost)')
plt.xlabel('Feature')
plt.ylabel('Cumulative Importance Score')
plt.xticks([])
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "feature_importance_top138.png"), dpi=120)
plt.show()
print("Importance plot saved.")

# ── Save label encoder ────────────────────────────────────────────────────────
joblib.dump(le, os.path.join(output_dir, "label_encoder.pkl"))
print("Label encoder saved.")

# ── Build reduced dataset ──────────────────────────────────────────────────────
print(f"\nBuilding reduced dataset with {TOP_N} features + label...")
df_reduced = df[top_features + [label_col]].copy()
print(f"Reduced shape: {df_reduced.shape}")

# ── Save reduced dataset ───────────────────────────────────────────────────────
print(f"Saving to {output_file} ...")
df_reduced.to_csv(output_file, index=False)
print("Done.")

print("\n========== SUMMARY ==========")
print(f"Original features : {X.shape[1]}")
print(f"Selected features : {TOP_N}")
print(f"Output file       : {output_file}")
print(f"Rows              : {df_reduced.shape[0]:,}")
