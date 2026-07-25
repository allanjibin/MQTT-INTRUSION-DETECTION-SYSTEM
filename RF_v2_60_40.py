import pandas as pd
import numpy as np
import time
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, classification_report,
                             confusion_matrix)

train_path = r"C:\mqttdataset\BCCC-IoT-MQTT-IDS-2025\New folder\strat\Train_Balanced_v2_60.csv"
test_path  = r"C:\mqttdataset\BCCC-IoT-MQTT-IDS-2025\New folder\strat\Test_Balanced_v2_40.csv"
output_dir = r"C:\mqttdataset\BCCC-IoT-MQTT-IDS-2025\New folder\strat"

print("Loading training data...")
train_df = pd.read_csv(train_path, low_memory=False)
print(f"Train shape: {train_df.shape}")

print("Loading test data...")
test_df = pd.read_csv(test_path, low_memory=False)
print(f"Test shape: {test_df.shape}")

X_train = train_df.drop(columns=['label'])
y_train = train_df['label']
X_test  = test_df.drop(columns=['label'])
y_test  = test_df['label']

print("\nConverting all features to numeric...")
X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
X_test  = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)
print("Done.")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42,
    verbose=1
)

print("\nTraining Random Forest...")
train_start = time.time()
model.fit(X_train, y_train)
train_time = time.time() - train_start
print(f"Training time: {train_time:.4f} seconds")

joblib.dump(model, f"{output_dir}\\model_RF_v2_60_40.pkl")
print("Model saved.")

print("\nTesting...")
test_start = time.time()
y_pred = model.predict(X_test)
test_time = time.time() - test_start
print(f"Testing time: {test_time:.4f} seconds")

pd.DataFrame({'y_true': y_test, 'y_pred': y_pred}).to_csv(
    f"{output_dir}\\Predictions_RF_v2_60_40.csv", index=False)
print("Predictions saved.")

accuracy  = accuracy_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall    = recall_score(y_test, y_pred, average='weighted', zero_division=0)

print("\n========== RANDOM FOREST RESULTS (v2 60:40) ==========")
print(f"Accuracy:      {accuracy:.4f}")
print(f"F1 Score:      {f1:.4f}")
print(f"Precision:     {precision:.4f}")
print(f"Recall:        {recall:.4f}")
print(f"Training Time: {train_time:.4f} seconds")
print(f"Testing Time:  {test_time:.4f} seconds")
print("\n--- Per-Class Metrics ---")
print(classification_report(y_test, y_pred, zero_division=0))

print("Generating confusion matrix...")
labels = sorted(y_test.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
            xticklabels=labels, yticklabels=labels,
            ax=ax, annot_kws={"size": 9})
plt.title('Confusion Matrix - Random Forest v2 (60:40)\n(% of actual class)', fontsize=13)
plt.ylabel('Actual', fontsize=11)
plt.xlabel('Predicted', fontsize=11)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig(f"{output_dir}\\ConfusionMatrix_RF_v2_60_40.png", dpi=120, bbox_inches='tight')
plt.show()
print("Confusion matrix saved.")

pd.DataFrame({
    'Model': ['Random Forest'],
    'Dataset': ['v2'],
    'Split': ['60:40'],
    'Accuracy': [accuracy],
    'F1 Score': [f1],
    'Precision': [precision],
    'Recall': [recall],
    'Training Time (s)': [train_time],
    'Testing Time (s)': [test_time]
}).to_csv(f"{output_dir}\\Results_RF_v2_60_40.csv", index=False)
print("Results saved.")
