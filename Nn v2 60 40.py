import pandas as pd
import numpy as np
import time
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, classification_report,
                             confusion_matrix)

tf.random.set_seed(42)
np.random.seed(42)

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
y_train_raw = train_df['label']
X_test  = test_df.drop(columns=['label'])
y_test_raw  = test_df['label']

print("\nConverting all features to numeric...")
X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
X_test  = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)
print("Done.")

# Neural networks are sensitive to feature scale
print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
joblib.dump(scaler, f"{output_dir}\\scaler_NN_v2_60_40.pkl")
print("Done.")

print("\nEncoding labels...")
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train_raw)
y_test_enc  = le.transform(y_test_raw)
joblib.dump(le, f"{output_dir}\\label_encoder_NN_v2_60_40.pkl")
n_classes = len(le.classes_)
print(f"Classes ({n_classes}): {list(le.classes_)}")

y_train_cat = keras.utils.to_categorical(y_train_enc, num_classes=n_classes)
y_test_cat  = keras.utils.to_categorical(y_test_enc, num_classes=n_classes)

n_features = X_train_scaled.shape[1]

model = keras.Sequential([
    layers.Input(shape=(n_features,)),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(n_classes, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)

print("\nTraining Neural Network...")
train_start = time.time()
history = model.fit(
    X_train_scaled, y_train_cat,
    validation_split=0.1,
    epochs=50,
    batch_size=1024,
    callbacks=[early_stop],
    verbose=1
)
train_time = time.time() - train_start
print(f"Training time: {train_time:.4f} seconds")

model.save(f"{output_dir}\\model_NN_v2_60_40.keras")
print("Model saved.")

print("\nTesting...")
test_start = time.time()
y_pred_probs = model.predict(X_test_scaled, batch_size=2048)
y_pred_enc = np.argmax(y_pred_probs, axis=1)
test_time = time.time() - test_start
print(f"Testing time: {test_time:.4f} seconds")

y_pred = le.inverse_transform(y_pred_enc)
y_test_labels = le.inverse_transform(y_test_enc)

pd.DataFrame({'y_true': y_test_labels, 'y_pred': y_pred}).to_csv(
    f"{output_dir}\\Predictions_NN_v2_60_40.csv", index=False)
print("Predictions saved.")

accuracy  = accuracy_score(y_test_labels, y_pred)
f1        = f1_score(y_test_labels, y_pred, average='weighted')
precision = precision_score(y_test_labels, y_pred, average='weighted', zero_division=0)
recall    = recall_score(y_test_labels, y_pred, average='weighted', zero_division=0)

print("\n========== NEURAL NETWORK RESULTS (v2 60:40) ==========")
print(f"Accuracy:      {accuracy:.4f}")
print(f"F1 Score:      {f1:.4f}")
print(f"Precision:     {precision:.4f}")
print(f"Recall:        {recall:.4f}")
print(f"Training Time: {train_time:.4f} seconds")
print(f"Testing Time:  {test_time:.4f} seconds")
print("\n--- Per-Class Metrics ---")
print(classification_report(y_test_labels, y_pred, zero_division=0))

print("Generating confusion matrix...")
labels = sorted(np.unique(y_test_labels))
cm = confusion_matrix(y_test_labels, y_pred, labels=labels)
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
            xticklabels=labels, yticklabels=labels,
            ax=ax, annot_kws={"size": 9})
plt.title('Confusion Matrix - Neural Network v2 (60:40)\n(% of actual class)', fontsize=13)
plt.ylabel('Actual', fontsize=11)
plt.xlabel('Predicted', fontsize=11)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig(f"{output_dir}\\ConfusionMatrix_NN_v2_60_40.png", dpi=120, bbox_inches='tight')
plt.show()
print("Confusion matrix saved.")

# Training curves — useful to show the mentor the model converged properly
fig2, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history['loss'], label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Val Loss')
axes[0].set_title('Loss')
axes[0].set_xlabel('Epoch')
axes[0].legend()
axes[1].plot(history.history['accuracy'], label='Train Acc')
axes[1].plot(history.history['val_accuracy'], label='Val Acc')
axes[1].set_title('Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].legend()
plt.tight_layout()
plt.savefig(f"{output_dir}\\TrainingCurves_NN_v2_60_40.png", dpi=120, bbox_inches='tight')
plt.show()
print("Training curves saved.")

pd.DataFrame({
    'Model': ['Neural Network'],
    'Dataset': ['v2'],
    'Split': ['60:40'],
    'Accuracy': [accuracy],
    'F1 Score': [f1],
    'Precision': [precision],
    'Recall': [recall],
    'Training Time (s)': [train_time],
    'Testing Time (s)': [test_time]
}).to_csv(f"{output_dir}\\Results_NN_v2_60_40.csv", index=False)
print("Results saved.")
