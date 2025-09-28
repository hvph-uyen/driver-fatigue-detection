import csv
import time
import torch
import torch.nn as nn
import joblib
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

# ----------------- CONFIG -----------------
SAMPLING_RATE = 12  # Hz
WINDOW_SIZE = 2 * SAMPLING_RATE
LABEL_MAP = {'A': 0, 'B': 1, 'C': 2}
LABEL_INV_MAP = {0: 'A', 1: 'B', 2: 'C'}

# ----------------- UTILS -----------------
def balance_data_with_oversampling(X, y):
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    print("After oversampling:", dict(zip(*np.unique(y_resampled, return_counts=True))))
    return X_resampled, y_resampled

def get_class_weights(y, num_classes=3):
    present_classes = np.unique(y)
    class_weights = compute_class_weight(class_weight='balanced', classes=present_classes, y=y)
    weights_tensor = torch.ones(num_classes, dtype=torch.float32)
    for idx, cls in enumerate(present_classes):
        weights_tensor[cls] = class_weights[idx]
    return weights_tensor

def compute_apen(signal, m=2, r_ratio=0.2):
    signal = np.array(signal)
    N = len(signal)
    r = r_ratio * np.std(signal)

    def _phi(m):
        x = np.array([signal[i:i + m] for i in range(N - m + 1)])
        C = np.sum([np.sum(np.max(np.abs(x - xi), axis=1) <= r) - 1 for xi in x]) / (N - m + 1)
        return np.log(C / (N - m + 1) + 1e-10)

    return abs(_phi(m) - _phi(m + 1))

# ----------------- MODEL -----------------
class LightweightFatigueNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 12)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(12, 6)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(6, 3)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

# ----------------- DATA PROCESSING -----------------
def load_and_process_csv(filepath, for_training=True):
    df = pd.read_csv(filepath, on_bad_lines='skip')

    for col in ['yaw_rate_deg', 'steering_angle_deg']:
        df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

    if for_training:
        df['LABEL'] = df['LABEL'].astype(str).str.strip()
        df['LABEL'] = df['LABEL'].map(LABEL_MAP)
        df = df.dropna(subset=['LABEL'])
        df['LABEL'] = df['LABEL'].astype(int)

    X_features, Y_labels, timestamps = [], [], []

    for i in range(0, len(df) - WINDOW_SIZE + 1, WINDOW_SIZE):
        window = df.iloc[i:i + WINDOW_SIZE]
        swa = window['steering_angle_deg'].values
        ya = window['yaw_rate_deg'].values

        apen_swa = compute_apen(swa)
        apen_ya = compute_apen(ya)

        X_features.append([apen_swa, apen_ya])
        timestamps.append(window['time'].iloc[0])

        if for_training:
            label = window['LABEL'].mode()[0]
            Y_labels.append(label)

    if for_training:
        return np.array(X_features), np.array(Y_labels)
    else:
        return np.array(X_features), timestamps

# ----------------- PREDICTION -----------------
def predict_from_csv(csv_path, model, scaler):
    X_new, timestamps = load_and_process_csv(csv_path, for_training=False)
    if len(X_new) == 0:
        return None

    X_scaled = scaler.transform(X_new)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        preds = torch.argmax(outputs, dim=1).numpy()

    labels = [LABEL_INV_MAP[p] for p in preds]
    most_common_label = Counter(labels).most_common(1)[0][0]
    return most_common_label

def load_model_and_scaler(model_path="fatigue_model.pth", scaler_path="scaler.pkl"):
    model = LightweightFatigueNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    scaler = joblib.load(scaler_path)
    print("Model and scaler loaded successfully.")
    return model, scaler
