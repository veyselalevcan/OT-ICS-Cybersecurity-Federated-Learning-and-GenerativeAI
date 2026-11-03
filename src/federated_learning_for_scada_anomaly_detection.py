
"""Federated Learning for SCADA Anomaly Detection"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score,
                           recall_score, f1_score, roc_auc_score,
                           classification_report, precision_recall_curve)
from tensorflow import keras
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense
import tensorflow_federated as tff
import shap

# Configuration
SELECTED_SENSORS = ["LIT101", "FIT101", "MV101", "P101"]
NUM_CLIENTS = 3
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 50
BATCH_SIZE = 32

def load_and_preprocess_data(file_paths):
    """Load and preprocess SCADA data from multiple CSV files"""
    dfs = [pd.read_csv(f) for f in file_paths]
    df = pd.concat(dfs, ignore_index=True)
    df = df[SELECTED_SENSORS]

    # Normalization
    scaler = MinMaxScaler()
    df[SELECTED_SENSORS] = scaler.fit_transform(df[SELECTED_SENSORS])

    return df, scaler

def prepare_federated_data(df):
    """Prepare data for federated learning"""
    client_data = np.array_split(df, NUM_CLIENTS)
    clients = []

    for data in client_data:
        X_train, X_test, y_train, y_test = train_test_split(
            data, data, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        clients.append((X_train, X_test, y_train, y_test))

    return clients

def create_fl_model(input_shape=len(SELECTED_SENSORS)):
    """Create base FL model architecture"""
    model = Sequential([
        Dense(16, activation='relu', input_shape=(input_shape,)),
        Dense(8, activation='relu'),
        Dense(input_shape, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_fl_model(clients, rounds=5):
    """Federated learning training process"""
    global_model = create_fl_model()

    for round in range(rounds):
        print(f"🚀 FL Training - Round {round + 1}")

        local_weights = []
        for (X_train, _, y_train, _) in clients:
            local_model = create_fl_model()
            local_model.fit(X_train, y_train,
                          epochs=EPOCHS,
                          batch_size=BATCH_SIZE,
                          verbose=0)
            local_weights.append(local_model.get_weights())

        # Federated averaging
        new_weights = [np.mean(layers, axis=0) for layers in zip(*local_weights)]
        global_model.set_weights(new_weights)
        print(f"✅ FL Model Updated - Round {round + 1}")

    return global_model

def evaluate_model(model, X_test, y_test, threshold):
    """Evaluate model performance"""
    predictions = model.predict(X_test)
    mse = np.mean(np.power(X_test - predictions, 2), axis=1)
    y_pred = (mse > threshold).astype(int)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, mse)
    }

    print("\n📊 Performance Metrics:")
    for name, value in metrics.items():
        print(f"{name.capitalize()}: {value:.4f}")

    return metrics, mse

def find_optimal_threshold(model, X_train, y_train):
    """Find optimal anomaly detection threshold"""
    train_preds = model.predict(X_train)
    train_mse = np.mean(np.power(X_train - train_preds, 2), axis=1)

    precision, recall, thresholds = precision_recall_curve(y_train, train_mse)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    optimal_idx = np.argmax(f1_scores)

    return thresholds[optimal_idx]

def simulate_fdia_attack(df, attack_ratio=0.05):
    """Simulate False Data Injection Attack"""
    attack_indices = np.random.choice(df.index,
                                    size=int(attack_ratio * len(df)),
                                    replace=False)

    # Modify sensor values
    df.loc[attack_indices, "LIT101"] += np.random.uniform(0.1, 0.5, size=len(attack_indices))
    df.loc[attack_indices, "FIT101"] *= np.random.uniform(1.2, 1.5, size=len(attack_indices))

    # Create labels
    df["Attack_Label"] = 0
    df.loc[attack_indices, "Attack_Label"] = 1

    return df, attack_indices

def visualize_attack(df):
    """Visualize FDIA attack on sensor data"""
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["LIT101"], label="LIT101 (Level Sensor)", color="blue", alpha=0.8)
    plt.scatter(df.index[df["Attack_Label"] == 1],
               df["LIT101"][df["Attack_Label"] == 1],
               color="red", label="FDIA Attack", marker="o")
    plt.title("FDIA Attack Simulation: LIT101 Sensor Values")
    plt.xlabel("Time (Index)")
    plt.ylabel("Sensor Value (Normalized)")
    plt.legend()
    plt.grid()
    plt.show()

# Main execution
if __name__ == "__main__":
    # 1. Data Preparation
    file_paths = [f'/content/2021062{day}_{time}.csv' for day, time in
                 [("2_093704", "3_095710"), ("3_110023", "3_120309"),
                  ("3_130632", "3_140948"), ("4_100741", "4_111312"),
                  ("4_122008", "4_132251")]]

    df, scaler = load_and_preprocess_data(file_paths)
    clients = prepare_federated_data(df)

    # 2. Model Training
    fl_model = train_fl_model(clients, rounds=5)
    fl_model.summary()

    # 3. Attack Simulation
    df, attack_indices = simulate_fdia_attack(df.copy())
    visualize_attack(df)

    # 4. Evaluation
    X = df[SELECTED_SENSORS].values
    y = df["Attack_Label"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Find optimal threshold
    optimal_threshold = find_optimal_threshold(fl_model, X_train, y_train)
    print(f"\n🔍 Optimal Threshold: {optimal_threshold:.4f}")

    # Evaluate
    metrics, mse_scores = evaluate_model(fl_model, X_test, y_test, optimal_threshold)

    # SHAP Analysis
    explainer = shap.DeepExplainer(fl_model, X_train[:100])
    shap_values = explainer.shap_values(X_test[:50])
    shap.summary_plot(shap_values, X_test[:50], feature_names=SELECTED_SENSORS)
