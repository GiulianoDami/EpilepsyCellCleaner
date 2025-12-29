import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    return pd.read_csv(file_path)

def save_results(results, output_path):
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

def split_data(data, test_size=0.2, random_state=42):
    features = data.drop('target', axis=1)
    target = data['target']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def normalize_data(data):
    return (data - data.min()) / (data.max() - data.min())

def read_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def write_config(config, config_path):
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)