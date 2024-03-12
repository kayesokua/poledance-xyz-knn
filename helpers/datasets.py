import pandas as pd
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

def load_train_data(source="dataset/pose_data_augmented_res.csv"):
    data = pd.read_csv(source)
    all_features = data.iloc[:, 5:]
    X = scaler.fit_transform(all_features)
    y = data['label_encoded'].values
    feature_names = all_features.columns
    return X, y, feature_names

def load_feature_weights(source="feature_weights_no_index.csv"):
    weights_df = pd.read_csv(source)
    return weights_df

def load_test_data(source):
    data = pd.read_csv(source)
    all_features = data.iloc[:, 5:]
    scaler = StandardScaler()
    X = scaler.fit_transform(all_features)
    y = data['secs'].values
    feature_names = all_features.columns
    return X, y, feature_names