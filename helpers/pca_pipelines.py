# Basic Operations
import pandas as pd
import numpy as np

# Data Visualization
from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def pca_reduced_knn_training_validation(X_scaled,y,pose_names,n_components,n_neighbors,test_size):
    unique_poses = np.unique(pose_names)
    color_values = cm.viridis(np.linspace(0, 1, num=len(unique_poses)))    
    X_train, X_test, y_train, y_test, pose_names_train, pose_names_test = train_test_split(X_scaled, y, pose_names, stratify=y, test_size=test_size)

    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
    model.fit(X_train_pca, y_train)

    y_pred = model.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, pose in enumerate(unique_poses):
        idx = pose_names_test == pose
        axes[0].scatter(X_test_pca[idx, 0], X_test_pca[idx, 1], color=color_values[i], label=pose)
    axes[0].set_title(f'Test Data (PCA {n_components} Components)')
    axes[0].set_xlabel('Principal Component 1')
    axes[0].set_ylabel('Principal Component 2')
    axes[0].grid(True)
    axes[0].set_xlim([-15, 15])
    axes[0].set_ylim([-15, 15])

    # Predictions Plot
    for i, pose in enumerate(unique_poses):
        idx = pose_names_test == pose
        axes[1].scatter(X_test_pca[idx, 0], X_test_pca[idx, 1], color=color_values[i], label=pose)
    axes[1].set_title(f'KNN Predictions (PCA {n_components} Components)')
    axes[1].set_xlabel('Principal Component 1')
    axes[1].grid(True)
    axes[1].set_xlim([-15, 15])
    axes[1].set_ylim([-15, 15])

    # True Labels Plot
    correct = y_pred == y_test
    incorrect = ~correct
    axes[2].scatter(X_test_pca[correct, 0], X_test_pca[correct, 1], c='green', marker='o', alpha=0.7, label='Correct')
    axes[2].scatter(X_test_pca[incorrect, 0], X_test_pca[incorrect, 1], c='red', marker='x', alpha=0.7, label='Incorrect')
    axes[2].set_title(f'Correct/Incorrect Predictions (PCA {n_components} Components)')
    axes[2].set_xlabel('Principal Component 1')
    axes[2].set_ylabel('Principal Component 2')
    axes[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axes[2].grid(True)
    axes[2].set_xlim([-15, 15])
    axes[2].set_ylim([-15, 15])
        
    if len(unique_poses) < 11:
        axes[0].legend(loc='lower center', bbox_to_anchor=(0.5, -0.5), fontsize='small', ncol=2)
        
    plt.tight_layout()
    plt.show()
    
    return accuracy, precision, recall, f1, conf_matrix