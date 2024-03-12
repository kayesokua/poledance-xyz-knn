import pandas as pd

data = pd.read_csv("data/training-selected/all/pose_data_augmented_z_res_0301.csv")

p_labels = data['label_encoded'].values
d_labels = data['difficulty'].values

features = pd.concat([data.iloc[:, 6:9], data.iloc[:, 39:93], data.iloc[:, 105:]], axis=1)
feature_names = features.columns

feature_sets = {'logr': ['landmark_00_y', 'a_lft_hip_to_ankle', 'landmark_21_y', 'stomach_y', 'landmark_28_y', 'chest_y', 'landmark_24_y', 'landmark_12_y', 'landmark_18_y', 'a_lft_shoulder_to_wrist'], 'gini': ['stomach_y', 'hip_y', 'chest_y', 'landmark_00_y', 'd_elbows', 'd_wrists', 'landmark_24_y', 'd_rgt_shoulder_to_ankle', 'd_ankles', 'd_knees'], 'permutation': ['a_rgt_hip_to_ankle', 'a_rgt_shoulder_to_wrist', 'd_knees', 'a_lft_hip_to_ankle', 'd_lft_shoulder_to_ankle', 'd_ankles', 'a_mid_hip_to_knees', 'd_rgt_shoulder_to_ankle', 'd_wrists', 'a_lft_shoulder_to_wrist'], 'shap': ['landmark_27_y', 'landmark_00_y', 'hip_y', 'd_lft_shoulder_to_ankle', 'landmark_24_y', 'landmark_26_y', 'a_rgt_hip_to_ankle', 'a_mid_hip_to_knees', 'landmark_14_y', 'a_lft_hip_to_ankle'], 'upper1': ['landmark_11_y', 'landmark_12_y', 'landmark_13_y', 'landmark_14_y', 'landmark_15_y', 'landmark_16_y'], 'upper2': ['chest_y', 'chest_y', 'hip_y', 'a_lft_shoulder_to_wrist', 'a_rgt_shoulder_to_wrist', 'a_nose_to_rgt_shoulder', 'a_nose_to_lft_shoulder', 'd_wrists', 'd_elbows'], 'lower1': ['landmark_23_y', 'landmark_24_y', 'landmark_25_y', 'landmark_26_y', 'landmark_27_y', 'landmark_28_y'], 'lower2': ['d_nose_to_rgt_knee', 'd_nose_to_lft_knee', 'd_rgt_shoulder_to_ankle', 'd_lft_shoulder_to_ankle', 'd_wrists', 'd_elbows', 'd_knees', 'd_ankles', 'a_mid_hip_to_knees', 'a_lft_hip_to_ankle', 'a_rgt_hip_to_ankle']}