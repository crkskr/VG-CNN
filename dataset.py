import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from config import CONFIG

def map_features_to_wave(sorted_feature_indices, period):
    n_features = len(sorted_feature_indices)
    x = np.arange(n_features)
    wave_pattern = np.cos(2 * np.pi * x / period)
    wave_sorted_indices = np.argsort(wave_pattern)
    final_order = np.zeros(n_features, dtype=int)
    weak_to_strong_features = sorted_feature_indices[::-1]
    final_order[wave_sorted_indices] = weak_to_strong_features
    return final_order

def get_feature_order(X_train):
    vars = np.var(X_train, axis=0)
    sorted_indices_desc = np.argsort(vars)[::-1]
    return map_features_to_wave(sorted_indices_desc, period=CONFIG['kernel_size'])

class RawGenomicDataset(Dataset):
    def __init__(self, data_dir, x_file, y_file):
        x_path = os.path.join(data_dir, x_file)
        y_path = os.path.join(data_dir, y_file)
        
        x_df = pd.read_csv(x_path, header=0, index_col=0, sep=None, engine='python')
        y_df = pd.read_csv(y_path, header=0, index_col=0, sep=None, engine='python')
        
        x_df.index = x_df.index.astype(str)
        y_df.index = y_df.index.astype(str)
        common = x_df.index.intersection(y_df.index)
        x_df, y_df = x_df.loc[common], y_df.loc[common]
        
        self.feature_names = x_df.columns.tolist() 
        self.X_raw = x_df.values.astype(np.float32)
        self.Y_raw = y_df.values.astype(np.float32)
        if self.Y_raw.shape[1] > 1: 
            self.Y_raw = self.Y_raw[:, 0:1]

    def __len__(self): 
        return len(self.X_raw)
        
    def __getitem__(self, idx): 
        return self.X_raw[idx], self.Y_raw[idx]
