import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import shap
import warnings
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import CONFIG, set_seed
from dataset import RawGenomicDataset, get_feature_order
from model import SOTA_Model

warnings.filterwarnings("ignore")

def train_and_get_all_models(dataset):
    kf = KFold(n_splits=CONFIG['k_folds'], shuffle=True, random_state=CONFIG['seed'])
    all_folds_info = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        set_seed(CONFIG['seed'])
        
        X_train, Y_train = dataset.X_raw[train_idx], dataset.Y_raw[train_idx]
        X_val, Y_val = dataset.X_raw[val_idx], dataset.Y_raw[val_idx]
        
        y_mean, y_std = np.mean(Y_train), np.std(Y_train) + 1e-6
        Y_train = (Y_train - y_mean) / y_std
        Y_val = (Y_val - y_mean) / y_std
        
        f_idx = get_feature_order(X_train)
        X_train = X_train[:, f_idx]
        X_val = X_val[:, f_idx]
        
        x_mean, x_std = np.mean(X_train, axis=0), np.std(X_train, axis=0) + 1e-6
        X_train = (X_train - x_mean) / x_std
        X_val = (X_val - x_mean) / x_std
        
        X_train_t = torch.from_numpy(X_train).unsqueeze(1)
        Y_train_t = torch.from_numpy(Y_train)
        X_val_t = torch.from_numpy(X_val).unsqueeze(1)
        Y_val_t = torch.from_numpy(Y_val)
        
        model = SOTA_Model(input_length=X_train_t.shape[-1]).to(CONFIG['device'])
        optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        criterion = nn.MSELoss()
        
        train_dl = DataLoader(TensorDataset(X_train_t, Y_train_t), batch_size=CONFIG['batch_size'], shuffle=True)
        val_dl = DataLoader(TensorDataset(X_val_t, Y_val_t), batch_size=CONFIG['batch_size'], shuffle=False)
        
        fold_best_r = -1.0
        fold_best_state = None
        
        for epoch in range(CONFIG['epochs']):
            model.train()
            for X, y in train_dl:
                X, y = X.to(CONFIG['device']), y.to(CONFIG['device'])
                optimizer.zero_grad()
                loss = criterion(model(X), y)
                loss.backward()
                optimizer.step()
            
            model.eval()
            preds, truths = [], []
            with torch.no_grad():
                for X, y in val_dl:
                    X = X.to(CONFIG['device'])
                    p = model(X).cpu().numpy().flatten()
                    preds.extend(p)
                    truths.extend(y.numpy().flatten())
            
            curr_r = 0.0
            if np.std(preds) > 1e-6: 
                curr_r, _ = pearsonr(truths, preds)
            scheduler.step(curr_r)
            
            if curr_r > fold_best_r: 
                fold_best_r = curr_r
                fold_best_state = copy.deepcopy(model.state_dict())
                
        all_folds_info.append({
            'fold_id': fold + 1,
            'r_score': fold_best_r,
            'state_dict': fold_best_state,
            'f_idx': f_idx,
            'x_mean': x_mean,
            'x_std': x_std,
            'train_idx': train_idx,
            'val_idx': val_idx
        })

    all_folds_info.sort(key=lambda x: x['r_score'], reverse=True)
    return all_folds_info


def calculate_shap_for_fold(dataset, fold_info, bg_size, output_filename):
    model = SOTA_Model(input_length=dataset.X_raw.shape[1]).to(CONFIG['device'])
    model.load_state_dict(fold_info['state_dict'])
    model.eval()
    
    train_raw = dataset.X_raw[fold_info['train_idx']]
    train_norm = (train_raw - fold_info['x_mean']) / fold_info['x_std']
    train_ordered = train_norm[:, fold_info['f_idx']]
    
    actual_bg_size = min(bg_size, len(train_ordered))
    bg_indices = np.random.choice(len(train_ordered), actual_bg_size, replace=False)
    bg_tensor = torch.from_numpy(train_ordered[bg_indices]).unsqueeze(1).to(CONFIG['device'])
    
    val_raw = dataset.X_raw[fold_info['val_idx']]
    val_norm = (val_raw - fold_info['x_mean']) / fold_info['x_std']
    val_ordered = val_norm[:, fold_info['f_idx']]
    test_tensor = torch.from_numpy(val_ordered).unsqueeze(1).to(CONFIG['device'])
    
    explainer = shap.GradientExplainer(model, bg_tensor)
    shap_scores_list = []
    
    for i in range(len(test_tensor)):
        single_sample = test_tensor[i:i+1]
        shap_values = explainer.shap_values(single_sample)
        
        if isinstance(shap_values, list):
            shap_scores_list.append(np.abs(shap_values[0]))
        else:
            shap_scores_list.append(np.abs(shap_values))
            
        torch.cuda.empty_cache()
        
    shap_scores_ordered = np.concatenate(shap_scores_list, axis=0).mean(axis=(0, 1))

    num_snps = len(dataset.feature_names)
    shap_scores_original = np.zeros(num_snps)
    shap_scores_original[fold_info['f_idx']] = shap_scores_ordered.flatten() 
    
    results_df = pd.DataFrame({
        'Raw_SNP_Name': dataset.feature_names,
        'SHAP_Score': shap_scores_original
    })
    
    results_df = results_df.sort_values(by='SHAP_Score', ascending=False)
    
    try:
        results_df[['Chromosome', 'Position']] = results_df['Raw_SNP_Name'].str.split('_', n=1, expand=True)
    except Exception:
        results_df['Chromosome'] = results_df['Raw_SNP_Name']
        results_df['Position'] = "Unknown"
        
    final_csv_df = results_df[['Chromosome', 'Position', 'SHAP_Score']]
    final_csv_df.to_csv(output_filename, index=False)


if __name__ == "__main__":
    set_seed(CONFIG['seed'])
    dataset = RawGenomicDataset(CONFIG['data_dir'], CONFIG['x_file'], CONFIG['y_file'])
    
    all_folds_info = train_and_get_all_models(dataset)
    
    for i in range(5):
        rank = i + 1
        fold_info = all_folds_info[i] 
        set_seed(CONFIG['seed']) 
        calculate_shap_for_fold(
            dataset=dataset, 
            fold_info=fold_info, 
            bg_size=200, 
            output_filename=f"shap_AGE_Rank{rank}_bg200.csv"
        )
