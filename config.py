import torch
import random
import numpy as np

CONFIG = {
    "data_dir": "data/vega",        
    "x_file": "X_g.csv",
    "y_file": "y_train_age_g.csv",  
    "lr": 0.0005999835231764475,
    "batch_size": 16,
    "weight_decay": 9.070563101514655e-05,
    "kernel_size": 25,    
    "cbam_ratio": 16,     
    "dropout_rate": 0.15, 
    "epochs": 50,
    "k_folds": 5,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 2025,
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
