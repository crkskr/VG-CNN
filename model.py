import torch
import torch.nn as nn
from config import CONFIG

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid_c = nn.Sigmoid()
        self.conv_s = nn.Conv1d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid_s = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = x * self.sigmoid_c(avg_out + max_out)
        avg_out_s = torch.mean(out, dim=1, keepdim=True)
        max_out_s, _ = torch.max(out, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out_s, max_out_s], dim=1)
        out = out * self.sigmoid_s(self.conv_s(x_cat))
        return out

class AdaptiveThresholdReLU(nn.Module):
    def __init__(self, num_channels, init_val=0.05):
        super(AdaptiveThresholdReLU, self).__init__()
        self.bias = nn.Parameter(torch.full((1, num_channels, 1), init_val))
        self.relu = nn.ReLU()
        
    def forward(self, x): 
        return self.relu(x - self.bias)

class SOTA_Model(nn.Module):
    def __init__(self, input_length):
        super(SOTA_Model, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=CONFIG['kernel_size'], stride=5, padding=CONFIG['kernel_size']//2)
        self.bn1 = nn.BatchNorm1d(32)
        self.act1 = AdaptiveThresholdReLU(32, init_val=0.05)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.act2 = AdaptiveThresholdReLU(64, init_val=0.05)
        
        self.pool = nn.MaxPool1d(2)
        self.attn = CBAM(64, ratio=CONFIG['cbam_ratio'], kernel_size=7)
        
        dummy = torch.zeros(1, 1, input_length)
        with torch.no_grad():
            x = self.conv1(dummy)
            x = self.pool(self.conv2(x))
            out_dim = x.view(1, -1).shape[1]

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(CONFIG['dropout_rate']), 
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.pool(self.act2(self.bn2(self.conv2(x))))
        x = self.attn(x)
        return self.head(x)
