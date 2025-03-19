import os

os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
os.environ["MLFLOW_ENABLE_ROCM_MONITORING"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from skimage import io
import numpy as np
from timm.layers import BlurPool2d
from torch.nn.utils import weight_norm
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.pytorch
import json
import psutil
import pickle


def log_system_metrics(epoch):
    # Measure CPU usage over a short interval (1 second)
    cpu_usage = psutil.cpu_percent(interval=1)
    # Measure memory usage (percentage of used RAM)
    mem_usage = psutil.virtual_memory().percent
    # Log these metrics to MLflow using the current epoch as the step
    mlflow.log_metric("cpu_usage", cpu_usage, step=epoch)
    mlflow.log_metric("memory_usage", mem_usage, step=epoch)
    print(f"Epoch {epoch}: CPU usage = {cpu_usage}%, Memory usage = {mem_usage}%")

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

class SEMDataset(Dataset):
    def __init__(self, images_dir, labels_csv, transform=None):
        self.images_dir = images_dir
        self.labels_df = pd.read_csv(labels_csv)
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        filename = row['filename']
        label_db = row['snr_db']
        image_path = os.path.join(self.images_dir, filename)
        image_tensor = torch.load(image_path, weights_only=True)
        if self.transform:
            image_tensor = self.transform(image_tensor)
        return image_tensor, label_db

class EvoNormB0(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(EvoNormB0, self).__init__()
        # Learnable scaling and shifting parameters
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.beta  = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        # Learnable parameter v that modulates the input (for nonlinearity)
        self.v     = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.eps   = eps

    def forward(self, x):
        # x shape: (B, C, H, W)
        # Compute instance variance (over spatial dimensions)
        inst_var = x.var(dim=[2, 3], keepdim=True, unbiased=False)
        # Compute batch variance (over batch and spatial dimensions)
        batch_var = x.var(dim=[0, 2, 3], keepdim=True, unbiased=False)
        # Compute the denominator using a max between the batch std and a modulated version of x plus instance std
        denom = torch.max(torch.sqrt(batch_var + self.eps), self.v * x + torch.sqrt(inst_var + self.eps))
        return (x / denom) * self.gamma + self.beta

class ECALayer(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((np.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)


class PSDGuidedFusion(nn.Module):
    def __init__(self, channels=16):
        super().__init__()
        self.psd_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.var_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.raw_proj = nn.Conv2d(channels, channels, kernel_size=1)
        
        # Learn frequency-dependent fusion weights
        self.freq_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.GELU(),
            nn.Conv2d(channels // 4, 3, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, raw, var, psd):
        # Project to common space
        p_psd = self.psd_proj(psd)
        p_var = self.var_proj(var)
        p_raw = self.raw_proj(raw)
        
        # Frequency-adaptive weighting
        att_weights = self.freq_att(p_psd)  # Expected shape: [B, 3, 1, 1]
        att0 = att_weights[:, 0:1, :, :]  # Shape: [B, 1, 1, 1]
        att1 = att_weights[:, 1:2, :, :]
        att2 = att_weights[:, 2:3, :, :]
        
        fused = att0 * p_raw + att1 * p_var + att2 * p_psd
        
        return fused


###############################################
# 2. Define the CNN Model that Fuses Statistical Features
###############################################
class CNNFusionModel(nn.Module):
    def __init__(self):
        super(CNNFusionModel, self).__init__()
        fusion_channels = 16

        # Projection layers to map 1-channel maps to fusion_channels
        self.raw_proj_conv = nn.Conv2d(1, fusion_channels, kernel_size=3, padding=1)
        self.var_proj_conv = nn.Conv2d(1, fusion_channels, kernel_size=3, padding=1)
        self.psd_proj_conv = nn.Conv2d(1, fusion_channels, kernel_size=3, padding=1)

        # PSD-guided fusion module
        self.psd_guided_fusion = PSDGuidedFusion(channels=fusion_channels)

        # CNN backbone: Note that the input now has fusion_channels instead of 3
        self.conv1 = nn.Sequential(
            nn.Conv2d(fusion_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Halve spatial dimensions.
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Assuming input images are 256x256, after 3 pooling layers: 256/8 = 32.
        self.fc = nn.Sequential(
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Predict SNR (in dB)
        )

    def forward(self, x):
        # x: (B, 1, H, W) – raw image
        B, C, H, W = x.size()

        # Compute Local Variance Map (still single channel)
        local_var = self.compute_local_variance(x)  # (B, 1, H, W)

        # Compute PSD Map (log scale, single channel)
        psd_map = self.compute_psd_map(x)           # (B, 1, H, W)

        # Project each map to fusion_channels
        raw_16 = self.raw_proj_conv(x)         # (B, fusion_channels, H, W)
        var_16 = self.var_proj_conv(local_var)   # (B, fusion_channels, H, W)
        psd_16 = self.psd_proj_conv(psd_map)     # (B, fusion_channels, H, W)

        # Fuse the features using PSDGuidedFusion
        fused = self.psd_guided_fusion(raw_16, var_16, psd_16)  # (B, fusion_channels, H, W)

        # Pass the fused representation through the CNN backbone
        out = self.conv1(fused)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(B, -1)
        out = self.fc(out)
        return out

    def compute_local_variance(self, x):
        # Compute the local variance using a sliding window (kernel_size=7)
        kernel_size = 7
        pad = kernel_size // 2
        mean = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)
        mean_sq = F.avg_pool2d(x * x, kernel_size=kernel_size, stride=1, padding=pad)
        variance = mean_sq - mean * mean
        return variance

    def compute_psd_map(self, x: torch.Tensor, eps: float = 1e-8, fft_shift: bool = True, normalize: bool = True) -> torch.Tensor:
        """
        Compute the log power spectral density (PSD) map from an image.
    
        Args:
            x (Tensor): Input tensor with shape (B, C, H, W).
            eps (float): Small constant for numerical stability in the log transform.
            fft_shift (bool): If True, applies FFT shift to center the zero-frequency component.
            normalize (bool): If True, normalizes the log PSD map per image to zero mean and unit variance.
    
        Returns:
            Tensor: Log PSD map with shape (B, 1, H, W).
        """
        B, C, H, W = x.shape
    
        # Convert to grayscale if input has multiple channels (assumes RGB)
        if C > 1:
            x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    
        fft = torch.fft.fft2(x)
        if fft_shift:
            fft = torch.fft.fftshift(fft, dim=(-2, -1))
        psd = torch.abs(fft) ** 2
        psd_log = torch.log(psd + eps)
    
        if normalize:
            mean = psd_log.mean(dim=(-2, -1), keepdim=True)
            std = psd_log.std(dim=(-2, -1), keepdim=True) + eps
            psd_log = (psd_log - mean) / std
    
        if psd_log.dim() == 3:
            psd_log = psd_log.unsqueeze(1)
    
        return psd_log

# Paths configuration (update these paths to match your environment)
images_dir = r'C:\Users\USER\Downloads\Biofilm SEM Dataset PT FORMAT\Noisy'
labels_csv = r"C:\Users\USER\Downloads\Biofilm SEM Dataset PT FORMAT\Label\labels.csv"

# Initialize dataset
full_dataset = SEMDataset(images_dir, labels_csv)

# Dataset split: 60% train, 20% val, 20% test
torch.manual_seed(42)
total_size = len(full_dataset)
train_size = int(0.6 * total_size)
val_size = int(0.2 * total_size)
test_size = total_size - train_size - val_size

train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16, shuffle=False)
test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model, Loss, Optimizer
model = CNNFusionModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.00001)

# Training and validation loop
num_epochs = 100
best_val_loss = np.inf

total_params = sum(p.numel() for p in model.parameters())
############################################
# MLflow Integration and Training Execution
############################################
# Set MLflow tracking URI and experiment name
mlflow.enable_system_metrics_logging()

mlflow.set_tracking_uri("file:///C:\\Users\\USER\\mlruns")  # Modify as needed
mlflow.set_experiment("Conference Paper")
with mlflow.start_run():
    mlflow.log_param("total_params", total_params)
    mlflow.log_params({
        "img_size": 256,
        "batch_size": 16,
        "lr": 0.001,
        "weight_decay": 1e-5,
        "random_seed": 42,
        "num_epochs": 100,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "optimizer": "AdamW",
    })

    best_val_loss = np.inf

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            if images.ndim == 3:
                images = images.unsqueeze(1)
            labels = labels.float().to(device).unsqueeze(1)

            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                if images.ndim == 3:
                    images = images.unsqueeze(1)
                labels = labels.to(device).unsqueeze(1)
                preds = model(images)
                loss = criterion(preds, labels)
                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_loader.dataset)

        # Log losses to MLflow
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)

        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")

    # Load best model
    model.load_state_dict(torch.load("best_model.pth"))

    # Evaluate on test set
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            if images.ndim == 3:
                images = images.unsqueeze(1)
            preds = model(images).cpu().numpy().flatten()
            labels = labels.numpy().flatten()

            y_pred.extend(preds)
            y_true.extend(labels)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, a_min=1e-6, a_max=None))) * 100
    r2 = r2_score(y_true, y_pred)

    # Log evaluation metrics
    mlflow.log_metrics({
        "test_MAE": mae,
        "test_MSE": mse,
        "test_RMSE": rmse,
        "test_MAPE": mape,
        "test_R2": r2
    })

    # Print metrics
    print("\nTest Evaluation Metrics:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R-squared: {r2:.4f}")

    # Save model artifact to MLflow
    mlflow.pytorch.log_model(model, "cnn_fusion_model")
