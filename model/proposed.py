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
