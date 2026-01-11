import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.1):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_ch, out_ch, kernel_size,
            padding=padding, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            out_ch, out_ch, kernel_size,
            padding=padding, dilation=dilation
        )

        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        out = self.conv1(x)
        out = out[:, :, :-self.conv1.padding[0]]  # causal trim
        out = F.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = out[:, :, :-self.conv2.padding[0]]
        out = F.relu(out)
        out = self.dropout(out)

        res = x if self.downsample is None else self.downsample(x)
        return out + res
import torch

def smoothness_features(x):
    """
    x: (B, T) or (B, T, C)
    returns: (B, F)
    """
    if x.dim() == 3:
        x = x.mean(dim=-1)  # collapse channels if needed

    # first derivative
    dx = x[:, 1:] - x[:, :-1]
    # print('dx shape:', dx.shape, 'dx[0]=',x[0, 1:],'-', x[0, :-1])
    # second derivative
    ddx = dx[:, 1:] - dx[:, :-1]

    features = [
        dx.abs().mean(dim=1),            # mean absolute slope
        dx.abs().max(dim=1).values,      # max spike
        dx.var(dim=1),                   # slope variance
        ddx.abs().mean(dim=1),            # curvature
        x.var(dim=1),                    # signal variance
        torch.sum(dx.abs(), dim=1)        # total variation
    ]

    return torch.stack(features, dim=1)  # (B, 6)

# PCA Detector (Extract features using the PCA major image)
class PCADetector(torch.nn.Module):
    def __init__(self, num_bins=256, num_classes=1):
        super().__init__()
        
        self.num_bins = num_bins
        # Temporal Convolutional Network for feature extraction
        layers = []
        channels = [16, 64]
        kernel_size = 3
        for i in range(len(channels)):
            dilation = 2 ** i
            in_ch = 1 if i == 0 else channels[i-1]
            out_ch = channels[i]
            layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size, dilation)
            )

        self.network = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(channels[-1] + 6, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        

    def compute_histogram(self, image):
        # image: (B, C, H, W), float in [0,1]
        assert image.dim() == 4, "Input image must be of shape (B, C, H, W)"
        B, _, _, _ = image.shape
        # Convert to grayscale
        # image: (B, 3, H, W)
        grayimg = image[:, 0:1, :, :]   # keep channel dim
        
        # Flatten spatial dims
        x_flat = grayimg.view(B, -1)  # [B, H*W]
        histograms = []
        for i in range(B):
            hist = torch.histc(
                x_flat[i],
                bins=self.num_bins,
                min=0.0,
                max=1.0
            )
            hist = hist / (hist.sum() + 1e-6)  # normalize
            histograms.append(hist)
        
        histograms = torch.stack(histograms, dim=0)  # [B, num_bins]
        return histograms

    def forward(self, x, return_feat=False):
        histograms = self.compute_histogram(x)
        # x = histograms.unsqueeze(1)       # (B, 1, num_bins)
        tcn_out = self.network(histograms.unsqueeze(1))  # (B, C, num_bins)
        tcn_feat = self.global_pool(tcn_out).squeeze(-1)
        smooth_feat = smoothness_features(histograms)
        # print('smooth_feat shape:', smooth_feat.shape)
        # print('tcn_feat shape:', tcn_feat.shape)
        combined = torch.cat([tcn_feat, smooth_feat], dim=1)
        return self.fc(combined)
    
    def save_weights(self, weights_path):
        save_params = {k: v.cpu() for k, v in self.state_dict().items()}
        torch.save(save_params, weights_path)

    def load_weights(self, weights_path):
        weights = torch.load(weights_path)
        self.load_state_dict(weights)
