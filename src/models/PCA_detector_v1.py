import torch
import torch.nn as nn

# PCA Detector (Extract features using the PCA major image)
class PCADetector_v1(torch.nn.Module):
    def __init__(self, num_bins=256, num_classes=1):
        super().__init__()
        
        self.num_bins = num_bins
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(16),

            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
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
        x = histograms.unsqueeze(1)       # (B, 1, num_bins)
        x = self.features(x)     # (B, 64, num_bins)
        x = x.mean(dim=-1)       # global average pooling → (B, 64)
        return self.classifier(x)
        
    def save_weights(self, weights_path):
        save_params = {k: v.cpu() for k, v in self.state_dict().items()}
        torch.save(save_params, weights_path)

    def load_weights(self, weights_path):
        weights = torch.load(weights_path)
        self.load_state_dict(weights)
