import torch
from diffusers import StableDiffusionPipeline
from .artifact_extractor import VAEReconEncoder
from sklearn.decomposition import PCA
import torch.nn as nn

# PCA Detector (Extract features using the PCA major image)
class PCADetector(torch.nn.Module):
    def __init__(self, dim_artifact=512, num_bins=256, num_classes=1):
        super().__init__()
        
        self.num_bins = num_bins
        # Load the pre-trained VAE
        # model_id = "CompVis/stable-diffusion-v1-4"
        # vae = StableDiffusionPipeline.from_pretrained(model_id).vae
        # Freeze the VAE visual encoder
        # vae.requires_grad_(False)
        # self.artifact_encoder = VAEReconEncoder(vae)
        # Classifier
        # self.fc = torch.nn.Linear(dim_artifact, num_classes)
        self.net = nn.Sequential(
            nn.Linear(num_bins, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, num_classes)  # binary classification
        )

    def compute_histogram(self, image):
        # image: (B, C, H, W), float in [0,1]
        # print("Inside compute_histogram")
        # print("Input image shape:", image.shape)
        assert image.dim() == 4, "Input image must be of shape (B, C, H, W)"
        B, _, _, _ = image.shape
        # Convert to grayscale
        # image: (B, 3, H, W)
        grayimg = image[:, 0:1, :, :]   # keep channel dim
        # print("Grey image shape:", grayimg.shape)
        
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
        # print("Inside PCA Detector forward")
        # print("Input x shape:", x.shape)
        histograms = self.compute_histogram(x)
        # print("Computed histograms shape:", histograms.shape)
        out = self.net(histograms)
        return out

    def save_weights(self, weights_path):
        save_params = {k: v.cpu() for k, v in self.state_dict().items()}
        torch.save(save_params, weights_path)

    def load_weights(self, weights_path):
        weights = torch.load(weights_path)
        self.load_state_dict(weights)
