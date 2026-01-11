import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from cuml.decomposition import PCA
import cupy as cp

# PCA Detector (Extract features using the PCA major image)
class PCADetector(torch.nn.Module):
    def __init__(self, num_bins=256, num_classes=1):
        super().__init__()
        
        self.num_bins = num_bins
        self.pcalike = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=1, bias=False),
            # nn.ReLU(),
        )
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
        # self.weight = nn.Sequential(
        #     nn.Linear(64, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 1)
        # )
        

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

    def compute_pca_major_scikit(self,images):
        """
        images: (B, 3, H, W)
        returns: (B, 1, H, W)  first PCA component per image
        """
        B, C, H, W = images.shape
        assert C == 3

        pca_majors = []

        for b in range(B):
            img = images[b]                      # (3,H,W)
            img_rgb = img.permute(1, 2, 0)  # (H,W,3)
            flat = img_rgb.reshape(-1, 3)
            flat_np = flat.detach().cpu().numpy()
            pca = PCA(n_components=1)
            pc1 = pca.fit_transform(flat_np)
            pc1 = pc1.reshape(H, W)
            pc1 = (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-8)
            pc1 = torch.from_numpy(pc1).to(images.device)
            pca_majors.append(pc1)
        pca_majors = torch.stack(pca_majors, dim=0)  # (B,H,W)
        return pca_majors.unsqueeze(1)                # (B,1,H,W)

    def compute_pca_major(self,images):
        """
        images: (B, 3, H, W)
        returns: (B, 1, H, W)  first PCA component per image
        """
        B, C, H, W = images.shape
        assert C == 3

        pca_majors = []

        for b in range(B):
            img = images[b]                      # (3,H,W)
            x = img.permute(1, 2, 0).reshape(-1, 3)  # (HW,3)

            # center
            mean = x.mean(dim=0, keepdim=True)
            x_centered = x - mean

            # covariance (3x3)
            cov = x_centered.T @ x_centered / (x_centered.shape[0] - 1)

            # eigen decomposition
            eigvals, eigvecs = torch.linalg.eigh(cov)

            # principal direction (largest eigenvalue)
            pc1 = eigvecs[:, torch.argmax(eigvals)]  # (3,)

            # project pixels
            proj = x_centered @ pc1                  # (HW,)

            # reshape to image
            pca_img = proj.reshape(H, W)
            pca_majors.append(pca_img)

        pca_majors = torch.stack(pca_majors, dim=0)  # (B,H,W)
        return pca_majors.unsqueeze(1)                # (B,1,H,W)


    def forward(self, x, return_pca=False):
        # print("Inside PCA Detector forward")
        # print("Input x shape:", x.shape)
        pca_major=self.compute_pca_major_scikit(x)
        # print("PCA major shape:", pca_major.shape)
        histograms = self.compute_histogram(pca_major)
        # print("Computed histograms shape:", histograms.shape)
        x = histograms.unsqueeze(1)       # (B, 1, num_bins)
        x = self.features(x)     # (B, 64, num_bins)
        x = x.mean(dim=-1)       # global average pooling → (B, num_bins)
        x1 = self.classifier(x)
        if return_pca:
            # lambda_w = self.weight(x)
            # return x1, pca_major, lambda_w
            return x1, pca_major
        return x1
        
    def save_weights(self, weights_path):
        save_params = {k: v.cpu() for k, v in self.state_dict().items()}
        torch.save(save_params, weights_path)

    def load_weights(self, weights_path):
        weights = torch.load(weights_path)
        self.load_state_dict(weights)
