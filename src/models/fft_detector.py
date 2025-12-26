import torch
from diffusers import StableDiffusionPipeline
from .artifact_extractor import VAEReconEncoder

# Filtered frequency Detector (Extract middle frequency features using a learned mask)
class FreqDetector(torch.nn.Module):
    def __init__(self, dim_artifact=512, num_classes=1):
        super().__init__()
        # Load the pre-trained VAE
        model_id = "CompVis/stable-diffusion-v1-4"
        vae = StableDiffusionPipeline.from_pretrained(model_id).vae
        # Freeze the VAE visual encoder
        vae.requires_grad_(False)
        self.artifact_encoder = VAEReconEncoder(vae)
        # Classifier
        self.fc = torch.nn.Linear(dim_artifact, num_classes)
        
    def forward(self, x, return_feat=False):
        freq_image = torch.fft.fftn(x * 255, dim=(-2, -1))
        freq_image = torch.fft.fftshift(freq_image, dim=(-2, -1))
        freq_image = torch.log(torch.abs(freq_image) + 1e-8) / 255.0
        feat = self.artifact_encoder(freq_image)
        out = self.fc(feat)
        
        if return_feat:
            return feat, out
        return out

    def save_weights(self, weights_path):
        save_params = {k: v.cpu() for k, v in self.state_dict().items()}
        torch.save(save_params, weights_path)

    def load_weights(self, weights_path):
        weights = torch.load(weights_path)
        self.load_state_dict(weights)
