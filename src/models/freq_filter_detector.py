import torch
from diffusers import StableDiffusionPipeline
from .artifact_extractor import VAEReconEncoder
from .fft_filter import fft_filter

# Filtered frequency Detector (Extract middle frequency features using a learned mask)
class FreqFilterDetector(torch.nn.Module):
    def __init__(self, dim_artifact=512, radiuslow=40, radiushigh=120, num_classes=1):
        super().__init__()
        # Load the pre-trained VAE
        model_id = "CompVis/stable-diffusion-v1-4"
        vae = StableDiffusionPipeline.from_pretrained(model_id).vae
        # Freeze the VAE visual encoder
        vae.requires_grad_(False)
        self.artifact_encoder = VAEReconEncoder(vae)
        self.fft_filter_module = fft_filter(radiuslow=radiuslow, radiushigh=radiushigh)
        # Classifier
        self.fc = torch.nn.Linear(dim_artifact, num_classes)
        print("inside FreqFilter detector: torch.nn.Linear(", dim_artifact,",", num_classes,")")
  
    def forward(self, x, return_feat=False, return_mask=False, debug=False):
        freq_filter_image = self.fft_filter_module(x,return_mask, debug)
            
        feat = self.artifact_encoder(freq_filter_image)
        out = self.fc(feat)
        if debug:
            self.debug_tensors = self.fft_filter_module.debug_tensors

        if return_feat and return_mask:
            return feat, out, mask
        if return_feat:
            return feat, out
        if return_mask:
            return out, mask
        return out

    def save_weights(self, weights_path):
        save_params = {k: v.cpu() for k, v in self.state_dict().items()}
        torch.save(save_params, weights_path)

    def load_weights(self, weights_path):
        weights = torch.load(weights_path)
        self.load_state_dict(weights)
