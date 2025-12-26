import torch.nn as nn
from transformers import CLIPModel
import torchvision.models as models
from src.models import RealDetector_v0
from src.models import CospyDetector
from src.models import FreqFilterDetector
from src.models import PCADetector
from src.models import FreqDetector


def get_model(model_name="resnet50"):
    if model_name == "resnet50":
        model = models.resnet50(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)  # binary
    elif model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)
    elif model_name == "cospy":
        model = CospyDetector(num_classes=2)
        print("getting model architecture from CospyDetector")
    elif model_name == "real_detector_v0":
        model = RealDetector_v0(num_classes=2)
        print("getting model architecture from RealDetector_v0")
    elif model_name == "freq_filter":
        model = FreqFilterDetector(num_classes=2)
        print("getting model architecture from FreqFilterDetector")
    elif model_name == "freq":
        model = FreqDetector(num_classes=2)
        print("getting model architecture from FreqDetector")
    elif model_name == "PCA":
        model = PCADetector(num_classes=2)
        print("getting model architecture from PCADetector")
    elif model_name == "clip_vit":
        model, self.preprocess = clip.load(name, device="cpu") # self.preprecess will not be used during training, which is handled in Dataset class 
        modelfc = nn.Linear( CHANNELS[name], 2)

    
        class CLIP_ViT_Classifier(nn.Module):
            def __init__(self):
                super().__init__()

                # Load full CLIP model
                clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

                # Extract ONLY the vision encoder
                self.backbone = clip.vision_model

                # CLIP ViT-B/32 output is 512-d
                feature_dim = 512

                # Add final layer for 2-class classification
                self.classifier = nn.Linear(feature_dim, 2)

            def forward(self, x):
                # CLIP expects pixel_values argument
                out = self.backbone(pixel_values=x)

                # pooler_output → (batch, 512)
                feats = out.pooler_output

                logits = self.classifier(feats)
                return logits
        
        return CLIP_ViT_Classifier()
    else:
        raise ValueError("Model not supported")

    return model
