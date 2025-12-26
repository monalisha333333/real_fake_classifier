import os
import torch
from tqdm import tqdm
import pandas as pd
from src.utils import accuracy
from torchvision.utils import save_image

def save_masks(masks, mask_dir, paths):
    # Ensure masks are on CPU
    masks = masks.detach().cpu()

    # If masks are [B,1,H,W] → [B,H,W]
    if masks.dim() == 4 and masks.size(1) == 1:
        masks = masks.squeeze(1)

    for i in range(masks.size(0)):
        filename = os.path.basename(paths[i])
        name, _ = os.path.splitext(filename)

        save_path = os.path.join(mask_dir, f"{name}_mask.png")

        # Normalize to [0,1] for PNG
        mask = (masks[i] * 255).to(torch.uint8)
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

        save_image(mask, save_path)

def evaluate_model(model, dataloader, device, result_ds,result_dir):
    model.eval()
    total_acc = 0

    with torch.no_grad():
        for index, batch in enumerate(tqdm(dataloader)):
        # for index, batch in enumerate(dataloader):
        # for images, labels in dataloader:
            images, labels, paths  = batch[0], batch[1], batch[2]
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # outputs, masks = model(images,False, True)
            # save_masks(masks,result_dir, paths)
            total_acc += accuracy(outputs, labels)
            preds = outputs.argmax(dim=1)
            if index == 0:
                predictions = preds
                targets = labels
                paths_list = paths
            else:
                predictions = torch.cat((predictions, preds), 0)
                targets = torch.cat((targets,labels), 0)
                paths_list += paths
        # Write predictions
        df = pd.DataFrame({
            "image_name": paths_list,
            "target" : targets.cpu().numpy(),
            "predicted" : predictions.cpu().numpy()
        })
        df.to_csv(result_ds, index=False)    

    return total_acc / len(dataloader)