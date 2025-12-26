import os
import yaml
import json
import torch
import pandas as pd
from src.dataset import get_dataloaders
from src.model import get_model
from src.eval import evaluate_model

def validate(test_dir, batch_size, num_workers, model_name, result_dir):

    result_ds = result_dir+"/pred_results.csv"
    _, test_loader  = get_dataloaders(
        test_dir, test_dir, 
        batch_size, num_workers,
        model_name
    )
    acc = evaluate_model(model, test_loader, device,result_ds,result_dir)
    print(f"Final Validation Accuracy: {acc:.4f}")

    with open(result_dir+"/pred_acc.json", "w") as f:
        json.dump({"validation accuracy": acc*100}, f, indent=4)    

if __name__=="__main__":
    with open("config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = cfg["device"]
    result_dir = cfg["result_dir"]+"/"+cfg["model_name"]
    os.makedirs(result_dir,exist_ok=True)
    model = get_model(cfg["model_name"]).to(device)
    checkpoint_dir = "checkpoints/"+cfg["model_name"]+"/best_acc.pth"
    print("Loading model from ", checkpoint_dir)
    model.load_state_dict(torch.load(checkpoint_dir))

    if isinstance(cfg["test_dir"], list):
        for test_dir in cfg["test_dir"]:
            data_type=test_dir.split("/")[1]
            result_dir_datatyp = result_dir+"/"+data_type
            os.makedirs(result_dir_datatyp,exist_ok=True)
            validate(test_dir, cfg["batch_size"], cfg["num_workers"], cfg["model_name"], result_dir_datatyp)
    else:
        validate(cfg["test_dir"], cfg["batch_size"], cfg["num_workers"], cfg["model_name"], result_dir)
