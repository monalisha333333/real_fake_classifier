import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # must be BEFORE torch import
import time
import logging
import yaml
import json
import torch
import pandas as pd
from src.dataset import get_dataloaders
from src.model import get_model
from src.eval import evaluate_model

def validate(test_dir, batch_size, num_workers, model_name, result_dir, model, device):

    result_ds = result_dir+"/pred_results.csv"
    _, test_loader  = get_dataloaders(
        test_dir, test_dir, 
        batch_size, num_workers,
        model_name
    )
    acc = evaluate_model(model, test_loader, device,result_ds,result_dir)
    print(f"Validation Accuracy: {acc:.4f}")

    with open(result_dir+"/pred_acc.json", "w") as f:
        json.dump({"validation accuracy": acc*100}, f, indent=4)    
    return acc

if __name__=="__main__":
 
    st=time.time() 
    torch.cuda.memory.reset_peak_memory_stats()
    with open("config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    device = cfg["device"]
    result_dir = cfg["result_dir"]+"/"+cfg["model_name"]
    os.makedirs(result_dir,exist_ok=True)
    
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    file_handler = logging.FileHandler(os.path.join(result_dir, "test.log"))
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    model = get_model(cfg["model_name"]).to(device)
    if cfg["model_name"] != "cospy_pre":
        checkpoint_dir = "checkpoints/"+cfg["model_name"]+"/best_acc.pth"
        logger.info("Loading model from ", checkpoint_dir)
        model.load_state_dict(torch.load(checkpoint_dir))
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")

    if isinstance(cfg["test_dir"], list):
        for test_dir in cfg["test_dir"]:
            logger.info(f"Testing on {test_dir} ...")
            data_type=test_dir.split("/")[-1]
            result_dir_datatyp = result_dir+"/"+data_type
            os.makedirs(result_dir_datatyp,exist_ok=True)
            acc = validate(test_dir, cfg["batch_size"], cfg["num_workers"], cfg["model_name"], \
                result_dir_datatyp, model, device)
            logger.info(f"Test on {data_type} dataset - Acc: {acc:.4f}")
    else:
        acc = validate(cfg["test_dir"], cfg["batch_size"], cfg["num_workers"], cfg["model_name"], \
                result_dir, model, device)
        logger.info(f"Test on dataset - Acc: {acc:.4f}")
    logger.info(f"Total Evaluation Time: {time.time()-st:.2f} seconds")
    mem = torch.cuda.max_memory_allocated(0) / (1024**2)
    logger.info(f"Max GPU memory allocated (GPU 0): {mem:.2f} MB")
