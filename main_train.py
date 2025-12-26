import os
import yaml
import time
import torch
import logging
from torch import nn, optim
from src.model import get_model
from src.dataset import get_dataloaders
from src.train import train_one_epoch, validate
from src.plot_metrics import plot_metrics, plot_train_batch_metrics

st=time.time()
with open("config/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

device = cfg["device"]
result_dir = cfg["result_dir"]+"/"+cfg["model_name"]+"/"
os.makedirs(result_dir,exist_ok=True)

logger = logging.getLogger("train_logger")
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

file_handler = logging.FileHandler(os.path.join(result_dir, "train.log"))
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

with open(result_dir+"config.yaml", "w") as f:
    yaml.safe_dump(cfg, f)

train_loader, val_loader = get_dataloaders(
    cfg["train_dir"], cfg["val_dir"], 
    cfg["batch_size"], cfg["num_workers"],
    cfg["model_name"],
)

model = get_model(cfg["model_name"]).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"])
max_val_acc = 0
checkpoint_dir = "checkpoints/"+cfg["model_name"]+"/"
os.makedirs(checkpoint_dir,exist_ok=True)

train_losses=[]
train_accuracies=[]
val_losses=[]
val_accuracies=[]
train_batch_losses=[]
train_batch_accuracies=[]
val_batch_losses=[]
val_batch_accuracies=[]

tolerance_limit = 5
no_acc_increase_cnt = 0

for epoch in range(cfg["num_epochs"]):
    train_loss, train_acc, train_loss_h, train_acc_h, val_loss_h, val_acc_h, max_val_acc = train_one_epoch(
        model, train_loader, val_loader, criterion, optimizer, device, tolerance_limit,max_val_acc, checkpoint_dir
    )
    
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    train_batch_losses.extend(train_loss_h)
    train_batch_accuracies.extend(train_acc_h)
    val_batch_losses.extend(val_loss_h)
    val_batch_accuracies.extend(val_acc_h)

    val_loss, val_acc = validate(
        model, val_loader, criterion, device
    )
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    logger.info(
        f"Epoch [{epoch+1}/{cfg['num_epochs']}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_acc:.4f}"
        f"Validation Loss: {val_loss:.4f} "
        f"Validation Acc: {val_acc:.4f}"
    )
    if val_acc > max_val_acc:
        max_val_acc = val_acc
        torch.save(model.state_dict(), checkpoint_dir+"best_acc.pth")
        logger.info(
            f"Model saved at {checkpoint_dir} with val accuracy {max_val_acc*100:.2f}] "
        )
        no_acc_increase_cnt = 0
    else:
        no_acc_increase_cnt += 1
    
    if val_acc == 1:
        logger.info(
            f"Training ends as validation accuracy is 100%" 
        )
        break

    if no_acc_increase_cnt == tolerance_limit:
        logger.info(
            f"Training ends as no increase in accuracy in last {no_acc_increase_cnt} epochs."
        )
        break

et=time.time()
logger.info(
    f"Total time taken (sec): {et-st}"
)

plot_metrics(train_losses, train_accuracies, val_losses, val_accuracies,result_dir)
plot_train_batch_metrics(train_batch_losses, train_batch_accuracies,val_batch_losses, val_batch_accuracies, result_dir)
logger.info(
    f"Training complete."
)