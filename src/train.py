import os
import torch
from src.utils import accuracy, save_checkpoint
from tqdm import tqdm
from collections import Counter
import torchvision.utils as vutils

def train_one_epoch(model, loader, val_loader, criterion, optimizer, device, 
    tolerance_limit, max_val_acc, checkpoint_dir):
    
    model.train()
    epoch_loss, epoch_acc = 0, 0
    loss_h = []
    acc_h = []
    val_loss_h = []
    val_acc_h = []
    # no_acc_increase_cnt = 0

    for batch, (images, labels, paths) in enumerate(tqdm(loader)):
        # print("Train labels ",Counter(labels))
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        # if batch == 5:
        #     outputs = model(images,debug=True)
        #     os.makedirs("debug", exist_ok=True)

        #     for name, tensor in model.debug_tensors.items():
        #         # ensure tensor is real
        #         if torch.is_complex(tensor):
        #             tensor = torch.abs(tensor)

        #         vutils.save_image(
        #             tensor,
        #             f"debug/{name}_{batch}.png",
        #             normalize=True
        #         )
        # else:
        #     outputs = model(images)
        outputs = model(images)
        loss = criterion(outputs, labels)
        acc = accuracy(outputs, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc
        loss_h.append(loss.item())
        acc_h.append(acc)

        # val_loss, val_acc = validate(
        #     model, val_loader, criterion, device
        # )
        # val_loss_h.append(val_loss)
        # val_acc_h.append(val_acc)
        
        # if val_acc > max_val_acc:
        #     max_val_acc = val_acc
        #     torch.save(model.state_dict(), checkpoint_dir+"best_acc.pth")
        #     no_acc_increase_cnt = 0
        # else:
        #     no_acc_increase_cnt += 1
    
        # if no_acc_increase_cnt > tolerance_limit:
        #     print("Training ends as no increase in accuracy in last ",no_acc_increase_cnt, " batches.")
        #     break


    return epoch_loss / len(loader), epoch_acc / len(loader), loss_h, acc_h, val_loss_h, val_acc_h, max_val_acc


def validate(model, loader, criterion, device):
    model.eval()
    epoch_loss, epoch_acc = 0, 0

    with torch.no_grad():
        for images, labels, _ in loader:
            # print("Val labels ",Counter(labels))
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            acc = accuracy(outputs, labels)

            epoch_loss += loss.item()
            epoch_acc += acc

    return epoch_loss / len(loader), epoch_acc / len(loader)
