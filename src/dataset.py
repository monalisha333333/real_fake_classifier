import os
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets
from src.transforms import get_transforms

class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        path = self.samples[index][0]
        return image, label, path

def get_dataloaders(train_dir, val_dir, batch_size, num_workers,model_name):
    train_tf, val_tf = get_transforms(model_name)

    if isinstance(train_dir, list):
        # train_sets = [datasets.ImageFolder(d, transform=train_tf) for d in train_dir]
        train_sets = [ImageFolderWithPaths(d, transform=train_tf) for d in train_dir]
        # i = 0
        # for s in train_sets:
        #     for d in s: 
        #         print(d[1])
        #         i += 1
        # print("Total ", i, "labels.")
        train_set = ConcatDataset(train_sets)
    else:
        # train_set = datasets.ImageFolder(train_dir, transform=train_tf)
        train_set = ImageFolderWithPaths(train_dir, transform=train_tf)
    if isinstance(val_dir, list):
        val_sets = [ImageFolderWithPaths(d, transform=train_tf) for d in val_dir]
        # val_sets = [datasets.ImageFolder(d, transform=val_tf) for d in val_dir]
        val_set = ConcatDataset(val_sets)
    else:
        val_set = ImageFolderWithPaths(val_dir, transform=val_tf)
        # val_set = datasets.ImageFolder(val_dir, transform=val_tf)
    # print(val_set.class_to_idx)

    # print(train_set)
    # return

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)

    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
