from torchvision import transforms

MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073],
    "resnet":[0.485, 0.456, 0.406],
    "default":[0, 0, 0]
}

STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711],
    "resnet":[0.229, 0.224, 0.225],
    "default":[1, 1, 1]
}
CROP_SIZE = {
    "resnet":[224, 224],
    "cospy":[384, 384],
    "default":[384, 384]
}
def get_transforms(model_name):
    if model_name.lower().startswith("resnet"):
        stats_from = "resnet"
    elif model_name.lower().startswith("clip"):
        stats_from = "clip"
    else:
        stats_from = "default"

    train_tf = transforms.Compose([
        transforms.RandomCrop(CROP_SIZE[stats_from], pad_if_needed=True),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize( mean=MEAN[stats_from], std=STD[stats_from] )
    ])

    val_tf = transforms.Compose([
        transforms.CenterCrop(CROP_SIZE[stats_from]),
        transforms.ToTensor(),
        transforms.Normalize( mean=MEAN[stats_from], std=STD[stats_from] )
    ])

    return train_tf, val_tf
