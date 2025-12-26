from matplotlib import pyplot as plt

def plot_metrics(train_losses, train_accuracies, val_losses, val_accuracies,plot_dir):
    plt.figure(figsize=(12, 5))

    epochs = range(1, len(train_losses) + 1)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Val Loss', marker='o')
    plt.title('Losses over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Acc', marker='o')
    plt.plot(epochs, val_accuracies, label='Val Acc', marker='o')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(plot_dir+"train_Metrics.png")


def plot_train_batch_metrics(train_losses, train_accuracies, val_losses, val_accuracies,plot_dir):
    plt.figure(figsize=(12, 5))

    epochs = range(1, len(train_losses) + 1)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    if len(val_losses)>0:
        plt.plot(epochs, val_losses, label='Val Loss', marker='o')
    plt.title('Losses over batches')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Acc', marker='o')
    if len(val_accuracies)>0:
        plt.plot(epochs, val_accuracies, label='Val Acc', marker='o')
    plt.title('Accuracy over batches')
    plt.xlabel('Batch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(plot_dir+"train_batch_Metrics.png")
