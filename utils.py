import matplotlib.pyplot as plt
import torch

from pathlib import Path

import model as mo

def save_model(model: torch.nn.Module, target_dir: str, model_name: str):
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with .pth or .pt"

    MODEL_SAVE_PATH = target_dir_path / model_name
    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=model.state_dict(),
               f=MODEL_SAVE_PATH)

def load_model(model_path: str, device: torch.device):
    loaded_model = mo.ModelV0()
    loaded_model.load_state_dict(torch.load(f=model_path,
                                            map_location=device))
    return loaded_model

def show_image(dataset):
    for i in range(5):

        image, label = dataset[i]
        image = image.squeeze()
        plt.imshow(image, cmap="gray") # show grayscale image
        plt.title(f"Label: {label}")   # show label
        plt.axis("off")
        plt.show()

def show_batch_size(dataLoader):
    image, label = next(iter(dataLoader))
    image = image.squeeze()
    print(f"Image batch shape: {image.shape}")
    print(f"Label batch shape: {label.shape}")

def show_batch_grid(dataLoader):
    images, labels = next(iter(dataLoader))  
    fig, axs = plt.subplots(8, 8, figsize=(10, 10)) 

    for i in range(64):
        image = images[i].squeeze()  
        label = labels[i]

        row, col = divmod(i, 8)
        axs[row, col].imshow(image, cmap="gray")
        axs[row, col].set_title(str(label))
        axs[row, col].axis("off")

    plt.tight_layout()
    plt.show()

def accuracy_fn(y_true, y_pred):
  correct = torch.eq(y_true, y_pred).sum().item()
  acc = (correct/len(y_pred)) * 100
  return acc


def showPredsResult(dataloader, model, device):
    images, labels = next(iter(dataloader))
    with torch.no_grad():
        y_logits = model(images[:5].to(device))

    print("Logits tensor shape:", y_logits.shape)
    print("Logits for the first 5 images:\n", y_logits)

    # To get the predicted class, find the index of the max logit
    y_preds = torch.argmax(y_logits, dim=1)
    print("\nPredicted classes:", y_preds)
    print("Actual labels:    ", labels[:5])

