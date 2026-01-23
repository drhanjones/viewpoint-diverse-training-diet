import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from viewpoint_diverse_training_diet.data_loader.webdataset_loader import (
    WebDatasetLoader,
)
import wandb


# ---- model (FROM SCRATCH) ----
def make_vgg16(num_classes: int):
    model = models.vgg16(weights=None)
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    return model


# ---- train/eval loops ----
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0

    for i, (x, y) in tqdm(enumerate(loader)):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        print(f"Logits shape: {logits.shape}, y shape: {y.unsqueeze(1).shape}")
        print(y)
        print(f"x shape: {x.shape}")
        loss = nn.functional.cross_entropy(logits, y.unsqueeze(1))

        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y.squeeze()).sum().item()
        total += x.size(0)

        if (i + 1) % 50 == 0:
            print(
                f"  Batch {i + 1:04d} | "
                f"train loss {loss_sum / total:.4f} acc {correct / total:.4f}"
            )

    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)

        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

    return loss_sum / total, correct / total


def run_training(
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    num_classes: int,
    epochs: int = 10,
    lr: float = 1e-3,
    use_wandb: bool = True,
    wandb_project: str = "viewpoint-diverse-training-diet",
    wandb_name: str = "vgg16-from-scratch",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize wandb
    if use_wandb:
        print("Initializing wandb...")
        wandb.init(
            project=wandb_project,
            name=wandb_name,
            entity="curriculum-of-vision",
            config={
                "architecture": "VGG16",
                "num_classes": num_classes,
                "epochs": epochs,
                "learning_rate": lr,
                "optimizer": "Adam",
                "device": device,
            },
        )

    model = make_vgg16(num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Log model to wandb
    if use_wandb:
        wandb.watch(model, log="all", log_freq=100)

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch:02d} -------------------------------")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.4f}"
        )

        # Log metrics to wandb
        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": tr_loss,
                    "train/accuracy": tr_acc,
                    "val/loss": va_loss,
                    "val/accuracy": va_acc,
                }
            )

    if use_wandb:
        wandb.finish()

    return model


from pathlib import Path


def main():
    DATASET_PATH = Path("/scratch-shared/athamma1/diverse_viewpoints_reshuffled_tmp")
    KEY_TO_CATEGORY_MAPPER_PATH = Path(
        "/home/athamma1/Projects/viewpoint_diversity_3d/3D-object-viewpoint-diversity/assets/key_to_category_mapper.json.gz"
    )

    assert DATASET_PATH.exists(), f"Dataset path {DATASET_PATH} does not exist."
    assert KEY_TO_CATEGORY_MAPPER_PATH.exists(), (
        f"Key to category mapper path {KEY_TO_CATEGORY_MAPPER_PATH} does not exist."
    )
    NUM_CLASSES = 25

    print("Initializing WebDatasetLoader...")
    wds_loader = WebDatasetLoader(
        dataset_path=DATASET_PATH,
        key_to_category_mapper_path=KEY_TO_CATEGORY_MAPPER_PATH,
        seed=42,
        batch_size=32,
    )
    print("WebDatasetLoader initialized.")

    print("Starting training...")
    model = run_training(
        train_loader=wds_loader.train_dataloader,
        val_loader=wds_loader.val_dataloader,
        num_classes=NUM_CLASSES,
        epochs=2,
        lr=1e-3,
    )

    save_checkpoint = True

    if save_checkpoint:
        checkpoint_path = Path("./vgg16_from_scratch.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
