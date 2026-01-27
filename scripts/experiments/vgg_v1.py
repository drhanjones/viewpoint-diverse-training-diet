import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm.auto import tqdm
from viewpoint_diverse_training_diet.data_loader.webdataset_loader import (
    DistributedWebDatasetLoader,
    WebDatasetLoader,
)
import wandb
import datetime
from pathlib import Path
import os
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from viewpoint_diverse_training_diet.utils.util import build_names


def make_vgg16(num_classes: int):
    model = models.vgg16(weights=None)
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    return model


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    ddp: bool,
    criterion: nn.Module,
    is_root_process: bool,
):
    """All ranks validate on their portion of data"""
    model.eval()

    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, disable=not is_root_process):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)  # Weighted by batch size
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    # Reduce metrics across all ranks
    if ddp:
        # Sum losses and counts across all GPUs
        metrics = torch.tensor([val_loss, correct, total], device=device)
        torch.distributed.all_reduce(metrics, op=torch.distributed.ReduceOp.SUM)
        val_loss, correct, total = metrics.tolist()

    avg_loss = val_loss / total
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def train(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    n_epoch: int,
    ddp: bool,
    is_root_process: bool,
    use_wandb: bool = True,
    logging_interval: int = 100,
):

    for epoch in range(n_epoch):
        model.train()
        if is_root_process:
            print(f"Epoch {epoch + 1}/{n_epoch}")

        epoch_loss = 0.0
        epoch_corrects = 0
        num_batches = 0
        total_samples = 0

        for batch_idx, (inputs, targets) in enumerate(
            tqdm(train_dataloader, disable=not is_root_process)
        ):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            corrects = outputs.argmax(dim=1).eq(targets).sum().item()

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)  # Weighted by batch size
            epoch_corrects += corrects
            total_samples += inputs.size(0)
            num_batches += 1

            if use_wandb and (batch_idx + 1) % logging_interval == 0:
                if is_root_process:
                    wandb.log(
                        {
                            "batch/train_loss": loss.item(),
                            "batch/moving_avg_loss": epoch_loss / total_samples,
                            "batch/train_accuracy": 100.0 * corrects / inputs.size(0),
                            "batch/moving_avg_accuracy": 100.0
                            * epoch_corrects
                            / total_samples,
                        }
                    )
                    print(
                        f"Batch {batch_idx + 1}, batch loss: {loss.item():.4f}, batch accuracy: {100.0 * corrects / inputs.size(0):.2f}% \n"
                    )
            elif (batch_idx + 1) % logging_interval == 0:
                print(
                    f"Batch {batch_idx + 1}, batch loss: {loss.item():.4f}, batch accuracy: {100.0 * corrects / inputs.size(0):.2f}% \n"
                )
        print(f"Completed epoch {epoch + 1} on Rank {os.environ.get('RANK', 'N/A')} \n")

        if ddp:
            torch.distributed.barrier()
            metrics = torch.tensor(
                [epoch_loss, epoch_corrects, total_samples], device=device
            )
            torch.distributed.all_reduce(metrics, op=torch.distributed.ReduceOp.SUM)
            epoch_loss, epoch_corrects, total_samples = metrics.tolist()

        if is_root_process:
            avg_epoch_loss = epoch_loss / total_samples
            print(
                f"  Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}, accuracy: {100.0 * epoch_corrects / total_samples:.2f}% \n"
            )
            if use_wandb:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train/loss": avg_epoch_loss,
                        "train/accuracy": 100.0 * epoch_corrects / total_samples,
                    }
                )

        val_loss, val_accuracy = validate(
            model, val_dataloader, device, ddp, criterion, is_root_process
        )
        if is_root_process:
            print(
                f"  Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}% \n"
            )
            if use_wandb:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "val/loss": val_loss,
                        "val/accuracy": val_accuracy,
                    }
                )

    if use_wandb:
        wandb.finish()

    return model


def main():
    DATASET_PATH = Path("/scratch-shared/athamma1/diverse_viewpoints_reshuffled_tmp")
    KEY_TO_CATEGORY_MAPPER_PATH = Path(
        "/home/athamma1/Projects/viewpoint_diversity_3d/3D-object-viewpoint-diversity/scripts/post_processing/reshuffle_shards/key_to_class_mapping.json.gz"
    )

    assert DATASET_PATH.exists(), f"Dataset path {DATASET_PATH} does not exist."
    assert KEY_TO_CATEGORY_MAPPER_PATH.exists(), (
        f"Key to category mapper path {KEY_TO_CATEGORY_MAPPER_PATH} does not exist."
    )
    # NUM_CLASSES = 25

    TEST_SETUP = False
    ddp = int(os.environ.get("RANK", -1)) != -1

    if ddp:
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])

        device = torch.device(
            f"cuda:{ddp_local_rank}" if torch.cuda.is_available() else "cpu"
        )
        is_root_process = ddp_rank == 0
        seed_offset = ddp_rank
        torch.accelerator.set_device_index(ddp_local_rank)
        torch.manual_seed(42 + seed_offset)
        wds_loader = DistributedWebDatasetLoader(
            dataset_path=DATASET_PATH,
            key_to_category_mapper_path=KEY_TO_CATEGORY_MAPPER_PATH,
            seed=42,
            batch_size=64,
            num_workers=16,
            train_val_split=0.6,
            rank=ddp_rank,
            world_size=ddp_world_size,
            test_setup=TEST_SETUP,
        )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_root_process = True
        seed_offset = 0
        ddp_rank = 0
        ddp_world_size = 1
        ddp_local_rank = 0
        torch.manual_seed(42 + seed_offset)
        wds_loader = WebDatasetLoader(
            dataset_path=DATASET_PATH,
            key_to_category_mapper_path=KEY_TO_CATEGORY_MAPPER_PATH,
            seed=42,
            batch_size=64,
            train_val_split=0.9,
        )

    NUM_CLASSES = wds_loader.get_num_classes()

    model = make_vgg16(NUM_CLASSES)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    print("Starting training...")
    use_wandb = True
    names_cfg = build_names(model_name="vgg16")
    if use_wandb and is_root_process:
        wandb.init(
            project="viewpoint-diverse-training-diet",
            name=names_cfg["experiment_name"],
            entity="curriculum-of-vision",
            config={
                "architecture": "VGG16",
                "num_classes": NUM_CLASSES,
                "optimizer": "Adam",
                "learning_rate": 1e-3,
                "batch_size": 64 * ddp_world_size,
                "device": str(device),
                "ddp_world_size": ddp_world_size,
            },
        )

        wandb.watch(model, log="all", log_freq=100)

    model = train(
        model=model,
        train_dataloader=wds_loader.train_dataloader,
        val_dataloader=wds_loader.val_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        n_epoch=2,
        ddp=ddp,
        is_root_process=is_root_process,
        use_wandb=use_wandb,
    )

    raw_model = model.module if ddp else model
    save_checkpoint = True

    if ddp:
        torch.distributed.barrier()

    if save_checkpoint and is_root_process:
        checkpoint_path = Path(f"{names_cfg['checkpoint_path']}")
        if not checkpoint_path.parent.exists():
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(raw_model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved to {checkpoint_path}")

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
