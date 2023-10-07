import os

import torch
import torchvision
import torchvision.transforms as transforms
from constants import (
    BATCH_SIZE,
    DATA_PATH,
    NUM_WORKERS,
    SIZE_H,
    SIZE_W,
    ckpt_name,
    image_mean,
    image_std,
)
from train import CNNRunner, Flatten

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
Flatten

if __name__ == "__main__":
    transformer = transforms.Compose(
        [
            transforms.Resize((SIZE_H, SIZE_W)),  # scaling images to fixed size
            transforms.ToTensor(),  # converting to tensors
            transforms.Normalize(
                image_mean, image_std
            ),  # normalize image data per-channel
        ]
    )
    # load dataset using torchvision.datasets.ImageFolder
    train_dataset = torchvision.datasets.ImageFolder(
        os.path.join(DATA_PATH, "train_11k"), transform=transformer
    )
    val_dataset = torchvision.datasets.ImageFolder(
        os.path.join(DATA_PATH, "val"), transform=transformer
    )
    # load test data also, to be used for final evaluation
    test_dataset = torchvision.datasets.ImageFolder(
        os.path.join(DATA_PATH, "test_labeled"), transform=transformer
    )
    n_train, n_val, n_test = len(train_dataset), len(val_dataset), len(test_dataset)

    train_batch_gen = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )

    val_batch_gen = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )

    test_batch_gen = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )

    best_model = None
    with open(ckpt_name, "rb") as f:
        best_model = torch.load(f)

    runner = CNNRunner(best_model, None, device, ckpt_name)
    val_stats = runner.validate(val_batch_gen, best_model, phase_name="val")
    test_stats = runner.validate(test_batch_gen, best_model, phase_name="test")

    if val_stats["f1"] > 0.75 and test_stats["f1"] > 0.75:
        print(
            "You have made fully-connected NN perform surprisingly well, call for the assistant."
        )
    else:
        print("Well done, move on to next block to improve performance.")
