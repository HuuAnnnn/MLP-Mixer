import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision import datasets


def load_data(
    save_path: str = "./data",
    batch_size: int = 32,
    train_ratio: float = 0.8,
):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ],
    )

    train_set = datasets.CIFAR10(
        root=save_path,
        train=True,
        download=True,
        transform=transform,
    )
    test_set = datasets.CIFAR10(
        root=save_path,
        train=False,
        download=True,
        transform=transform,
    )

    total_size = len(train_set)
    train_limit = int(train_ratio * total_size)
    val_limit = total_size - train_limit
    train_set, val_set = random_split(train_set, [train_limit, val_limit])

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )
    print("Dataset is loaded successfully!")
    return classes, train_loader, val_loader, test_loader
