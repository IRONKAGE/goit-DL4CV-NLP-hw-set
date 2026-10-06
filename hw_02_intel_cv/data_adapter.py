import os
import logging
from typing import Tuple, List

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from core.etl_core import SecureDownloader

logger = logging.getLogger("hw_02.data_adapter")

DATA_DIR = "data/hw_02"

def get_image_dataloaders(
    batch_size: int = 32,
    val_split: float = 0.15,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Завантажує, аугментує та розбиває набір даних Intel Image Classification.
    """
    # 1. Завантаження через Smart Router
    downloader = SecureDownloader(
        dataset_path="puneet6060/intel-image-classification",
        data_dir=DATA_DIR,
        zip_name="intel_images.zip"
    )
    downloader.download()

    # Витягуємо лише папки seg_train та seg_test
    extract_path = downloader.extract_atomically(target_extensions=('.jpg', '.jpeg'))

    # Очікувані шляхи після розпакування архіву Kaggle
    train_dir = os.path.join(DATA_DIR, "seg_train", "seg_train")
    test_dir = os.path.join(DATA_DIR, "seg_test", "seg_test")

    # 2. Трансформації та Аугментація
    # Статистика ImageNet для нормалізації (стандарт для ResNet)
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # 3. Завантаження через ImageFolder
    full_train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)

    class_names = full_train_dataset.classes

    # 4. Розділення тренувального набору на Train та Validation
    total_train = len(full_train_dataset)
    val_size = int(total_train * val_split)
    train_size = total_train - val_size

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # Вимикаємо аугментацію для валідації (технічний хак PyTorch)
    val_dataset.dataset.transform = test_transform

    # 5. Створення DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    logger.info(f"✅ Дані готові! Класи: {class_names}")
    logger.info(f"📊 Зображень -> Train: {train_size}, Val: {val_size}, Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader, class_names
