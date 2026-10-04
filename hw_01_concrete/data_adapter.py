# ==============================================================================
# 🗃️ DATA ADAPTER: Concrete Strength Dataset (HW 01)
# ==============================================================================
"""
Адаптер даних для першого проекту.
Забезпечує завантаження сирих CSV даних, розщеплення на Train/Validation,
безпечне масштабування ознак та генерацію PyTorch DataLoaders.
"""

import os
import urllib.request
import logging
from typing import Tuple

import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

# Імпорти з нашого MLOps-ядра
from core.datasets import SmartTabularDataset
from core.etl_core import SecureDownloader

logger = logging.getLogger("hw_01.data_adapter")

# ------------------------------------------------------------------------------
# 1. КОНФІГУРАЦІЯ НАБОРУ ДАНИХ
# ------------------------------------------------------------------------------
DATA_DIR = "data/hw_01"
CSV_FILENAME = "concrete.csv"
# Пряме посилання на відкритий набір даних Yeh Concrete (ідеальний для регресії)
FALLBACK_URL = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/concrete.csv"

def download_concrete_csv(target_path: str) -> None:
    """Резервний генератор для прямого завантаження CSV без Kaggle API."""
    logger.info(f"Завантаження сирого CSV з {FALLBACK_URL}...")
    urllib.request.urlretrieve(FALLBACK_URL, target_path)

# ------------------------------------------------------------------------------
# 2. ФАБРИКА DATALOADERS (Головна логіка)
# ------------------------------------------------------------------------------
def get_dataloaders(
    batch_size: int = 32,
    test_size: float = 0.2,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, StandardScaler]:
    """
    Повний пайплайн підготовки даних для навчання MLP.
    Повертає: (Train DataLoader, Val DataLoader, Навчений Scaler для інференсу)
    """
    csv_path = os.path.join(DATA_DIR, CSV_FILENAME)

    # 1. Безпечне завантаження через ядро інфраструктури
    if not os.path.exists(csv_path):
        logger.info("📦 Ініціалізація завантаження набору даних міцності бетону...")
        os.makedirs(DATA_DIR, exist_ok=True)

        downloader = SecureDownloader(
            data_dir=DATA_DIR,
            fallback_generator=download_concrete_csv
        )
        # Викликаємо генератор, оскільки ми качаємо прямий файл, а не архів
        downloader.download(target_filename=CSV_FILENAME)

    # 2. Блискавичне читання через Polars (Rust)
    df = pl.read_csv(csv_path)

    # Визначаємо колонки (у цьому наборі даних цільова змінна називається 'strength')
    target_col = "strength"
    feature_cols = [col for col in df.columns if col != target_col]

    # 3. Розбиття на Train та Validation (через Pandas API для сумісності з Sklearn)
    df_pandas = df.to_pandas()
    train_pd, val_pd = train_test_split(df_pandas, test_size=test_size, random_state=seed)

    # 4. Масштабування ознак (Z-Score Normalization)
    # ВАЖЛИВО: Навчаємо Scaler ТІЛЬКИ на тренувальній вибірці (запобігання Data Leakage)
    scaler = StandardScaler()
    train_pd[feature_cols] = scaler.fit_transform(train_pd[feature_cols])
    val_pd[feature_cols] = scaler.transform(val_pd[feature_cols])

    # Повертаємо оброблені дані у формат Polars для нашого SmartTabularDataset
    train_df = pl.from_pandas(train_pd)
    val_df = pl.from_pandas(val_pd)

    # 5. Обгортання у PyTorch Datasets із Graceful Degradation
    train_dataset = SmartTabularDataset(train_df, target_col, feature_cols)
    val_dataset = SmartTabularDataset(val_df, target_col, feature_cols)

    # 6. Створення PyTorch DataLoaders
    # drop_last=True для тренування запобігає помилкам з нестандартним розміром батчу
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    logger.info(f"✅ DataLoaders готові! Тренувальних батчів: {len(train_loader)}, Валідаційних: {len(val_loader)}")

    return train_loader, val_loader, scaler
