# ==============================================================================
# 🛡️ DATASETS CORE (Smart Reader & Graceful Degradation)
# ==============================================================================
"""
Ядро завантаження та обробки даних (Оновлення: Жовтень 2026).
Реалізує патерн Graceful Degradation (стійкість до битих файлів):
замість того, щоб зупиняти епоху через один пошкоджений JPEG або битий рядок CSV,
система логує помилку і безпечно обробляє або пропускає аномалію.
"""

import os
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import polars as pl
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset

logger = logging.getLogger("core.datasets")

# ------------------------------------------------------------------------------
# 1. БЕЗПЕЧНІ ЗАВАНТАЖУВАЧІ (Safe Loaders)
# ------------------------------------------------------------------------------
def read_tabular_data_safe(file_path: Union[str, Path]) -> Optional[pl.DataFrame]:
    """
    Блискавичне завантаження табличних даних через Polars (Rust).
    Підтримує CSV та Parquet з автоматичним визначенням формату.
    """
    path = Path(file_path)
    if not path.exists():
        logger.error(f"❌ Файл не знайдено: {path}")
        return None

    try:
        if path.suffix == ".parquet":
            df = pl.read_parquet(path)
        else:
            # Використовуємо infer_schema_length для швидкості та null_values для безпеки
            df = pl.read_csv(path, null_values=["NA", "N/A", "null", "NaN", ""])

        logger.info(f"💾 Завантажено {path.name}: {df.shape[0]} рядків, {df.shape[1]} колонок.")
        return df
    except Exception as e:
        logger.error(f"❌ Критична помилка читання {path.name}: {e}")
        return None


def read_image_safe(file_path: Union[str, Path], mode: str = "RGB") -> Optional[Image.Image]:
    """
    Безпечне завантаження зображення через Pillow 12.3+.
    Відловлює биті байти, відсутні файли або неправильні формати.
    """
    try:
        # Pillow ледачий (lazy), тому робимо .convert() та .verify() неявно під час завантаження в пам'ять
        with Image.open(file_path) as img:
            return img.convert(mode)
    except (UnidentifiedImageError, OSError) as e:
        logger.warning(f"⚠️ Битий файл або помилка читання зображення: {file_path} | Деталі: {e}")
        return None
    except Exception as e:
        logger.warning(f"⚠️ Невідома помилка при обробці {file_path}: {e}")
        return None


# ------------------------------------------------------------------------------
# 2. ПАТЕРН GRACEFUL DEGRADATION ДЛЯ PYTORCH DATASETS
# ------------------------------------------------------------------------------
class SmartImageDataset(Dataset):
    """
    Комп'ютерний зір (CV). Інтелектуальний Dataset, який не "падає" через биті файли.
    Використовується в HW 02 (Intel Image) та Final Project (PetFinder).
    """
    def __init__(
        self,
        image_paths: List[Union[str, Path]],
        labels: Optional[List[int]] = None,
        transform: Optional[Callable] = None,
        fallback_image_size: Tuple[int, int] = (224, 224)
    ):
        self.image_paths = [Path(p) for p in image_paths]
        self.labels = labels
        self.transform = transform
        self.fallback_image_size = fallback_image_size

        # Перевірка на невідповідність довжини (щоб уникнути Index Error)
        if self.labels is not None and len(self.image_paths) != len(self.labels):
            raise ValueError("Кількість зображень не співпадає з кількістю міток!")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Any]:
        img_path = self.image_paths[idx]
        img = read_image_safe(img_path)

        # ПАТЕРН: Graceful Degradation
        # Якщо файл битий, генеруємо "порожнє" чорне зображення правильної розмірності.
        # Модель отримає нульовий градієнт, але пайплайн навчання продовжить роботу.
        if img is None:
            img = Image.new("RGB", self.fallback_image_size, (0, 0, 0))

        if self.transform:
            img = self.transform(img)

        label = self.labels[idx] if self.labels is not None else -1

        return img, label


class SmartTabularDataset(Dataset):
    """
    Табличні дані (NLP/Регресія). Обгортка для Polars DataFrame.
    Використовується в HW 01 (Concrete) та HW 04 (Spam Detection).
    """
    def __init__(self, df: pl.DataFrame, target_col: str, feature_cols: List[str]):
        # Очищення від NaN рядків на льоту (Graceful Degradation)
        initial_len = len(df)
        self.df = df.drop_nulls(subset=feature_cols + [target_col])

        dropped = initial_len - len(self.df)
        if dropped > 0:
            logger.warning(f"⚠️ Видалено {dropped} рядків через пропущені значення (NaN).")

        # Конвертуємо Polars Series у NumPy, а потім у PyTorch Тензори (найшвидший шлях до GPU)
        self.features = torch.tensor(self.df.select(feature_cols).to_numpy(), dtype=torch.float32)

        # Перевіряємо тип цільової змінної для класифікації чи регресії
        target_dtype = torch.long if self.df[target_col].dtype in [pl.Int32, pl.Int64] else torch.float32
        self.targets = torch.tensor(self.df.select(target_col).to_numpy().squeeze(), dtype=target_dtype)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]
