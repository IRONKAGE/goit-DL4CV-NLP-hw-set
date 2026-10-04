# ==============================================================================
# 🧠 CORE PACKAGE INITIALIZATION (Deep Learning Infrastructure)
# ==============================================================================
"""
Ядро інфраструктури проекту (CV & NLP).
Об'єднує маршрутизацію обладнання, завантаження даних та універсальний Trainer.
Імпортування з цього модуля гарантує єдину точку доступу до всіх інструментів.
"""

import os
import logging
import warnings
from dotenv import load_dotenv

# ------------------------------------------------------------------------------
# 1. ЗАВАНТАЖЕННЯ СЕКРЕТІВ (.env)
# ------------------------------------------------------------------------------
# override=True гарантує, що локальний .env має пріоритет
load_dotenv(override=True)

# ------------------------------------------------------------------------------
# 2. СИСТЕМА ЛОГУВАННЯ ТА ПОПЕРЕДЖЕНЬ
# ------------------------------------------------------------------------------
# Глушимо набридливі UserWarning (наприклад, від трансформацій PyTorch Torchvision)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Єдиний стандарт логування для всього проекту
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("core")

# Вимикаємо зайвий спам від сторонніх бібліотек
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# ------------------------------------------------------------------------------
# 3. ФАСАД API (Package API Export)
# ------------------------------------------------------------------------------
try:
    # Імпортуємо з наших нових файлів
    from .hardware import get_hardware_config, clear_vram, log_system_info, set_global_seed
    from .datasets import read_tabular_data_safe, read_image_safe, SmartImageDataset, SmartTabularDataset
    from .trainer import BaseTrainer, EarlyStopping
    from .etl_core import SecureDownloader, smart_read_csv

    # __all__ жорстко контролює, що буде доступно при `from core import *`
    __all__ = [
        "get_hardware_config",
        "clear_vram",
        "log_system_info",
        "set_global_seed",
        "read_tabular_data_safe",
        "read_image_safe",
        "SmartImageDataset",
        "SmartTabularDataset",
        "BaseTrainer",
        "EarlyStopping",
        "SecureDownloader",
        "smart_read_csv",
        "logger"
    ]

except ImportError as e:
    logger.warning(f"Модулі ядра частково недоступні (можливо, відсутні залежності): {e}")
    __all__ = ["logger"]
