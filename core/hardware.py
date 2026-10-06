# ==============================================================================
# 🛠️ ML UTILITIES (Hardware Routing, Memory Management & Reproducibility)
# ==============================================================================
"""
Утиліти машинного навчання рівня Enterprise (Оновлення: Жовтень 2026).
Відповідають за інтелектуальну маршрутизацію обчислень, очищення відеопам'яті (VRAM)
та строгу фіксацію випадковості для 100% відтворюваності експериментів.
"""

import os
import gc
import random
import platform
import logging
import numpy as np

# Базове налаштування логера для інфраструктурного ядра
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("core.hardware")

# ------------------------------------------------------------------------------
# 1. ЗАБЕЗПЕЧЕННЯ БАЗОВОЇ ВІДТВОРЮВАНОСТІ (Base Reproducibility)
# ------------------------------------------------------------------------------
def set_global_seed(seed: int = None) -> None:
    """
    Жорстко фіксує базовий seed для Python та NumPy.
    Гарантує, що розщеплення даних та ініціалізація ваг будуть ідентичними при кожному запуску.
    (PyTorch фіксується окремо в апаратному маршрутизаторі).
    """
    if seed is None:
        seed = int(os.getenv("GLOBAL_SEED", 42))

    # 1. Фіксація стандартного модуля Python
    random.seed(seed)

    # 2. Фіксація хешування вбудованих структур (dict, set)
    # Запобігає випадковому порядку ітерації словників у Python 3.3+
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 3. Фіксація NumPy (використовується в Pandas, Polars та Scikit-Learn)
    np.random.seed(seed)

    # 4. Фіксація глобального PyTorch (страховка)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass

    # 5. Фіксація TensorFlow (Критично для tf.data)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    except ImportError:
        pass

    logger.info(f"🌱 Базовий seed зафіксовано: {seed} (Python, NumPy, Torch, TF)")

# ------------------------------------------------------------------------------
# 2. УЛЬТИМАТИВНИЙ МАРШРУТИЗАТОР ОБЧИСЛЮВАЛЬНИХ ПОТУЖНОСТЕЙ
# ------------------------------------------------------------------------------
def get_hardware_config(global_seed: int = None):
    """
    Ультимативний апаратний маршрутизатор PyTorch 2.14+.
    Автоматично шукає GPU (CUDA/ROCm/MPS/XPU). Якщо драйвери відсутні — провалюється на CPU.
    Підтримує примусовий Override через `.env` (ML_BACKEND=cpu).
    """
    if global_seed is None:
        global_seed = int(os.getenv("GLOBAL_SEED", 42))

    try:
        import torch
    except ImportError:
        logger.warning("⚠️ PyTorch не встановлено. Повертаю базовий 'cpu'.")
        return "cpu", "CPU (No PyTorch)"

    # 🎛️ 0. ПРИМУСОВИЙ OVERRIDE (Для Docker та ізольованих CI/CD пайплайнів)
    env_backend = os.getenv("ML_BACKEND", "").lower().strip()
    if env_backend == "cpu":
        device = torch.device("cpu")
        device_ui_name = "CPU (Примусово через .env)"
        torch.manual_seed(global_seed)
        logger.info(f"⚙️ Ініціалізовано: {device_ui_name}")
        return device, device_ui_name

    # 🥇 1. NVIDIA (CUDA) або AMD (ROCm) - High-End робочі станції (напр. RTX PRO 6000)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Архітектурна перевірка: чи зібрано PyTorch під AMD HIP/ROCm?
        if getattr(torch.version, "hip", None):
            device_ui_name = "ROCm (AMD Instinct / Radeon)"
        else:
            device_ui_name = f"CUDA ({torch.cuda.get_device_name(0)})"

        torch.cuda.manual_seed_all(global_seed)
        # Вимикаємо евристичну оптимізацію CuDNN для досягнення 100% детермінованості
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 🥈 2. Apple Metal API (MPS) - Універсальна підтримка M-серії та AMD
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")

        # Точна архітектурна ідентифікація можливостей системи (M-series vs AMD Radeon)
        if platform.machine() == "arm64":
            device_ui_name = "MPS (Apple Silicon M-Серія | Підтримує Docker GPU Passthrough)"
        else:
            device_ui_name = "MPS (AMD Radeon на macOS | Docker GPU не підтримується апаратно)"

        torch.mps.manual_seed(global_seed)

    # 🥉 3. Intel (Data Center Max Series / Arc)
    elif hasattr(torch, "xpu") and getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
        device = torch.device("xpu")
        device_ui_name = "XPU (Intel GPU / AI Accelerators)"
        torch.xpu.manual_seed_all(global_seed)

    # 🌐 4. Huawei Ascend (Enterprise Cloud NPU)
    elif hasattr(torch, "npu") and getattr(torch.npu, "is_available", lambda: False)():
        device = torch.device("npu")
        device_ui_name = "NPU (Huawei Ascend Accelerator)"
        torch.npu.manual_seed_all(global_seed)

    # 🧠 5. Meta MTIA (Специфічні акселератори Llama-інфраструктури)
    elif hasattr(torch, "mtia") and getattr(torch.mtia, "is_available", lambda: False)():
        device = torch.device("mtia")
        device_ui_name = "MTIA (Meta Training & Inference Accelerator)"
        torch.mtia.manual_seed_all(global_seed)

    # 🐢 6. АВТОМАТИЧНИЙ ФОЛБЕК НА CPU (Якщо драйвери відсутні або пошкоджені)
    else:
        device = torch.device("cpu")
        arch = platform.machine()
        device_ui_name = f"CPU ({arch})"
        torch.manual_seed(global_seed)

    logger.info(f"🚀 Апаратне прискорення: {device_ui_name}")
    return device, device_ui_name

# ------------------------------------------------------------------------------
# 3. МЕНЕДЖЕР ПАМ'ЯТІ (VRAM / RAM Garbage Collector)
# ------------------------------------------------------------------------------
def clear_vram(device) -> None:
    """
    Архітектурне звільнення відеопам'яті (VRAM) та оперативної пам'яті (RAM).
    Запобігає помилкам Out-Of-Memory (OOM) між епохами навчання або GridSearch ітераціями.
    """
    gc.collect()

    if device is None or type(device) == str:
        return

    try:
        import torch
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "xpu":
            torch.xpu.empty_cache()

        logger.debug(f"🧹 VRAM ({device.type.upper()}) та оперативну пам'ять успішно очищено")
    except Exception as e:
        logger.debug(f"🧹 Очищення VRAM пропущено: {e}")

# ------------------------------------------------------------------------------
# 4. ІНСПЕКТОР СЕРЕДОВИЩА (Environment Profiler)
# ------------------------------------------------------------------------------
def log_system_info() -> None:
    """
    Виводить короткий аудит системи під час старту пайплайну.
    Критично важливо для дебагінгу Docker-контейнерів, щоб розуміти, на якому ядрі працює код.
    """
    os_name = platform.system()
    os_release = platform.release()
    py_version = platform.python_version()

    logger.info(f"🖥️  Система: {os_name} {os_release} | Python: {py_version}")

# ------------------------------------------------------------------------------
# 5. МАРШРУТИЗАТОР ДЛЯ ДЕРЕВ РІШЕНЬ (XGBoost / LightGBM)
# ------------------------------------------------------------------------------
def get_boosting_kwargs(device):
    """
    Генерує оптимальні параметри апаратного прискорення для XGBoost (>=3.3) та LightGBM (>=4.6).
    Відповідає за безпечний fallback на CPU для архітектур без прямої підтримки OpenCL/CUDA.
    """
    device_type = getattr(device, "type", "cpu")

    # Базові налаштування (завжди навантажуємо всі ядра процесора через n_jobs=-1)
    xgb_kwargs = {"tree_method": "hist", "n_jobs": -1}
    lgbm_kwargs = {"n_jobs": -1, "verbose": -1}

    # 1. NVIDIA (CUDA) або AMD (ROCm)
    # XGBoost ідеально масштабується на CUDA-ядрах
    if device_type == "cuda":
        xgb_kwargs["device"] = "cuda"
        lgbm_kwargs["device_type"] = "gpu"

    # 2. Intel GPU (SYCL / XPU)
    # З версії XGBoost 2.0+ додано нативну підтримку SYCL для Intel Arc/Max
    elif device_type == "xpu":
        xgb_kwargs["device"] = "sycl"
        lgbm_kwargs["device_type"] = "gpu"

    # 3. Apple Metal (MPS) або чистий CPU
    # На macOS безпечніше рахувати дерева на потужних ядрах CPU (M5 Ultra),
    # оскільки MPS оптимізовано під тензорні графи PyTorch, а не під градієнтний бустинг.
    else:
        xgb_kwargs["device"] = "cpu"
        lgbm_kwargs["device_type"] = "cpu"

    logger.info(f"🌲 Бустинги (XGB/LGBM) змаршрутизовано на: {xgb_kwargs['device'].upper()} (n_jobs={xgb_kwargs['n_jobs']})")

    return xgb_kwargs, lgbm_kwargs

if __name__ == "__main__":
    # Демонстраційний запуск для перевірки конфігурації
    print("\n" + "="*60)
    log_system_info()
    set_global_seed()
    dev, name = get_hardware_config()
    clear_vram(dev)
    get_boosting_kwargs(dev)
    print("="*60 + "\n")
