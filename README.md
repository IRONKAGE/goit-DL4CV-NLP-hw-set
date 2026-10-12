# goit-dl4cv-nlp-hw-set

## 🧠 Deep Learning, Computer Vision & NLP Portfolio

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) [![Python 3.12](https://img.shields.io/badge/Python-3.12%2B-blue.svg?logo=Python)](https://docs.python.org/3.12/) [![Manager: uv](https://img.shields.io/badge/Manager-uv-pink.svg?logo=UV)](https://docs.astral.sh/uv/) [![Makefile](https://img.shields.io/badge/Makefile-grey.svg?logo=GNU)](https://www.gnu.org/software/make/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)

**Студент:** Oleh Hatsenko | **Курс:** GoIT Neoversity Master of Science in AI/ML

Це репозиторій-портфоліо глибоких нейромережевих архітектур у сферах Computer Vision (CV), Natural Language Processing (NLP) та MLOps.

Проект побудований на принципах **Clean Architecture**, апаратно-агностичного виконання (автоматична маршрутизація на MPS/CUDA/CPU) та сучасних індустріальних MLOps стандартів станом на вересень 2026 року. Розвідувальний аналіз даних (EDA)та тренування інтегровано через реактивне середовище Marimo.

За основу взяті ***Домашні Роботи і Фінальний Проект*** з курсу **Deep Learning (CV & NLP)** та масштабовані до рівня production-ready монорепозиторію.

---

## 🛠 Технологічний Стек (Bleeding Edge)

У проекті використовуються строго зафіксовані версії бібліотек (PEP 621), які гарантують стабільність виконання на ефемерних хмарних сесіях та локальних машинах з архітектурою Apple Silicon / x64:

### 🧠 Deep Learning & AI Frameworks

- [![torch 2.14.1](https://img.shields.io/badge/torch-2.14.1-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pypi.org/project/torch/2.14.1/) 🔸 Фундаментальний рушій обчислень. Завдяки модулю `core.hardware` автоматично утилізує Apple Metal Performance Shaders (MPS) або NVIDIA CUDA з fallback на CPU. Для збереження стабільності на **Legacy macOS (Intel x64 + AMD)** завдяки PEP 508 маркерам автоматично розгортається: [![torch 2.2.2](https://img.shields.io/badge/torch-2.2.2-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pypi.org/project/torch/2.2.2/)
- [![transformers 5.0.0](https://img.shields.io/badge/transformers-5.0.0-FFD21E.svg?logo=huggingface&logoColor=black)](https://huggingface.co/docs/transformers/index) 🔸 SOTA-моделі (CLIP, E5, MiniLM) сумісні з PyTorch для побудови потужних NLP та Vision екстракторів.
- [![torchvision 0.29.1](https://img.shields.io/badge/torchvision-0.29.1-150458.svg?logo=pytorch&logoColor=white)](https://pypi.org/project/torchvision/) 🔸 Робота з Computer Vision, трансформації даних та Transfer Learning (ResNet).
- [![ultralytics 8.4.172](https://img.shields.io/badge/ultralytics-8.4.172-00A9E0.svg?logo=yolo&logoColor=white)](https://pypi.org/project/ultralytics/) 🔸 SOTA-фреймворк для детекції об'єктів та сегментації (YOLOv8/YOLOv9/YOLO26).
- [![gensim 4.4.0](https://img.shields.io/badge/gensim-4.4.0-0194E2.svg?logo=shadow&logoColor=white)](https://pypi.org/project/gensim/) 🔸 Векторне моделювання семантики текстів (попередньо навчені Word2Vec/GloVe ембедінги).

### 🎨 EDA & UI Frameworks

- [![marimo 0.25.1](https://img.shields.io/badge/marimo-0.25.1-F0EDF9.svg?logo=loop&logoColor=white)](https://pypi.org/project/marimo/) 🔸 Реактивне Python-середовище нового покоління для створення інтерактивних дашбордів розвідувального аналізу та ML-пайплайнів.
- [![svelte 5.x](https://img.shields.io/badge/svelte-5.x-FF3E00.svg?logo=svelte&logoColor=white)](https://svelte.dev/) 🔸 Компілятор фронтенду (Control Panel) для візуалізації метрик моделі в реальному часі.
- [![biome 2.x](https://img.shields.io/badge/biome-2.x-60a5fa.svg?logo=biome&logoColor=white)](https://biomejs.dev/) 🔸 Надшвидкий Rust-тулчейн (Linter + Formatter) для екосистеми веб-фронтенду.

### 💾 Data Engineering

- [![polars 1.44.2](https://img.shields.io/badge/polars-1.44.2-FFD43B.svg?logo=polars&logoColor=white)](https://pypi.org/project/polars/) 🔸 Надшвидкий рушій обробки табличних даних (Rust) для модулів `SmartReader`.
- [![pillow 12.3.0](https://img.shields.io/badge/pillow-12.3.0-31A8FF.svg?logo=python&logoColor=white)](https://pypi.org/project/pillow/) 🔸 Високопродуктивна бібліотека для піксельних маніпуляцій та завантаження зображень.
- [![scikit-learn 1.9.1](https://img.shields.io/badge/scikit--learn-1.9.1-F7931E.svg?logo=scikit-learn&logoColor=white)](https://pypi.org/project/scikit-learn/) 🔸 Класичні метрики, векторизатори (TF-IDF, BoW), OOF-Stacking, kNN та спліти даних.

### ⚙️ MLOps, Tracking & Serving

- [![mlflow 3.16.1](https://img.shields.io/badge/mlflow-3.16.1-0194E2.svg?logo=mlflow&logoColor=white)](https://pypi.org/project/mlflow/) 🔸 Інфраструктура логування експериментів, параметрів та серіалізації артефактів всередині `BaseTrainer`.
- [![fastapi 0.142.2](https://img.shields.io/badge/fastapi-0.142.2-009688.svg?logo=fastapi&logoColor=white)](https://pypi.org/project/fastapi/) 🔸 Асинхронний бекенд мікросервісів для забезпечення Model Serving.
- [![docker 26.0](https://img.shields.io/badge/docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/) 🔸 Контейнеризація API для ізольованого Kubernetes-розгортання (`deploy/Dockerfile`).

---

## 📂 Архітектура та проекти

```text
goit-DL4CV-NLP-hw-set/              
│
├── .venv/                      # 🐍 Ізольоване віртуальне середовище (генерується автоматично через uv)
│
├── core/                       # ⚙️ MLOps Ядро (Спільна інфраструктура для всіх нейромереж)
│   ├── __init__.py             # 🧠 Ініціалізація ядра, зчитування .env, налаштування логерів та єдиний фасад API
│   ├── hardware.py             # 🔌 Ультимативний маршрутизатор заліза (MPS/CUDA/CPU) з Fallback-захистом
│   ├── datasets.py             # 🛡️ Smart Reader: Патерн Graceful Degradation (стійкість до битих файлів)
│   ├── etl_core.py             # 📦 SecureDownloader: 5-рівнева система завантаження (Kaggle/GDrive) із перевіркою SHA-256
│   └── trainer.py              # 🏎️ Універсальний BaseTrainer (Early Stopping, MLflow, Checkpointing)
│
├── data/                       # 💾 Директорія сирих даних (ігнорується Git)
│   └── .gitkeep                # ⚠️ Маркер для збереження директорії у системі контролю версій Git
│
├── deploy/                     # 🚀 Інфраструктура розгортання
│   └── Dockerfile              # 🐳 Легковаговий маніфест збірки FastAPI-інференсу для Kubernetes
│
├── control_panel/              # 🕹️ UI-панель керування моделями (Svelte 5 + Bun + FastAPI)
│
├── notebooks_eda/              # 📊 Інтерактивні Marimo-скрипти розвідувального аналізу (EDA)
│   ├── hw_01_eda.py            # 🏗️ Аналіз рецептур бетону (Кореляційні матриці, Слайдери віку)
│   ├── hw_02_eda.py            # 🖼️ Аналіз зображень пейзажів Intel (Розподіл класів, Зразки)
│   ├── hw_04_eda.py            # 📧 Аналіз дисбалансу, порівняння NLP-векторизаторів та інтерактивне Demo
│   ├── hw_05_eda.py            # 🌍 Аналіз паралельного корпусу (Source/Target Sentence Length)
│   └── hw_fp_eda.py            # 🐶 Мультимодальний пайплайн (Tri-Framework: TF ⟷ PyTorch ⟷ Keras 3)
│
├── hw_01_concrete/             # 🏗️ Проект 1: Concrete Strength Prediction (Регресія)
│   ├── data_adapter.py         # 🗃️ Конвертер Polars DataFrame у PyTorch DataLoader зі StandardScaler
│   ├── model.py                # 🧠 Архітектура багатошарового перцептрона (Feed-Forward NN) з Dropout
│   └── ui_labels.py            # 🌍 Локалізація наборів даних для UI
│
├── hw_02_intel_cv/             # 🖼️ Проект 2: Intel Image Classification (Computer Vision)
│   ├── data_adapter.py         # 🗃️ DataLoader з аугментаціями (RandomResizedCrop, ColorJitter) та Train/Val Split
│   ├── model.py                # 🧠 Transfer Learning на базі ResNet18 (заморожування вагів, нова FC-голова)
│   └── ui_labels.py            # 🌍 Локалізація наборів даних для UI
│
├── hw_04_spam_nlp/             # 📧 Проект 4: Spam Detection (NLP Comparative Study)
│   ├── data_adapter.py         # 🗃️ Векторизація тексту (BoW, TF-IDF) та завантаження ембедінгів GloVe Word2Vec
│   ├── nlp_core.py             # 🧮 Інкапсулятор для роботи з Gensim та Mean Pooling
│   ├── model.py                # 🧠 Універсальний класифікатор (MLP) для порівняння трьох NLP-підходів
│   ├── train.py                # 🚂 Тренування ансамблю моделей, порівняння ROC-AUC, Precision, F1 та авто-висновки
│   └── ui_labels.py            # 🌍 Локалізація наборів даних для UI
│
├── hw_05_translation/          # 🌍 Проект 5: Machine Translation (Seq2Seq Attention)
│   ├── data_adapter.py         # 🗃️ Генерація парних словників та паддінг-тензорів (Teacher Forcing Shift)
│   ├── model.py                # 🧠 Архітектура RNN Encoder-Decoder з механізмом уваги (Bahdanau Attention)
│   ├── train.py                # 🚂 Навчання, інтерактивний переклад та генерація Heatmap-матриці уваги
│   └── ui_labels.py            # 🌍 Локалізація наборів даних для UI
│
├── hw_fp_kaggle_petfinder/     # 🐶🐱 Фінальний Проект: Multimodal Ensemble (Kaggle)
│   ├── data_adapter.py         # 🗃️ Pure Pandas/Numpy лоадер (схема-агностичне читання, генерація мета-ознак)
│   ├── extractors.py           # 🏭 Ізольовані підпроцеси екстракції ознак (TF EfficientNetV2, PyTorch CLIP/Transformers)
│   ├── model.py                # 🧠 Keras 3 Late-Fusion мультимодальна мережа (OOF тренування)
│   ├── stacking.py             # 📐 Level-1 моделі (Ridge, HGB, kNN), OOF-стекінг, Honest Threshold Tuning (QWK)
│   ├── timing.py               # ⏱️ Утиліти для форматування таймерів виконання
│   └── ui_labels.py            # 🌍 Локалізація наборів даних для UI
│
├── models/                     # 📦 Директорія згенерованих артефактів (матриці плутанини, ROC-криві, best_model.pt)
│   └── .gitkeep                # ⚠️ Маркер для збереження директорії у системі контролю версій Git
│
├── mlruns/                     # 📊 База даних MLflow (історія Loss, гіперпараметрів та вагів моделей)
│   └── .gitkeep                # ⚠️ Маркер для збереження директорії у системі контролю версій Git
│
├── etl_pipeline.log            # 📝 Логи виконання процесів екстракції та трансформації даних
├── .editorconfig               # 📝 Стандарти форматування коду для командної розробки
├── .env                        # 🔐 Локальні змінні середовища (креденшнли, конфіги — ігнорується Git)
├── .env.example                # 📄 Безпечний шаблон змінних середовища для розгортання на нових машинах
├── .gitignore                  # 🚫 Правила ігнорування файлів (кеші, бази SQLite, .joblib моделі, CSV-наборами даних)
├── LICENSE                     # ⚖️ Ліцензія відкритого вихідного коду (MIT)
├── Makefile                    # 🪄 MLOps Оркестратор: єдина точка входу (Zero-Config)
├── pyproject.toml              # 📝 Сучасний стандарт опису метаданих та залежностей (PEP 621)
├── README.md                   # 📖 Головна архітектурна документація проекту
└── uv.lock                     # 🔒 Суворий лок-файл крос-платформних залежностей (надійніший за requirements.txt)
```

- **MLOps Core (`/core`):** Архітектурне ядро. Включає автоматичну детекцію обладнання (`hardware.py`), яка змушує нейромережі працювати на максимумі як на RTX Pro 6000, так і на Apple M-Серії. Впроваджено `SmartReader` з патерном Graceful Degradation для безперебійної обробки даних та універсальний `BaseTrainer` для стандартизації логування в MLflow та бере на себе рутину з Early Stopping.
- **Concrete Strength (`HW 01`):** Регресійна задача на табличних даних. Розробка глибокого багатошарового перцептрона (MLP) з використанням `StandardScaler` для прогнозування міцності бетону (оцінка за MSE, MAE та R²).
- **Intel Image Classification (`HW 02`):** Задачі класифікації зображень з використанням комп'ютерного зору. Застосування техніки Transfer Learning на базі попередньо навченої моделі `ResNet18` з налаштуванням аугментацій (RandomResizedCrop, ColorJitter).
- **Spam Detection (`HW 04`):** Порівняльне дослідження (Comparative Study) методів обробки природної мови. Еволюція від базових частотних мішків слів (BoW та TF-IDF) до попередньо навчених семантичних векторів `GloVe Word2Vec` з інтерактивним інференсом.
- **Machine Translation (`HW 05`):** Занурення у складні генеративні архітектури NLP. Побудова моделі `Seq2Seq` (RNN Encoder-Decoder) з механізмом уваги (Bahdanau Attention) та візуалізацією Attention Heatmaps (матриць уваги) під час перекладу.
- **Multimodal Fusion (`FP`):** Вершина інженерної думки курсу (Kaggle Challenge). Високопродуктивна *Tri-Framework* архітектура: `TensorFlow` (EfficientNetV2), `PyTorch` (CLIP, E5, MiniLM) та `Keras 3` (Late Fusion). Використовує ізольовані підпроцеси для екстракції ознак, 5-фолдовий OOF (Out-Of-Fold) стекінг з kNN/Ridge/HGB, та оптимізацію Quadratic Weighted Kappa (QWK) через Nelder-Mead з чесними порогами (Honest Thresholding).

## 🪬 Швидкий старт

Проект має **Zero-Config Boot**. Достатньо мати встановлений Python та утиліту GNU Make. 

Щоб побачити повний список команд інфраструктури, виконайте у терміналі:

```bash
make
```

> **Приклад типового пайплайну:**
> 1. `make setup` (створить середовище та завантажить PyTorch/YOLO).
> 2. `make eda-hw` (відкриє Marimo-дашборд для аналізу даних).
> 3. `make mlflow` (відкриє дашборд з графіками навчання експерименту).

## 💝 Контакти

Якщо цей репозиторій здався вам цікавим, і ви шукаєте інженера, який здатен побудувати надійний, об'єктно-орієнтований ML/DL пайплайн, готовий до хмарного розгортання — буду радий поспілкуватися!

- **LinkedIn:** [IRONKAGE](https://www.linkedin.com/in/ironkage)
