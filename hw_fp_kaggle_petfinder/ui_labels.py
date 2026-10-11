# Мапа відповідності оригінальних класів Kaggle до українського UI
# (приватне змагання містить класи 1..4; ключ 0 лишено для сумісності з оригінальною розміткою PetFinder)
ADOPTION_SPEED_UA = {
    0: "В той же день (0 днів)",
    1: "Блискавично (1-7 днів)",
    2: "Швидко (8-30 днів)",
    3: "Помірно (31-90 днів)",
    4: "Повільно (>100 днів / Не прилаштовано)"
}

# Мапа стовпчиків (Features) для публічного та приватного наборів даних
COLUMN_NAMES_UA = {
    "PetID": "ID Тварини",
    "AdoptionSpeed": "Швидкість прилаштування",
    "Description": "Текстовий опис",
    "Type": "Тип (Собака/Кіт)",
    "Name": "Ім'я",
    "Age": "Вік (місяців)",
    "Breed1": "Порода (основна)",
    "Breed2": "Порода (змішана)",
    "Gender": "Стать",
    "Color1": "Колір 1",
    "Color2": "Колір 2",
    "Color3": "Колір 3",
    "MaturitySize": "Розмір (дорослої)",
    "FurLength": "Довжина шерсті",
    "Vaccinated": "Вакцинація",
    "Dewormed": "Від паразитів",
    "Sterilized": "Стерилізація",
    "Health": "Стан здоров'я",
    "Quantity": "Кількість у профілі",
    "Fee": "Благодійний внесок",
    "State": "Штат / Регіон",
    "RescuerID": "ID Волонтера/Притулку",
    "VideoAmt": "Кількість відео",
    "PhotoAmt": "Кількість фотографій"
}

def get_adoption_label(speed_class: int) -> str:
    """Повертає локалізовану категорію швидкості прилаштування."""
    return ADOPTION_SPEED_UA.get(speed_class, "Невідомо")

def get_column_label(col_name: str) -> str:
    """Повертає локалізовану назву стовпчика (ознаки)."""
    return COLUMN_NAMES_UA.get(col_name, col_name)


# ==============================================================================
# 🇺🇦 ЛОКАЛІЗАЦІЯ НАЗВ МОДЕЛЕЙ, БЛОКІВ ОЗНАК, ОЗНАК ТА СТОВПЧИКІВ ТАБЛИЦЬ (дашборд ФП)
# ==============================================================================

# Технічні ключі моделей (stacking.py / model.py) → зрозумілі назви для UI
MODEL_LABELS_UA = {
    "tfidf_ridge": "Ridge · Текст (TF-IDF)",
    "knn_tfidf": "kNN · Текст (TF-IDF)",
    "hgb_tabular": "Бустинг (HGB) · мета-ознаки",
    "hgb_all": "Бустинг (HGB) · усі ознаки",
    "keras_fusion": "Keras 3 · Late-Fusion",
}

# Блоки ознак (кеш *.npz + службові) → назви
BLOCK_LABELS_UA = {
    "img_tf_effv2b3": "Зображення (TF EfficientNetV2-B3)",
    "img_clip_b32": "Зображення (PyTorch CLIP ViT-B/32)",
    "txt_e5_base": "Текст (PyTorch multilingual-E5)",
    "txt_minilm": "Текст (PyTorch MiniLM)",
    "tfidf": "Текст (TF-IDF)",
    "tfidf_svd": "Текст (TF-IDF, SVD)",
    "tabular": "Мета-ознаки",
}

_ALGO_UA = {"ridge": "Ridge", "knn": "kNN", "hgb": "Бустинг (HGB)"}

# Ознаки, які будує data_adapter.build_tabular()
FEATURE_LABELS_UA = {
    "n_images": "Кількість фото",
    "txt_chars": "Символів в описі",
    "txt_words": "Слів в описі",
    "txt_sents": "Речень в описі",
    "txt_excl": "Знаків оклику",
    "txt_upper": "Великих літер",
    "txt_digits": "Цифр в описі",
    "txt_nonascii": "Не-ASCII символів",
    "txt_uniq_ratio": "Частка унікальних слів",
    "txt_empty": "Порожній опис",
}

# Стовпчики звітів
REPORT_COLUMNS_UA = {
    "модель": "Модель",
    "QWK (округл.)": "QWK (округлення)",
    "QWK (пороги)": "QWK (оптим. пороги)",
}
NEIGHBOR_COLUMNS_UA = {
    "блок": "Блок ознак",
    "поріг cos": "Поріг косинусної схожості",
    "train із сусідом, %": "Train із сусідом, %",
    "мітка сусіда збігається, %": "Мітка сусіда збігається, %",
    "test із сусідом у train, %": "Test із сусідом у train, %",
}


def get_block_label(name: str) -> str:
    """Назва блоку ознак (img_tf_effv2b3 → «Зображення · TF EfficientNetV2-B3»)."""
    return BLOCK_LABELS_UA.get(name, name)


def get_model_label(name: str) -> str:
    """Назва моделі; для ridge_/knn_ + блок складається автоматично («Ridge · Зображення · ...»)."""
    if name in MODEL_LABELS_UA:
        return MODEL_LABELS_UA[name]
    algo, _, rest = name.partition("_")
    if algo in _ALGO_UA and rest:
        return f"{_ALGO_UA[algo]} · {get_block_label(rest)}"
    return name


def get_feature_label(name: str) -> str:
    """Назва ознаки; ознаки з суфіксом __freq позначаються як «(частота)»."""
    if name.endswith("__freq"):
        return f"{get_column_label(name[:-6])} (частота)"
    return FEATURE_LABELS_UA.get(name, get_column_label(name))


def localize_report(df):
    """Копія таблиці model_report() з українськими назвами моделей і стовпчиків (RMSE лишається терміном)."""
    out = df.copy()
    out["модель"] = out["модель"].map(get_model_label)
    return out.rename(columns=REPORT_COLUMNS_UA)


def localize_neighbors(df):
    out = df.copy()
    out["блок"] = out["блок"].map(get_block_label)
    return out.rename(columns=NEIGHBOR_COLUMNS_UA)
