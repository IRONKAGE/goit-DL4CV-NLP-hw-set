# Мапа відповідності оригінальних класів Kaggle до українського UI
INTEL_CLASSES_UA = {
    "buildings": "Будівлі",
    "forest": "Ліс",
    "glacier": "Льодовик",
    "mountain": "Гори",
    "sea": "Море",
    "street": "Вулиця"
}

def get_ua_label(eng_label: str) -> str:
    """Повертає локалізовану назву класу."""
    return INTEL_CLASSES_UA.get(eng_label, eng_label)
