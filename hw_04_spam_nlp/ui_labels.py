# Мапа відповідності оригінальних класів Kaggle до українського UI
SPAM_CLASSES_UA = {
    0: "Легітимний лист (Ham)",
    1: "Спам (Spam)"
}

def get_spam_label(label_int: int) -> str:
    """Повертає локалізовану назву класу для графіків та звітів."""
    return SPAM_CLASSES_UA.get(label_int, f"Невідомий клас ({label_int})")
