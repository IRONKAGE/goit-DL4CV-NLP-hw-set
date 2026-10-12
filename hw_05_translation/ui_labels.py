# Локалізація інтерфейсу для ДЗ №5
UI_LABELS = {
    "title": "Машинний Переклад (Seq2Seq Attention)",
    "src_lang": "Англійська",
    "trg_lang": "Грецька",
    "src_col": "Англійська (Оригінал)",
    "trg_true_col": "Грецька (Еталон)",
    "trg_pred_col": "Грецька (Прогноз моделі)",
    "table_tab": "Приклади перекладів",
    "attn_tab": "Матриця Уваги"
}

def get_ui_label(key: str) -> str:
    return UI_LABELS.get(key, key)
