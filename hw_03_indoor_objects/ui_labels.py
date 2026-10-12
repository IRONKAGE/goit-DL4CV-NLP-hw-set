# Мапа відповідності класів Indoor Object Detection до українського UI
# УВАГА: у YAML-конфігу класи лишаються англійськими (OpenCV/Ultralytics не вміють кирилицю в підписах),
# а українізація відбувається лише на рівні дашборду (таблиці, Plotly, малювання рамок через PIL + DejaVu).
import re

INDOOR_CLASSES_UA = {
    'door': 'Двері',
    'open door': 'Відчинені двері',
    'cabinet door': 'Дверцята шафи',
    'refrigerator door': 'Дверцята холодильника',
    'window': 'Вікно',
    'chair': 'Стілець',
    'table': 'Стіл',
    'cabinet': 'Шафа',
    'sofa/couch': 'Диван / Кушетка',
    'pillar': 'Стовп'
}

# Нормалізовані синоніми: у різних версіях наборах даних клас може називатись openedDoor / open_door / couch / pole тощо
_ALIASES = {
    'door': 'door',
    'opendoor': 'open door', 'openeddoor': 'open door',
    'cabinetdoor': 'cabinet door',
    'refrigeratordoor': 'refrigerator door', 'fridgedoor': 'refrigerator door',
    'window': 'window',
    'chair': 'chair',
    'table': 'table',
    'cabinet': 'cabinet',
    'sofacouch': 'sofa/couch', 'sofa': 'sofa/couch', 'couch': 'sofa/couch',
    'pillar': 'pillar', 'pole': 'pillar',
}


def canonical_class_key(class_name: str) -> str:
    """'openedDoor' / 'open_door' / 'Open Door' → 'open door'; невідомі назви повертаються як є (нижній регістр)."""
    flat = re.sub(r"[^a-z]", "", str(class_name).lower())
    return _ALIASES.get(flat, str(class_name).lower())


def get_indoor_label(class_name: str) -> str:
    """Повертає українську назву класу (з урахуванням синонімів); невідомі назви лишаються без змін."""
    return INDOOR_CLASSES_UA.get(canonical_class_key(class_name), class_name)


# Колонки results.csv та метрик Ultralytics → українська
METRIC_LABELS_UA = {
    'box_loss': 'Втрата рамок (box loss)',
    'cls_loss': 'Втрата класифікації (cls loss)',
    'dfl_loss': 'Втрата DFL (dfl loss)',
    'metrics/precision(B)': 'Precision (точність)',
    'metrics/recall(B)': 'Recall (повнота)',
    'metrics/mAP50(B)': 'mAP@0.5',
    'metrics/mAP50-95(B)': 'mAP@0.5:0.95',
}

SPLIT_LABELS_UA = {'train': 'Навчальна (train)', 'val': 'Валідаційна (val)', 'test': 'Тестова (test)'}


def get_metric_label(key: str) -> str:
    base = key.split('/', 1)[-1] if key.startswith(('train/', 'val/')) else key
    return METRIC_LABELS_UA.get(key, METRIC_LABELS_UA.get(base, key))


def get_split_label(split: str) -> str:
    return SPLIT_LABELS_UA.get(split, split)


# ГІПОТЕЗИ про причини плутанини пар класів (для розділу «Аналіз міжкласових помилок»).
# Це припущення на основі природи класів, а не доведені факти — у дашборді вони так і підписані.
_PAIR_REASONS = {
    frozenset({'door', 'open door'}): "один і той самий предмет у двох станах (закритий / відчинений): різниця лише в положенні полотна, тож рамки майже збігаються",
    frozenset({'cabinet', 'cabinet door'}): "вкладені об'єкти: дверцята — частина шафи, їхні рамки значно перекриваються (високий IoU)",
    frozenset({'cabinet door', 'refrigerator door'}): "візуально схожі прямокутні панелі з ручкою; відрізняє їх переважно контекст (кухня)",
    frozenset({'door', 'cabinet door'}): "схожа форма й ручка; різниця у розмірі та положенні відносно підлоги/стіни",
    frozenset({'door', 'refrigerator door'}): "схожа вертикальна панель; холодильник часто стоїть упритул до стіни",
    frozenset({'open door', 'cabinet door'}): "відчинені дверцята шафи й відчинені двері мають схожу перспективну форму",
    frozenset({'chair', 'table'}): "частково перекриваються у сцені (стільці під столом), оклюзія спотворює контур",
    frozenset({'chair', 'sofa/couch'}): "м'які меблі з близькою текстурою й формою; різняться лише масштабом",
    frozenset({'table', 'cabinet'}): "схожі прямокутні об'єкти біля стіни; можлива неоднозначність розмітки",
    frozenset({'window', 'door'}): "прямокутні світлі отвори; склінні двері можуть виглядати як вікна",
    frozenset({'pillar', 'door'}): "вузькі вертикальні об'єкти, що зливаються з фоном стіни",
}
_DEFAULT_REASON = ("гіпотеза не сформульована — варто переглянути приклади помилок: типові причини — "
                   "візуальна схожість, перекриття об'єктів, неоднозначна розмітка")


def get_confusion_reason(class_a: str, class_b: str) -> str:
    """Можлива причина плутанини пари класів (гіпотеза)."""
    key = frozenset({canonical_class_key(class_a), canonical_class_key(class_b)})
    return _PAIR_REASONS.get(key, _DEFAULT_REASON)
