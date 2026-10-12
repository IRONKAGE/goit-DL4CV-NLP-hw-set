"""
🗃️ DATA LAYER (HW 03: Indoor Objects Detection, YOLO-формат).

Що робить:
  * ensure_dataset()     — завантаження через core.etl_core.SecureDownloader (Kaggle) лише якщо даних ще немає;
  * discover_splits()    — знаходить train / val(valid) / test незалежно від глибини вкладеності архіву;
  * scan_split()/audit   — перевірка відповідності зображень ↔ файлів розмітки, коректності рядків YOLO;
  * prepare_dataset()    — генерує YAML з АБСОЛЮТНИМИ шляхами та іменами класів, взятими з набору даних (якщо є) або з резервного списку ТЗ (якщо немає власного YAML);
                           (порядок має збігатися з розміткою — лише тоді YAML валідний);
  * prepare_yolo_dataset() — сумісна обгортка (повертає шлях до YAML, як у попередній версії).
"""
import os
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger("hw_03.data_adapter")

# АБСОЛЮТНА НАДІЙНІСТЬ ШЛЯХІВ: знаходимо корінь проекту
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "hw_03_indoor")
KAGGLE_DATASET = "sabahesaraki/indoor-object-detection"
YAML_NAME = "indoor_objects.yaml"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Резервний список з умови ТЗ (використовується ЛИШЕ якщо в наборі даних немає власного yaml з іменами)
FALLBACK_CLASSES = [
    'door', 'open door', 'cabinet door', 'refrigerator door', 'window',
    'chair', 'table', 'cabinet', 'sofa/couch', 'pillar'
]

_SPLIT_ALIASES = {
    "train": "train", "training": "train",
    "val": "val", "valid": "val", "validation": "val",
    "test": "test", "testing": "test",
}


# ------------------------------------------------------------------------------
# 1. ПОШУК ВИБІРОК
# ------------------------------------------------------------------------------
def discover_splits(data_dir: str = DATA_DIR) -> Dict[str, Tuple[Path, Path]]:
    """
    Повертає {'train': (images_dir, labels_dir), 'val': ..., 'test': ...}.
    Підтримує обидва стандартні макети YOLO: <split>/images + <split>/labels  та  images/<split> + labels/<split>.
    """
    root = Path(data_dir)
    found: Dict[str, Tuple[Path, Path]] = {}
    if not root.exists():
        return found

    # Макет A: <split>/images та <split>/labels
    for img_dir in sorted(root.rglob("images"), key=lambda p: len(p.parts)):
        if not img_dir.is_dir():
            continue
        split = _SPLIT_ALIASES.get(img_dir.parent.name.lower())
        lbl_dir = img_dir.parent / "labels"
        if split and split not in found and lbl_dir.is_dir():
            found[split] = (img_dir, lbl_dir)

    # Макет B: images/<split> та labels/<split>
    for img_root in sorted(root.rglob("images"), key=lambda p: len(p.parts)):
        if not img_root.is_dir():
            continue
        for sub in img_root.iterdir():
            split = _SPLIT_ALIASES.get(sub.name.lower())
            lbl_dir = img_root.parent / "labels" / sub.name
            if sub.is_dir() and split and split not in found and lbl_dir.is_dir():
                found[split] = (sub, lbl_dir)
    return found


def ensure_dataset(data_dir: str = DATA_DIR) -> Dict[str, Tuple[Path, Path]]:
    """Завантажує й розпаковує набір даних, лише якщо train/val ще не знайдено на диску."""
    os.makedirs(data_dir, exist_ok=True)
    splits = discover_splits(data_dir)
    if "train" in splits and ("val" in splits or "test" in splits):
        logger.info("🔋 Набір даних уже на диску — завантаження пропущено.")
        return splits

    from core.etl_core import SecureDownloader  # ледачий імпорт: core тягне torch/mlflow

    try:
        downloader = SecureDownloader(
            dataset_path=KAGGLE_DATASET,
            data_dir=data_dir,
            zip_name="indoor_objects.zip",
        )
        downloader.download(target_filename="indoor_objects.zip")
        downloader.extract_atomically(
            target_extensions=('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.txt', '.yaml', '.yml'),
            expected_filename=".extracted_ok",          # маркер ідемпотентності (створюємо нижче)
        )
        Path(data_dir, ".extracted_ok").touch()
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"Не вдалося завантажити '{KAGGLE_DATASET}': {e}\n"
            "Перевір: KAGGLE_USERNAME/KAGGLE_KEY у .env, прийняті умови набору даних, мережу."
        ) from e

    splits = discover_splits(data_dir)
    if "train" not in splits:
        raise RuntimeError(f"Після розпакування не знайдено <split>/images + <split>/labels у {data_dir}")
    return splits


# ------------------------------------------------------------------------------
# 2. АУДИТ: відповідність зображень і розмітки, коректність YOLO-рядків
# ------------------------------------------------------------------------------
@dataclass
class SplitStats:
    split: str
    images_dir: Path
    labels_dir: Path
    n_images: int = 0
    n_label_files: int = 0
    images_without_labels: int = 0     # у YOLO це допустимі «фонові» зображення, але варто знати їх частку
    labels_without_images: int = 0     # реальна проблема: розмітка без картинки
    n_boxes: int = 0
    bad_lines: int = 0                 # рядки, які не парсяться або мають <5 полів
    polygon_lines: int = 0             # рядки з >5 полів (сегментація, а не рамки)
    out_of_range: int = 0              # координати поза [0, 1]
    class_counts: Counter = field(default_factory=Counter)
    sizes: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))   # (w, h) нормалізовані
    image_paths: List[Path] = field(default_factory=list)


def scan_split(split: str, img_dir: Path, lbl_dir: Path) -> SplitStats:
    st = SplitStats(split=split, images_dir=img_dir, labels_dir=lbl_dir)
    images = {p.stem: p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in IMG_EXT}
    labels = {p.stem: p for p in sorted(lbl_dir.glob("*.txt"))}
    st.n_images, st.n_label_files = len(images), len(labels)
    st.images_without_labels = len(set(images) - set(labels))
    st.labels_without_images = len(set(labels) - set(images))
    st.image_paths = list(images.values())

    sizes = []
    for lp in labels.values():
        for line in lp.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if not parts:
                continue
            if len(parts) < 5:
                st.bad_lines += 1
                continue
            if len(parts) > 5:
                st.polygon_lines += 1
            try:
                cls = int(float(parts[0]))
                x, y, w, h = (float(v) for v in parts[1:5])
            except ValueError:
                st.bad_lines += 1
                continue
            if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)):
                st.out_of_range += 1
            st.class_counts[cls] += 1
            sizes.append((w, h))
    st.n_boxes = int(sum(st.class_counts.values()))
    st.sizes = np.array(sizes, dtype=float) if sizes else np.zeros((0, 2))
    return st


def read_yolo_labels(label_path: Path, img_w: int, img_h: int) -> List[dict]:
    """YOLO-рядки → [{'cls': int, 'xyxy': (x1,y1,x2,y2) у пікселях}]."""
    out = []
    if not Path(label_path).exists():
        return out
    for line in Path(label_path).read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            c = int(float(parts[0]))
            x, y, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        out.append({"cls": c, "xyxy": ((x - w / 2) * img_w, (y - h / 2) * img_h, (x + w / 2) * img_w, (y + h / 2) * img_h)})
    return out


# ------------------------------------------------------------------------------
# 3. ІМЕНА КЛАСІВ ТА YAML
# ------------------------------------------------------------------------------
def parse_source_names(data_dir: str = DATA_DIR) -> Optional[List[str]]:
    """Імена класів з власного yaml набору даних (не з нашого згенерованого)."""
    root = Path(data_dir)
    for p in sorted(list(root.rglob("*.yaml")) + list(root.rglob("*.yml")), key=lambda q: len(q.parts)):
        if p.name == YAML_NAME:
            continue
        try:
            cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        names = (cfg or {}).get("names") if isinstance(cfg, dict) else None
        if isinstance(names, dict):
            try:
                return [names[k] for k in sorted(names, key=lambda x: int(x))]
            except (TypeError, ValueError):
                continue
        if isinstance(names, list) and names:
            return [str(n) for n in names]
    return None


@dataclass
class IndoorDataset:
    root: str
    yaml_path: str
    names: List[str]
    names_source: str
    splits: Dict[str, SplitStats]
    issues: List[str]

    @property
    def nc(self) -> int:
        return len(self.names)

    def support(self, split: str) -> np.ndarray:
        """К-сть об'єктів кожного класу у вибірці (для таблиці метрик)."""
        st = self.splits.get(split)
        return np.array([st.class_counts.get(i, 0) for i in range(self.nc)]) if st else np.zeros(self.nc, int)

    def splits_table(self) -> pd.DataFrame:
        rows = []
        for name, st in self.splits.items():
            rows.append({
                "split": name, "images": st.n_images, "label_files": st.n_label_files,
                "no_label": st.images_without_labels, "orphan_labels": st.labels_without_images,
                "boxes": st.n_boxes, "bad_lines": st.bad_lines + st.out_of_range,
            })
        return pd.DataFrame(rows)

    def class_table(self) -> pd.DataFrame:
        data = {"id": list(range(self.nc)), "name": self.names}
        for split, st in self.splits.items():
            data[split] = [st.class_counts.get(i, 0) for i in range(self.nc)]
        return pd.DataFrame(data)


def prepare_dataset(data_dir: str = DATA_DIR) -> IndoorDataset:
    splits_dirs = ensure_dataset(data_dir)
    stats = {name: scan_split(name, *dirs) for name, dirs in splits_dirs.items()}

    issues: List[str] = []
    max_id = max((max(st.class_counts) for st in stats.values() if st.class_counts), default=-1)

    src_names = parse_source_names(data_dir)
    if src_names and len(src_names) > max_id:
        names, source = src_names, "data.yaml самого набору даних"
    elif len(FALLBACK_CLASSES) > max_id:
        names, source = list(FALLBACK_CLASSES), "список із ТЗ (власного yaml у наборі даних не знайдено)"
        issues.append("Імена класів взято зі списку ТЗ — порядок не звірено з yaml набору даних.")
    else:
        names, source = [f"class_{i}" for i in range(max_id + 1)], "згенеровано автоматично"
        issues.append(f"У розмітці є id до {max_id}, а відомих імен лише {len(FALLBACK_CLASSES)}.")

    for name, st in stats.items():
        if st.labels_without_images:
            issues.append(f"[{name}] {st.labels_without_images} файлів розмітки без зображення.")
        if st.bad_lines or st.out_of_range:
            issues.append(f"[{name}] {st.bad_lines} некоректних рядків, {st.out_of_range} з координатами поза [0,1].")
        if st.polygon_lines:
            issues.append(f"[{name}] {st.polygon_lines} рядків мають >5 полів (схоже на полігони, а не рамки).")
        if st.n_images and st.images_without_labels / st.n_images > 0.2:
            issues.append(f"[{name}] {st.images_without_labels} із {st.n_images} зображень без розмітки (>20%).")

    if "val" not in stats and "test" in stats:
        issues.append("Валідаційної вибірки немає — для оцінки використано test.")
        splits_dirs = {**splits_dirs, "val": splits_dirs["test"]}
        stats["val"] = scan_split("val", *splits_dirs["val"])
    if "val" not in stats:
        raise RuntimeError("У наборі даних немає ні val, ні test — нічого оцінювати.")

    cfg = {"path": str(Path(data_dir).resolve()), "train": str(stats["train"].images_dir.resolve()),
           "val": str(stats["val"].images_dir.resolve())}
    if "test" in stats:
        cfg["test"] = str(stats["test"].images_dir.resolve())
    cfg["nc"] = len(names)
    cfg["names"] = {i: n for i, n in enumerate(names)}

    yaml_path = os.path.join(data_dir, YAML_NAME)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    logger.info(f"✅ YOLO YAML конфіг згенеровано: {yaml_path} (класів: {len(names)}, джерело імен: {source})")

    return IndoorDataset(root=str(data_dir), yaml_path=yaml_path, names=names, names_source=source,
                         splits=stats, issues=issues)


def prepare_yolo_dataset(data_dir: str = DATA_DIR) -> str:
    """Сумісність зі старим API: повертає лише шлях до YAML."""
    return prepare_dataset(data_dir).yaml_path
