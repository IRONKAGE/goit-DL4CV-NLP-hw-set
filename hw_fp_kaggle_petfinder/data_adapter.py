"""
🗃️ DATA LAYER (pure pandas/numpy — БЕЗ імпорту TF/Torch/Keras).

Чому так: цей модуль імпортується і в ноутбуці, і в ізольованих підпроцесах-екстракторах
(TF-процес не повинен тягнути torch і навпаки — на macOS це класичне джерело
конфліктів libomp). Тому тут лише pandas / numpy / sklearn.

Що вміє:
  * ensure_data()        — завантаження ТІЛЬКИ приватного змагання (без fallback на публічне);
  * load_competition()   — схема-агностичне читання train/test/sample_submission + індекс ВСІХ фото;
  * build_tabular()      — автоматичні табличні + текстові мета-ознаки для будь-яких колонок;
  * make_folds()         — стратифіковані (і групові, якщо є RescuerID) фолди;
  * load_embeddings()    — завантаження кешованих ембедингів (TF / PyTorch екстрактори);
  * make_submission()    — валідація та збереження submission.csv.
"""
from __future__ import annotations

import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("hw_fp.data_adapter")

COMPETITION = "deep-learning-for-computer-vision-and-nlp-2026-10"
DATA_DIR = "data/hw_fp"
ID_COL, TEXT_COL, TARGET_COL = "PetID", "Description", "AdoptionSpeed"
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
_TRAIL_IDX = re.compile(r"^(?P<pid>.+?)[-_](?P<idx>\d+)$")


# ------------------------------------------------------------------------------
# 1. ЗАВАНТАЖЕННЯ (тільки приватне змагання)
# ------------------------------------------------------------------------------
def find_file(data_dir: str, name: str) -> Optional[Path]:
    hits = sorted(Path(data_dir).rglob(name), key=lambda p: len(p.parts))
    return hits[0] if hits else None


def ensure_data(data_dir: str = DATA_DIR) -> None:
    """Завантажує та розпаковує приватне змагання. Публічного fallback НЕМАЄ — краще впасти гучно."""
    has_csv = find_file(data_dir, "train.csv") is not None and find_file(data_dir, "test.csv") is not None
    has_img = next(Path(data_dir).rglob("*.jpg"), None) is not None if Path(data_dir).exists() else False
    if has_csv and has_img:
        logger.info("🔋 Дані вже на диску — завантаження пропущено.")
        return

    from core.etl_core import SecureDownloader  # ледачий імпорт: core тягне torch/mlflow

    def _kaggle_private(target_path: str) -> None:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.competition_download_files(COMPETITION, path=data_dir, quiet=False)
        zips = sorted(Path(data_dir).glob("*.zip"), key=os.path.getmtime)
        if not zips:
            raise FileNotFoundError("Kaggle API не залишив .zip у каталозі даних")
        os.replace(zips[-1], target_path)

    try:
        dl = SecureDownloader(
            dataset_path=None,
            fallback_generator=_kaggle_private,
            data_dir=data_dir,
            zip_name="competition.zip",
        )
        dl.download(target_filename="competition.zip")
        dl.extract_atomically(expected_filename="train.csv")
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"Не вдалося завантажити приватне змагання '{COMPETITION}': {e}\n"
            "Перевір: (1) KAGGLE_USERNAME/KAGGLE_KEY у .env, (2) натиснуто 'Join/Late Submission' і прийнято Rules, "
            "(3) акаунт верифіковано телефоном. Публічний PetFinder НЕ використовується навмисно "
            "(інша схема, 5 класів)."
        ) from e


# ------------------------------------------------------------------------------
# 2. СХЕМА-АГНОСТИЧНЕ ЧИТАННЯ
# ------------------------------------------------------------------------------
@dataclass
class Competition:
    data_dir: str
    train: pd.DataFrame
    test: pd.DataFrame
    sample_sub: Optional[pd.DataFrame]
    id_col: str
    text_col: str
    target_col: str
    images: Dict[str, List[str]]

    @property
    def n_train(self) -> int:
        return len(self.train)

    @property
    def n_test(self) -> int:
        return len(self.test)

    @property
    def all_ids(self) -> List[str]:
        return list(self.train[self.id_col].astype(str)) + list(self.test[self.id_col].astype(str))

    @property
    def y(self) -> np.ndarray:
        return self.train[self.target_col].to_numpy(dtype=np.float32)

    @property
    def lo(self) -> int:
        return int(self.train[self.target_col].min())

    @property
    def hi(self) -> int:
        return int(self.train[self.target_col].max())

    @property
    def full(self) -> pd.DataFrame:
        """train + test (без таргета), порядок = all_ids."""
        tr = self.train.drop(columns=[self.target_col])
        te = self.test.drop(columns=[self.target_col], errors="ignore")
        return pd.concat([tr, te], ignore_index=True)

    @property
    def full_text(self) -> pd.Series:
        t = self.full[self.text_col].fillna("").astype(str).str.strip()
        return t.where(t != "", "no description")


def _read_csv(path: Path, use_core: bool = False) -> pd.DataFrame:
    """use_core=True → 3-рівневий smart_read_csv із core.etl_core (engine routing → encoding → data healing).
    У TF-підпроцесі use_core=False: імпорт core тягне torch/mlflow, а ми свідомо тримаємо TF-процес чистим."""
    if use_core:
        from core.etl_core import smart_read_csv
        return smart_read_csv(str(path), path.name)
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


def index_images(data_dir: str, valid_ids: List[str]) -> Dict[str, List[str]]:
    """PetID -> відсортований список ВСІХ його фото. Підтримує `ID-1.jpg`, `ID_1.jpg` та `.../ID/1.jpg`."""
    ids = set(valid_ids)
    out: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    unmatched = 0
    for p in Path(data_dir).rglob("*"):
        if p.suffix.lower() not in IMG_EXT or p.name.startswith("dummy"):
            continue
        stem, pid, order = p.stem, None, 0
        m = _TRAIL_IDX.match(stem)
        cands = ([m.group("pid")] if m else []) + [stem, p.parent.name]
        for c in cands:
            if c in ids:
                pid = c
                break
        if pid is None:
            unmatched += 1
            continue
        order = int(m.group("idx")) if (m and m.group("pid") == pid) else 0
        out[pid].append((order, str(p)))
    if unmatched:
        logger.warning(f"⚠️ {unmatched} зображень не зіставлено з жодним PetID.")
    return {k: [p for _, p in sorted(v, key=lambda t: (t[0], t[1]))] for k, v in out.items()}


def _guess_text_col(df: pd.DataFrame, exclude: set) -> Optional[str]:
    best, best_len = None, 0.0
    for c in df.columns:
        if c in exclude or pd.api.types.is_numeric_dtype(df[c]):
            continue
        mean_len = df[c].dropna().astype(str).str.len().mean()
        if mean_len and mean_len > best_len:
            best, best_len = c, mean_len
    return best if best_len > 25 else None


def load_competition(data_dir: str = DATA_DIR, use_core_reader: bool = False) -> Competition:
    f_tr, f_te = find_file(data_dir, "train.csv"), find_file(data_dir, "test.csv")
    f_sub = find_file(data_dir, "sample_submission.csv")
    if f_tr is None or f_te is None:
        raise FileNotFoundError(f"train.csv/test.csv не знайдено в {data_dir}. Запусти ensure_data().")
    train, test = _read_csv(f_tr, use_core_reader), _read_csv(f_te, use_core_reader)
    sub = _read_csv(f_sub, use_core_reader) if f_sub else None

    if ID_COL not in train.columns or TARGET_COL not in train.columns:
        raise KeyError(f"У train.csv очікуються колонки {ID_COL}/{TARGET_COL}. Є: {list(train.columns)}")
    if TARGET_COL in test.columns:
        logger.warning("test.csv містить таргет — ігноруємо його (оцінювання на Kaggle).")
        test = test.drop(columns=[TARGET_COL])

    text_col = TEXT_COL if TEXT_COL in train.columns else _guess_text_col(train, {ID_COL, TARGET_COL})
    if text_col is None:
        logger.warning("Текстової колонки не знайдено — створюю порожню.")
        text_col = "__text__"
        train[text_col] = ""
        test[text_col] = ""
    train[ID_COL] = train[ID_COL].astype(str)
    test[ID_COL] = test[ID_COL].astype(str)
    assert train[ID_COL].is_unique and test[ID_COL].is_unique, "Дублікати PetID"
    overlap = set(train[ID_COL]) & set(test[ID_COL])
    if overlap:
        logger.warning(f"⚠️ {len(overlap)} PetID одночасно в train і test!")

    comp = Competition(
        data_dir=str(data_dir), train=train, test=test, sample_sub=sub,
        id_col=ID_COL, text_col=text_col, target_col=TARGET_COL, images={},
    )
    comp.images = index_images(data_dir, comp.all_ids)
    n_with = sum(1 for p in comp.all_ids if p in comp.images)
    n_files = sum(len(v) for v in comp.images.values())
    logger.info(
        f"📂 train={comp.n_train}, test={comp.n_test}, класи {comp.lo}..{comp.hi}, "
        f"текст='{text_col}', тварин із фото: {n_with}/{len(comp.all_ids)}, фото: {n_files} "
        f"(~{n_files / max(n_with, 1):.1f} на тварину)"
    )
    return comp


# ------------------------------------------------------------------------------
# 3. ТАБЛИЧНІ + ТЕКСТОВІ МЕТА-ОЗНАКИ (авто-детект, без хардкоду колонок)
# ------------------------------------------------------------------------------
def build_tabular(comp: Competition, max_onehot: int = 12, max_ordinal: int = 40) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Повертає (tab_tree, tab_nn) для train+test (порядок = all_ids).
      tab_tree — числа + ординальні коди + frequency-encoding (NaN дозволені; для GBM).
      tab_nn   — числа (медіана замість NaN) + one-hot низькокардинальних (для нейромережі).
    """
    full = comp.full
    skip = {comp.id_col, comp.text_col, comp.target_col}
    tree, nn = {}, {}

    for c in [c for c in full.columns if c not in skip]:
        s = full[c]
        if pd.api.types.is_bool_dtype(s):
            s = s.astype(int)
        if pd.api.types.is_numeric_dtype(s):
            v = s.astype("float32")
            tree[c] = v.to_numpy()
            nn[c] = v.fillna(v.median()).to_numpy()
        else:
            s = s.astype("object").where(s.notna(), None)
            nun = s.nunique(dropna=True)
            freq = s.map(s.value_counts(dropna=True)).astype("float32") / len(s)
            tree[f"{c}__freq"] = freq.fillna(0).to_numpy()
            nn[f"{c}__freq"] = tree[f"{c}__freq"]
            if nun <= max_ordinal:
                codes, _ = pd.factorize(s)
                tree[c] = np.where(codes < 0, np.nan, codes).astype("float32")
            if nun <= max_onehot:
                oh = pd.get_dummies(s, prefix=c, dummy_na=True, dtype="float32")
                for col in oh.columns:
                    nn[col] = oh[col].to_numpy()

    # --- текстові мета-ознаки
    t = full[comp.text_col].fillna("").astype(str)
    meta = {
        "txt_chars": t.str.len(),
        "txt_words": t.str.split().str.len().fillna(0),
        "txt_sents": t.str.count(r"[.!?]+"),
        "txt_excl": t.str.count("!"),
        "txt_upper": t.str.count(r"[A-ZА-ЯІЇЄҐ]"),
        "txt_digits": t.str.count(r"\d"),
        "txt_nonascii": t.apply(lambda x: sum(ord(ch) > 127 for ch in x)),
        "txt_uniq_ratio": t.apply(lambda x: len(set(x.lower().split())) / max(len(x.split()), 1)),
        "txt_empty": (t.str.strip() == "").astype(int),
    }
    for k, v in meta.items():
        tree[k] = v.astype("float32").to_numpy()
        nn[k] = np.log1p(v.astype("float32").to_numpy()) if k != "txt_uniq_ratio" else v.astype("float32").to_numpy()

    # --- кількість фото
    n_img = full[comp.id_col].map(lambda p: len(comp.images.get(p, []))).astype("float32").to_numpy()
    tree["n_images"] = n_img
    nn["n_images"] = np.log1p(n_img)

    return pd.DataFrame(tree), pd.DataFrame(nn)


# ------------------------------------------------------------------------------
# 4. ФОЛДИ ТА ЕМБЕДИНГИ
# ------------------------------------------------------------------------------
def make_folds(comp: Competition, n_splits: int = 5, seed: int = 42) -> np.ndarray:
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    y = comp.train[comp.target_col].to_numpy().astype(int)
    folds = np.full(len(y), -1, dtype=int)
    group_col = next((c for c in ("RescuerID", "rescuer_id") if c in comp.train.columns), None)
    if group_col:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        it = splitter.split(np.zeros(len(y)), y, groups=comp.train[group_col].astype(str))
        logger.info(f"🧩 Групові фолди за {group_col}")
    else:
        it = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)
    for k, (_, va) in enumerate(it):
        folds[va] = k
    return folds


def feature_dir(data_dir: str = DATA_DIR) -> Path:
    return Path(data_dir) / "features"


def load_embeddings(comp: Competition, data_dir: str = DATA_DIR) -> Dict[str, np.ndarray]:
    """Завантажує всі кешовані ембединги (*.npz) і перевіряє, що порядок PetID збігається."""
    blocks: Dict[str, np.ndarray] = {}
    expected = np.array(comp.all_ids, dtype=str)
    for f in sorted(feature_dir(data_dir).glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        if not np.array_equal(z["ids"].astype(str), expected):
            logger.warning(f"⚠️ {f.name}: порядок PetID не збігається — блок пропущено (видали файл і перерахуй).")
            continue
        blocks[f.stem] = z["emb"].astype(np.float32)
        logger.info(f"🧱 {f.stem}: {blocks[f.stem].shape}")
    return blocks


# ------------------------------------------------------------------------------
# 5. SUBMISSION
# ------------------------------------------------------------------------------
def make_submission(comp: Competition, labels: np.ndarray, out_path: Optional[str] = None) -> pd.DataFrame:
    sub = pd.DataFrame({comp.id_col: comp.test[comp.id_col].values,
                        comp.target_col: np.asarray(labels).astype(int)})
    assert len(sub) == comp.n_test and sub[comp.id_col].is_unique
    assert sub[comp.target_col].between(comp.lo, comp.hi).all(), "Прогноз поза діапазоном класів"
    if comp.sample_sub is not None and list(comp.sample_sub.columns) != list(sub.columns):
        raise ValueError(f"Колонки submission {list(sub.columns)} ≠ sample_submission {list(comp.sample_sub.columns)}")
    out_path = out_path or os.path.join(comp.data_dir, "results", "submission.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sub.to_csv(out_path, index=False)
    logger.info(f"💾 {out_path}: {len(sub)} рядків, розподіл {sub[comp.target_col].value_counts().sort_index().to_dict()}")
    return sub
