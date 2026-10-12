"""
📐 АНАЛІТИКА HW 03 (без залежності від ultralytics/plotly — лише numpy/pandas/PIL/matplotlib).

Тут живе уся логіка:
  * метрики по класах: Precision / Recall / F1 / AP, найкращі й найгірші класи;
  * матриця помилок: нормалізація, топ-пар плутанини, пропуски (FN) та хибні спрацювання (FP);
  * діагностика кривих навчання (перенавчання / недонавчання);
  * малювання рамок з УКРАЇНСЬКИМИ підписами (PIL + шрифт DejaVu із matplotlib: кирилиця без «[][][]»);
  * автоматичні висновки (3–5 речень) і рекомендації, що спираються на реально отримані числа.
"""
import base64
import io
import os
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# ------------------------------------------------------------------------------
# Назви колонок таблиць
# ------------------------------------------------------------------------------
COL_ID, COL_NAME_EN, COL_CLASS = "ID", "Назва в YAML", "Клас"
COL_P, COL_R, COL_F1 = "Precision (точність)", "Recall (повнота)", "F1"
COL_AP50, COL_AP, COL_N = "AP@0.5", "AP@0.5:0.95", "Об'єктів у вибірці"

# Орієнтири з ТЗ для порівняння (нижня й верхня межі)
REF_MAP50 = (0.40, 0.60)
REF_MAP95 = (0.25, 0.40)


def fmt_elapsed(seconds: float) -> str:
    s = int(round(max(float(seconds), 0.0)))
    if s < 60:
        return f"{s} с"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m} хв {s:02d} с"
    h, m = divmod(m, 60)
    return f"{h} год {m:02d} хв"


def pct(x: float, digits: int = 1) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{digits}f}%"


# ------------------------------------------------------------------------------
# Пристрій і шляхи
# ------------------------------------------------------------------------------
def yolo_device(torch_device, override: str = "auto") -> str:
    """core.hardware → рядок для Ultralytics ('mps' / 'cuda:0' / 'cpu')."""
    if override and override != "auto":
        return override
    t = getattr(torch_device, "type", str(torch_device))
    return {"mps": "mps", "cuda": "cuda:0"}.get(t, "cpu")


def resolve_run(project_root: str, model_file: str, smoke: bool) -> Dict[str, str]:
    stem = os.path.splitext(os.path.basename(model_file))[0]
    runs_dir = os.path.join(project_root, "runs", "detect")
    name = f"indoor_{stem}" + ("_smoke" if smoke else "")
    return {"runs_dir": runs_dir, "run_name": name,
            "run_dir": os.path.join(runs_dir, name),
            "weights": os.path.join(runs_dir, name, "weights", "best.pt"),
            "results_csv": os.path.join(runs_dir, name, "results.csv")}


# ------------------------------------------------------------------------------
# Малювання: кирилиця через DejaVu (є в matplotlib, що вже в залежностях проекту)
# ------------------------------------------------------------------------------
_FONT_CACHE: Dict[int, ImageFont.ImageFont] = {}
_PALETTE = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
            (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207)]


def _font(size: int):
    if size not in _FONT_CACHE:
        try:
            from matplotlib import font_manager

            _FONT_CACHE[size] = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)
        except Exception:  # noqa: BLE001
            _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


def class_color(i: int):
    return _PALETTE[int(i) % len(_PALETTE)]


def draw_boxes(img: Image.Image, boxes: List[dict], names: List[str], label_fn: Callable[[str], str],
               show_conf: bool = False, color=None, width: Optional[int] = None) -> Image.Image:
    """boxes: [{'cls': int, 'xyxy': (x1,y1,x2,y2), 'conf': float|None}] у пікселях."""
    img = img.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    short = min(img.size)
    width = width or max(2, round(short / 200))
    font = _font(max(12, round(short / 32)))
    for b in boxes:
        x1, y1, x2, y2 = b["xyxy"]
        c = int(b["cls"])
        col = color or class_color(c)
        d.rectangle([x1, y1, x2, y2], outline=col, width=width)
        label = label_fn(names[c]) if 0 <= c < len(names) else f"клас {c}"
        if show_conf and b.get("conf") is not None:
            label += f" {float(b['conf']):.2f}"
        l, t, r, bt = d.textbbox((0, 0), label, font=font)
        tw, th = r - l, bt - t
        ty = max(0, y1 - th - 6)
        d.rectangle([x1, ty, x1 + tw + 8, ty + th + 6], fill=col)
        d.text((x1 + 4, ty + 2), label, fill=(255, 255, 255), font=font)
    return img


def pil_to_html(img: Image.Image, max_width: int = 640, quality: int = 85) -> str:
    """<img> з вбудованим JPEG (base64) — працює в marimo та в експорті в .ipynb без файлів на диску."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return (f'<img src="data:image/jpeg;base64,{b64}" '
            f'style="max-width:{max_width}px;width:100%;border-radius:8px;"/>')


# ------------------------------------------------------------------------------
# Метрики по класах
# ------------------------------------------------------------------------------
def per_class_table(box, names: List[str], label_fn: Callable[[str], str], support: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    box — ultralytics Metric (model.val().box): масиви p, r, f1, ap50, ap та ap_class_index.
    Precision / Recall / F1 Ultralytics рахує при порозі впевненості, що максимізує F1.
    Класи, яких немає у вибірці, отримують NaN.
    """
    nc = len(names)
    out = {k: np.full(nc, np.nan) for k in ("p", "r", "f1", "ap50", "ap")}
    idx = np.asarray(getattr(box, "ap_class_index", np.arange(len(getattr(box, "p", [])))), dtype=int)
    for key in out:
        vals = getattr(box, key, None)
        if vals is None:
            continue
        vals = np.asarray(vals, dtype=float)
        out[key][idx[:len(vals)]] = vals[:len(idx)]
    if np.isnan(out["f1"]).all():                       # запасний розрахунок F1
        den = out["p"] + out["r"]
        out["f1"] = np.where(den > 0, 2 * out["p"] * out["r"] / np.where(den > 0, den, 1), np.nan)

    df = pd.DataFrame({
        COL_ID: np.arange(nc), COL_NAME_EN: names, COL_CLASS: [label_fn(n) for n in names],
        COL_P: out["p"], COL_R: out["r"], COL_F1: out["f1"], COL_AP50: out["ap50"], COL_AP: out["ap"],
    })
    df[COL_N] = np.asarray(support, dtype=int) if support is not None else np.nan
    return df


def overall_metrics(box, df: pd.DataFrame) -> Dict[str, float]:
    def _g(attr, col):
        v = getattr(box, attr, None)
        return float(v) if v is not None else float(np.nanmean(df[col]))
    return {"mp": _g("mp", COL_P), "mr": _g("mr", COL_R), "map50": _g("map50", COL_AP50), "map95": _g("map", COL_AP),
            "f1": float(np.nanmean(df[COL_F1]))}


def classes_detected(df: pd.DataFrame, ap50_threshold: float) -> pd.DataFrame:
    return df[df[COL_AP50] >= ap50_threshold]


# ------------------------------------------------------------------------------
# Матриця помилок (Ultralytics: matrix[pred, true]; останній індекс = фон)
# ------------------------------------------------------------------------------
def confusion_true_pred(matrix: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Рядки — справжній клас, стовпці — передбачений; нормалізація по справжньому класу."""
    t = np.asarray(matrix, dtype=float).T
    if not normalize:
        return t
    s = t.sum(axis=1, keepdims=True)
    return np.divide(t, s, out=np.zeros_like(t), where=s > 0)


def classification_accuracy(matrix: np.ndarray, nc: int) -> float:
    """Частка правильно класифікованих серед усіх зіставлень «передбачення ↔ справжній об'єкт» (без фону)."""
    m = np.asarray(matrix, dtype=float)[:nc, :nc]
    return float(np.trace(m) / m.sum()) if m.sum() > 0 else float("nan")


def top_confusions(matrix: np.ndarray, names: List[str], k: int = 6, min_count: int = 1) -> List[dict]:
    """Найчастіші помилки «справжній клас → передбачений інший клас» (без фону)."""
    m = np.asarray(matrix, dtype=float)
    nc = len(names)
    res = []
    for i in range(nc):                               # справжній
        total = m[:, i].sum()
        if total <= 0:
            continue
        for j in range(nc):                           # передбачений
            if i == j or m[j, i] < min_count:
                continue
            res.append({"true": names[i], "pred": names[j], "count": int(m[j, i]), "fraction": float(m[j, i] / total)})
    res.sort(key=lambda r: (-r["fraction"], -r["count"]))
    return res[:k]


def background_rates(matrix: np.ndarray, names: List[str]) -> pd.DataFrame:
    """Пропущені об'єкти (FN: справжній клас → «фон») та хибні спрацювання (FP: «фон» → клас) по класах."""
    m = np.asarray(matrix, dtype=float)
    nc = len(names)
    rows = []
    for i in range(nc):
        true_total = m[:, i].sum()
        pred_total = m[i, :].sum()
        rows.append({"name": names[i],
                     "missed": float(m[nc, i] / true_total) if true_total > 0 else np.nan,
                     "false_pos": float(m[i, nc] / pred_total) if pred_total > 0 else np.nan})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------
# Криві навчання
# ------------------------------------------------------------------------------
def read_results(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def loss_pairs(df: pd.DataFrame) -> List[str]:
    """Суфікси втрат, що мають і train/, і val/ та не є суцільними нулями (напр. dfl_loss у YOLO26 відсутній)."""
    out = []
    for c in df.columns:
        if c.startswith("train/") and c.endswith("loss"):
            suf = c.split("/", 1)[1]
            v = f"val/{suf}"
            if v in df.columns and not (np.allclose(df[c].fillna(0), 0) and np.allclose(df[v].fillna(0), 0)):
                out.append(suf)
    return out


def fit_diagnostics(df: pd.DataFrame) -> Dict[str, object]:
    n = len(df)
    d: Dict[str, object] = {"n_epochs": n, "verdict": "insufficient"}
    pairs = loss_pairs(df)
    main = "box_loss" if "box_loss" in pairs else (pairs[0] if pairs else None)
    if main:
        tr, va = df[f"train/{main}"].to_numpy(float), df[f"val/{main}"].to_numpy(float)
        d.update(loss_name=main, train_final=float(tr[-1]), val_final=float(va[-1]), train_first=float(tr[0]),
                 best_val_epoch=int(np.argmin(va)) + 1, val_min=float(va.min()),
                 val_rise=float(va[-1] / va.min() - 1) if va.min() > 0 else 0.0)
    m50 = df["metrics/mAP50(B)"].to_numpy(float) if "metrics/mAP50(B)" in df.columns else None
    if m50 is not None and n:
        d.update(map50_final=float(m50[-1]), map50_best=float(m50.max()), map50_best_epoch=int(np.argmax(m50)) + 1,
                 map50_gain_last=float(m50[-1] - m50[max(0, n - 3)]))
    if n >= 3 and main:
        overfit = d["val_rise"] > 0.05 and d["best_val_epoch"] < n and d["train_final"] < d["train_first"]
        growing = m50 is not None and d["map50_gain_last"] > 0.005 and d["map50_best_epoch"] >= n - 1
        d["verdict"] = "overfit" if overfit else ("underfit" if growing else "ok")
    return d


def describe_fit(d: Dict[str, object]) -> str:
    n, v = d["n_epochs"], d["verdict"]
    if v == "insufficient":
        return f"Епох замало ({n}) для надійного висновку про перенавчання чи недонавчання."
    base = (f"train {d['loss_name']} {d['train_first']:.3f} → {d['train_final']:.3f}, "
            f"val {d['loss_name']}: мінімум {d['val_min']:.3f} на епосі {d['best_val_epoch']}, фінал {d['val_final']:.3f}")
    if v == "overfit":
        return f"Ознаки перенавчання: валідаційна втрата зросла на {100 * d['val_rise']:.0f}% від мінімуму, тоді як навчальна падала ({base})."
    if v == "underfit":
        return (f"Ознаки недонавчання: mAP@0.5 ще зростав наприкінці (приріст {100 * d['map50_gain_last']:.1f} п.п. за останні епохи), "
                f"плато не досягнуто ({base}).")
    return f"Криві стабільні: помітного перенавчання немає, метрика вийшла на плато ({base})."


# ------------------------------------------------------------------------------
# Автоматичні висновки (3–5 речень) та рекомендації
# ------------------------------------------------------------------------------
def _range_word(x: float, ref) -> str:
    return "нижче очікуваного діапазону" if x < ref[0] else ("у межах очікуваного діапазону" if x <= ref[1] else "вище очікуваного діапазону")


def build_summary(model_name: str, epochs: int, overall: Dict[str, float], df: pd.DataFrame, top_conf: List[dict],
                  fit: Dict[str, object], ap50_threshold: float, label_fn: Callable[[str], str],
                  reason_fn: Callable[[str, str], str]) -> List[str]:
    out = []
    w50, w95 = _range_word(overall["map50"], REF_MAP50), _range_word(overall["map95"], REF_MAP95)
    verdict = w50 if w50 == w95 else f"mAP@0.5 {w50}, а mAP@0.5:0.95 {w95}"
    out.append(f"Модель {model_name} за {epochs} епох досягла mAP@0.5 = {pct(overall['map50'])} і mAP@0.5:0.95 = {pct(overall['map95'])} "
               f"на валідаційній вибірці (середні Precision {pct(overall['mp'])}, Recall {pct(overall['mr'])}, F1 {pct(overall['f1'])}); "
               f"орієнтири з ТЗ — 40–60% та 25–40%, тож результат {verdict}.")

    ok = classes_detected(df, ap50_threshold)
    ranked = df.dropna(subset=[COL_F1]).sort_values(COL_F1, ascending=False)
    best = ", ".join(f"«{r[COL_CLASS]}» (F1 {pct(r[COL_F1], 0)})" for _, r in ranked.head(3).iterrows())
    worst = ", ".join(f"«{r[COL_CLASS]}» (F1 {pct(r[COL_F1], 0)})" for _, r in ranked.tail(3).iloc[::-1].iterrows())
    goal = "досягнуто" if len(ok) >= 3 else "не досягнуто"
    out.append(f"Мету ТЗ (розпізнавати щонайменше 3 з 10 класів) {goal}: {len(ok)} класів мають AP@0.5 ≥ {pct(ap50_threshold, 0)}; "
               f"найкраще визначаються {best}, найгірше — {worst}.")

    out.append(describe_fit(fit))

    if top_conf:
        t = top_conf[0]
        extra = f" та «{label_fn(top_conf[1]['true'])}» → «{label_fn(top_conf[1]['pred'])}» ({pct(top_conf[1]['fraction'], 0)})" if len(top_conf) > 1 else ""
        out.append(f"Найчастіше модель плутає «{label_fn(t['true'])}» з «{label_fn(t['pred'])}» ({pct(t['fraction'], 0)} об'єктів цього класу){extra}; "
                   f"ймовірна причина (гіпотеза) — {reason_fn(t['true'], t['pred'])}.")
    else:
        out.append("Значних міжкласових плутанин матриця не показала — основні втрати припадають на пропущені об'єкти (клас «фон»).")
    return out


def build_recommendations(df: pd.DataFrame, fit: Dict[str, object], top_conf: List[dict], label_fn: Callable[[str], str]) -> List[str]:
    recs = []
    v = fit["verdict"]
    if v == "underfit":
        recs.append(f"**Довше навчання:** mAP@0.5 ще зростав наприкінці ({fit['n_epochs']} епох) — збільшити кількість епох (50–100) і лишити `patience` для ранньої зупинки.")
    elif v == "overfit":
        recs.append(f"**Регуляризація:** val-втрата почала зростати після епохи {fit['best_val_epoch']} — сильніші аугментації (mosaic, copy-paste, HSV), `weight_decay`, `patience`; брати `best.pt`.")
    else:
        recs.append("**Тонке налаштування:** криві стабільні — підбір `lr0`, `imgsz` та тривалості навчання за валідаційним mAP.")

    ranked = df.dropna(subset=[COL_F1]).sort_values(COL_F1)
    weak = ranked.head(3)
    if len(weak) and df[COL_N].notna().any():
        total = df[COL_N].sum()
        rare = [r[COL_CLASS] for _, r in weak.iterrows() if total > 0 and r[COL_N] / total < 0.08]
        if rare:
            recs.append("**Баланс класів:** слабкі класи мають мало прикладів ("
                        + ", ".join(f"«{c}»" for c in rare) + ") — докласти даних, oversampling або copy-paste аугментація саме для них.")
    if top_conf:
        t = top_conf[0]
        recs.append(f"**Розмітка/ієрархія пар класів:** пара «{label_fn(t['true'])}» ↔ «{label_fn(t['pred'])}» плутається найчастіше — "
                    "перевірити узгодженість розмітки, додати контекстні приклади або розглянути атрибутний підхід (об'єкт + стан) замість двох окремих класів.")
    recs.append("**Більша модель / роздільність:** спробувати `yolo26m`/`yolo26l` або `imgsz=800–1024` (дрібні й тонкі об'єкти на кшталт стовпів) ціною часу навчання.")
    recs.append("**Пороги за класами:** підібрати поріг впевненості окремо для кожного класу за кривою F1 (замість єдиного `conf=0.25`); використати TTA на інференсі.")
    return recs
