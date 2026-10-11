"""
🏭 FEATURE EXTRACTORS — кожен фреймворк у своїй зоні, кожен у СВОЄМУ процесі.

    python -m hw_fp_kaggle_petfinder.extractors all            # усе (кешується в data/hw_fp/features/*.npz)
    python -m hw_fp_kaggle_petfinder.extractors tf-images
    python -m hw_fp_kaggle_petfinder.extractors torch-images
    python -m hw_fp_kaggle_petfinder.extractors torch-text --model intfloat/multilingual-e5-base --name txt_e5_base

  🔶 TensorFlow  : tf.data (read_file → decode → resize_with_pad → prefetch) + EfficientNetV2 (KERAS_BACKEND=tensorflow)
  🔥 PyTorch     : CLIP ViT (зображення) + multilingual трансформери (текст), MPS/CUDA/CPU
  🧠 Keras 3     : фінальна fusion-модель (див. model.py, бекенд torch) — у ноутбуці

Ізоляція процесів прибирає конфлікти libomp між TF і Torch на macOS та конфлікт бекендів Keras.
Модуль на верхньому рівні НЕ імпортує tf/torch/keras.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import data_adapter as da

logger = logging.getLogger("hw_fp.extractors")

MAX_IMAGES_PER_PET = int(os.getenv("FP_MAX_IMAGES", "6"))

# (кеш-ім'я, тип, додаткові аргументи)
JOBS = [
    ("img_tf_effv2b3", "tf-images", {}),
    ("img_clip_b32", "torch-images", {"model": "openai/clip-vit-base-patch32", "revision": "refs/pr/66"}),
    ("txt_e5_base", "torch-text", {"model": "intfloat/multilingual-e5-base"}),
    ("txt_minilm", "torch-text", {"model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"}),
]


# ------------------------------------------------------------------------------
# Допоміжне
# ------------------------------------------------------------------------------
def _save(data_dir: str, name: str, emb: np.ndarray, ids: List[str], extra: Optional[dict] = None) -> None:
    fdir = da.feature_dir(data_dir)
    fdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(fdir / f"{name}.npz", emb=emb.astype(np.float32), ids=np.array(ids, dtype=str), **(extra or {}))
    logger.info(f"💾 {name}: {emb.shape} → {fdir / (name + '.npz')}")


def _finalize(acc: np.ndarray, cnt: np.ndarray) -> np.ndarray:
    """Середнє по фото тварини; тварини без жодного фото отримують середній вектор."""
    emb = np.zeros_like(acc)
    ok = cnt > 0
    emb[ok] = acc[ok] / cnt[ok, None]
    if (~ok).any():
        emb[~ok] = emb[ok].mean(0) if ok.any() else 0.0
        logger.warning(f"⚠️ {int((~ok).sum())} тварин без декодованих фото → середній вектор (Graceful Degradation)")
    return emb


# ------------------------------------------------------------------------------
# 🔶 TensorFlow: tf.data + EfficientNetV2
# ------------------------------------------------------------------------------
def run_tf_images(data_dir: str, name: str, backbone: str = "EfficientNetV2B3", size: int = 300,
                  batch: int = 32, tta: bool = False) -> None:
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf  # noqa: WPS433
    import keras
    from tqdm import tqdm

    comp = da.load_competition(data_dir)
    ids = comp.all_ids
    pairs = [(i, p) for i, pid in enumerate(ids) for p in comp.images.get(pid, [])[:MAX_IMAGES_PER_PET]]
    if not pairs:
        raise RuntimeError("Не знайдено жодного зображення")
    paths = np.array([p for _, p in pairs])
    owners = np.array([i for i, _ in pairs], dtype=np.int32)
    logger.info(f"🔶 TF {tf.__version__} | Keras {keras.__version__} ({keras.backend.backend()}) | {len(paths)} фото")

    def _load(path, owner):
        raw = tf.io.read_file(path)
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        # ВАЖЛИВО: Keras EfficientNet має вбудований rescaling і чекає пікселі 0..255 (НЕ 0..1!)
        img = tf.image.resize_with_pad(img, size, size, method="bilinear", antialias=True)
        return tf.cast(img, tf.float32), owner

    ds = (
        tf.data.Dataset.from_tensor_slices((paths, owners))
        .map(_load, num_parallel_calls=tf.data.AUTOTUNE)
        .ignore_errors()                      # биті файли пропускаються, owner йде разом з елементом
        .batch(batch)
        .prefetch(tf.data.AUTOTUNE)
    )

    net = getattr(keras.applications, backbone)(
        include_top=False, weights="imagenet", pooling="avg", input_shape=(size, size, 3)
    )
    dim = int(net.output.shape[-1])

    @tf.function(reduce_retracing=True)
    def fwd(x):
        f = net(x, training=False)
        if tta:
            f = 0.5 * (f + net(tf.image.flip_left_right(x), training=False))
        return f

    acc = np.zeros((len(ids), dim), np.float32)
    cnt = np.zeros(len(ids), np.float32)
    t0 = time.time()
    for x, o in tqdm(ds, total=int(np.ceil(len(paths) / batch)), desc=f"TF {backbone}", file=sys.stdout):
        f = fwd(x).numpy()
        o = o.numpy()
        np.add.at(acc, o, f)
        np.add.at(cnt, o, 1.0)
    logger.info(f"⏱ {time.time() - t0:.0f} c, декодовано {int(cnt.sum())}/{len(paths)} фото")
    _save(data_dir, name, _finalize(acc, cnt), ids)


# ------------------------------------------------------------------------------
# 🔥 PyTorch
# ------------------------------------------------------------------------------
def _is_device_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(k in msg for k in ("mps", "cuda", "out of memory", "xpu", "not implemented for", "device"))


class EnvError(RuntimeError):
    """Проблема середовища (версії пакетів), а не прискорювача — повтор на CPU безглуздий."""


def _require_transformers_torch() -> None:
    import torch
    import transformers
    from transformers.utils import is_torch_available

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.disable_progress_bar()
        hf_logging.set_verbosity_error()
    except Exception:  # noqa: BLE001
        pass
    if not is_torch_available():
        raise EnvError(
            f"transformers {transformers.__version__} не бачить PyTorch {torch.__version__}: ця версія вимагає новішого torch "
            "і мовчки вимикає PyTorch. Зафіксуй у pyproject.toml старішу версію: спочатку \"transformers==5.0.0\" "
            "(у README цього релізу заявлено PyTorch 2.1+), а якщо повідомлення лишиться — \"transformers==4.46.3\"; "
            "потім `make setup`."
        )


def _pick_device(torch, pref: str = "auto"):
    """Апаратний маршрутизатор — з core.hardware (CUDA/ROCm/MPS/XPU/CPU + ML_BACKEND=cpu з .env).
    Власна логіка нижче — лише запасний варіант, якщо core недоступний."""
    if pref != "auto":
        return torch.device(pref)
    try:
        from core import get_hardware_config
        dev, name = get_hardware_config()
        logger.info(f"⚙️ core.hardware → {name}")
        return dev if not isinstance(dev, str) else torch.device(dev)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"core.hardware недоступний ({e}) — використовую запасну логіку")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _free(model, device) -> None:
    """Звільнення VRAM/RAM через core.hardware.clear_vram (фолбек — просто gc)."""
    del model
    try:
        from core import clear_vram
        clear_vram(device)
    except Exception:  # noqa: BLE001
        import gc
        gc.collect()


_CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
_CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)


def _load_clip_img(path: str, size: int = 224):
    from PIL import Image

    try:
        from core import read_image_safe       # Graceful Degradation із core.datasets (None для битого файлу)
    except Exception:  # noqa: BLE001
        def read_image_safe(p, mode="RGB"):
            try:
                with Image.open(p) as _im:
                    return _im.convert(mode)
            except Exception:  # noqa: BLE001
                return None
    try:
        im = read_image_safe(path)
        if im is None:
            return None
        w, h = im.size
        s = size / min(w, h)
        im = im.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
        w, h = im.size
        l, t = (w - size) // 2, (h - size) // 2
        im = im.crop((l, t, l + size, t + size))
        arr = (np.asarray(im, dtype=np.float32) / 255.0 - _CLIP_MEAN) / _CLIP_STD
        return arr.transpose(2, 0, 1)
    except Exception:  # noqa: BLE001
        return None


def _torch_images(data_dir: str, name: str, model_name: str, batch: int, device_pref: str, revision=None) -> None:
    _require_transformers_torch()
    import torch
    from tqdm import tqdm
    from transformers import CLIPModel

    comp = da.load_competition(data_dir)
    ids = comp.all_ids
    pairs = [(i, p) for i, pid in enumerate(ids) for p in comp.images.get(pid, [])[:MAX_IMAGES_PER_PET]]
    device = _pick_device(torch, device_pref)
    torch.set_num_threads(os.cpu_count() or 4)
    logger.info(f"🔥 torch {torch.__version__} | {model_name} | {device} | {len(pairs)} фото")

    model = CLIPModel.from_pretrained(model_name, revision=revision).eval().to(device)
    dim = model.config.projection_dim
    acc = np.zeros((len(ids), dim), np.float32)
    cnt = np.zeros(len(ids), np.float32)

    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as pool, torch.inference_mode():
        for s in tqdm(range(0, len(pairs), batch), desc="CLIP", file=sys.stdout):
            chunk = pairs[s:s + batch]
            arrs = list(pool.map(lambda t: _load_clip_img(t[1]), chunk))
            keep = [(i, a) for (i, _), a in zip(chunk, arrs) if a is not None]
            if not keep:
                continue
            x = torch.from_numpy(np.stack([a for _, a in keep])).to(device)
            pooled = model.vision_model(pixel_values=x).pooler_output     # однаково у 4.x та 5.x
            f = model.visual_projection(pooled)                            # = старий get_image_features()
            f = torch.nn.functional.normalize(f, dim=-1).float().cpu().numpy()
            o = np.array([i for i, _ in keep])
            np.add.at(acc, o, f)
            np.add.at(cnt, o, 1.0)
    _save(data_dir, name, _finalize(acc, cnt), ids)
    _free(model, device)


def run_torch_images(data_dir: str, name: str, model_name: str = "openai/clip-vit-base-patch32",
                     batch: int = 64, device: str = "auto", revision=None) -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    try:
        _torch_images(data_dir, name, model_name, batch, device, revision)
    except EnvError:
        raise
    except Exception as e:  # noqa: BLE001
        if device == "cpu" or not _is_device_error(e):
            raise                                   # не проблема прискорювача — повтор на CPU нічого не змінить
        logger.warning(f"⚠️ Збій на прискорювачі ({e}). Повтор на CPU...")
        _torch_images(data_dir, name, model_name, batch, "cpu", revision)


def _torch_text(data_dir: str, name: str, model_name: str, max_len: int, batch: int, device_pref: str) -> None:
    _require_transformers_torch()
    import torch
    from tqdm import tqdm
    from transformers import AutoModel, AutoTokenizer

    comp = da.load_competition(data_dir)
    texts = comp.full_text.tolist()
    if "e5" in model_name.lower():
        texts = ["query: " + t for t in texts]          # вимога сімейства E5
    device = _pick_device(torch, device_pref)
    torch.set_num_threads(os.cpu_count() or 4)
    logger.info(f"🔥 torch {torch.__version__} | {model_name} | {device} | {len(texts)} текстів")

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(device)
    order = np.argsort([len(t) for t in texts])          # сортування за довжиною → мінімум padding
    out = np.zeros((len(texts), model.config.hidden_size), np.float32)

    with torch.inference_mode():
        for s in tqdm(range(0, len(texts), batch), desc="Text", file=sys.stdout):
            idx = order[s:s + batch]
            enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)   # mean pooling
            out[idx] = torch.nn.functional.normalize(e, dim=-1).float().cpu().numpy()
    _save(data_dir, name, out, comp.all_ids)
    _free(model, device)


def run_torch_text(data_dir: str, name: str, model_name: str, max_len: int = 256,
                   batch: int = 32, device: str = "auto") -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    try:
        _torch_text(data_dir, name, model_name, max_len, batch, device)
    except EnvError:
        raise
    except Exception as e:  # noqa: BLE001
        if device == "cpu" or not _is_device_error(e):
            raise                                   # не проблема прискорювача — повтор на CPU нічого не змінить
        logger.warning(f"⚠️ Збій на прискорювачі ({e}). Повтор на CPU...")
        _torch_text(data_dir, name, model_name, max_len, batch, "cpu")


# ------------------------------------------------------------------------------
# Оркестратор (викликається з ноутбука) — підпроцес на кожен екстрактор
# ------------------------------------------------------------------------------
_TQDM = re.compile(r"(\d+)%\|[^|]*\|\s*(\S+)\s*\[([^\]]*)\]")


def _run_streaming(cmd, env, cwd, on_line=None) -> None:
    """Запускає підпроцес, дзеркалить його вивід у термінал і віддає колбеку оновлення tqdm: (pct, count, 'elapsed<eta, rate')."""
    env = {**env, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, universal_newlines=True)       # '\r' tqdm → окремі рядки
    try:
        for line in proc.stdout:
            if line.lstrip().startswith("Loading weights"):
                continue                                # шум від transformers 5.x (сотні рядків)
            sys.stdout.write(line if line.endswith("\n") else line + "\n")
            m = _TQDM.search(line)
            if m and on_line:
                on_line(m.group(1), m.group(2), m.group(3))
    finally:
        proc.stdout.close()
        rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def ensure_all(data_dir: str = da.DATA_DIR, force: bool = False, jobs=None, on_job=None) -> Dict[str, str]:
    """on_job(name, phase, status): phase='start'|'progress'|'done' — для прогрес-бару в ноутбуці
    ('progress' несе рядок tqdm підпроцесу: відсоток, лічильник, elapsed<ETA, швидкість)."""
    data_dir = str(Path(data_dir).resolve())
    root = Path(__file__).resolve().parents[1]
    report: Dict[str, str] = {}
    for name, kind, extra in (jobs or JOBS):
        cache = da.feature_dir(data_dir) / f"{name}.npz"
        framework = "TensorFlow" if kind.startswith("tf") else "PyTorch"
        if on_job:
            on_job(name, "start", framework)
        if cache.exists() and not force:
            report[name] = f"✅ кеш ({framework})"
            if on_job:
                on_job(name, "done", report[name])
            continue
        cmd = [sys.executable, "-m", "hw_fp_kaggle_petfinder.extractors", kind, "--data-dir", data_dir, "--name", name]
        if "model" in extra:
            cmd += ["--model", extra["model"]]
        if "revision" in extra:
            cmd += ["--revision", extra["revision"]]
        env = {**os.environ, "PYTHONPATH": f"{root}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
               "KMP_DUPLICATE_LIB_OK": "TRUE"}
        env["KERAS_BACKEND"] = "tensorflow" if kind.startswith("tf") else "torch"
        logger.info(f"🚀 {name} [{framework}] → {' '.join(cmd[2:4])}")
        try:
            _run_streaming(cmd, env, str(root),
                           (lambda pct, cnt, tm, _n=name: on_job(_n, "progress", f"{pct}% ({cnt}) · {tm}")) if on_job else None)
            report[name] = f"🆕 пораховано ({framework})"
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ {name} не вдалося (код {e.returncode}). Пайплайн продовжить без цього блоку.")
            report[name] = f"❌ збій ({framework})"
        if on_job:
            on_job(name, "done", report[name])
    return report


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    for _noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3", "filelock"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)       # без сотень рядків 'HTTP Request: GET ...'
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["all", "tf-images", "torch-images", "torch-text"])
    ap.add_argument("--data-dir", default=da.DATA_DIR)
    ap.add_argument("--name", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--revision", default=None, help="гілка/PR на Hugging Face (напр. refs/pr/66 із safetensors)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    if a.kind == "all":
        print(ensure_all(a.data_dir, force=a.force))
    elif a.kind == "tf-images":
        run_tf_images(a.data_dir, a.name or "img_tf_effv2b3")
    elif a.kind == "torch-images":
        run_torch_images(a.data_dir, a.name or "img_clip_b32", model_name=a.model or "openai/clip-vit-base-patch32",
                         device=a.device, revision=a.revision)
    else:
        model = a.model or "intfloat/multilingual-e5-base"
        slug = a.name or "txt_" + re.sub(r"[^a-z0-9]+", "_", model.split("/")[-1].lower())
        run_torch_text(a.data_dir, slug, model, device=a.device)


if __name__ == "__main__":
    main()
