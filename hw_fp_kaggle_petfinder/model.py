"""
🧠 KERAS 3 (бекенд PyTorch) — Late-Fusion мультимодальна голова поверх КЕШОВАНИХ ембедингів.

Для чого Keras: швидко збирати/експериментувати з багатовходовою архітектурою
(Functional API), тоді як важкі «очі» (TF EfficientNetV2, PyTorch CLIP) та «мова»
(PyTorch multilingual трансформери) вже пораховані екстракторами.

Кожен блок (img_tf, img_clip, txt_e5, txt_minilm, tfidf_svd, tabular) має власну гілку;
їх конкатенація → MLP → 1 нейрон (регресія під QWK). Тренування:
фіксований розклад (CosineDecay) без early-stopping на валідації → чисті OOF-прогнози.
"""
import os

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("KERAS_TORCH_DEVICE", "cpu")   # MLP на 6к рядків швидший на CPU; AMD-MPS тут лише ризик

# Keras ≤3.3 ігнорує KERAS_TORCH_DEVICE і сам обирає MPS (AMD GPU), де `aten::_foreach_mul_` у оптимізаторі
# падає на CPU при кожному кроці → ~60 с на епоху для крихітної MLP. Тому ховаємо MPS від Keras ДО його імпорту.
if os.environ.get("KERAS_TORCH_DEVICE", "cpu") == "cpu":
    try:
        import torch as _torch

        _torch.backends.mps.is_available = lambda: False
    except Exception:  # noqa: BLE001
        pass

import gc
import logging
import sys
from typing import Dict, Optional, Tuple

import numpy as np

import torch
import keras
from keras import layers
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from tqdm import tqdm

logger = logging.getLogger("hw_fp.model")


class _EpochHook(keras.callbacks.Callback):
    """Прогрес по епохах: консольний tqdm (stdout, як у попередній версії) + колбек для UI marimo."""

    def __init__(self, tag: str, epochs: int, on_epoch=None, console: bool = True):
        super().__init__()
        self._tag, self._epochs, self._on_epoch, self._console = tag, epochs, on_epoch, console
        self._pbar = None

    def on_train_begin(self, logs=None):
        if self._console:
            self._pbar = tqdm(total=self._epochs, desc=self._tag, unit="ep", file=sys.stdout, leave=False)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self._pbar is not None:
            self._pbar.update(1)
            self._pbar.set_postfix(loss=f"{logs.get('loss', float('nan')):.4f}", val=f"{logs.get('val_loss', float('nan')):.4f}")
        if self._on_epoch:
            self._on_epoch(self._tag, epoch, self._epochs, logs)

    def on_train_end(self, logs=None):
        if self._pbar is not None:
            self._pbar.close()


def prepare_blocks(emb_blocks: Dict[str, np.ndarray], tab_nn, tfidf_svd: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Стандартизація кожного блоку (unsupervised, по train+test — це дозволено, таргет не використовується)."""
    blocks: Dict[str, np.ndarray] = {}
    for name, X in emb_blocks.items():
        blocks[name] = np.clip(StandardScaler().fit_transform(X), -6, 6).astype(np.float32)
    if tfidf_svd is not None:
        blocks["tfidf_svd"] = np.clip(StandardScaler().fit_transform(tfidf_svd), -6, 6).astype(np.float32)
    tab = np.asarray(tab_nn, dtype=np.float32)
    qt = QuantileTransformer(n_quantiles=min(200, len(tab)), output_distribution="normal", random_state=0)
    blocks["tabular"] = np.clip(qt.fit_transform(tab), -5, 5).astype(np.float32)
    return blocks


def build_fusion_model(block_dims: Dict[str, int], y_mean: float, total_steps: int,
                       hidden: int = 256, drop: float = 0.4, lr: float = 1e-3, wd: float = 1e-2) -> keras.Model:
    inputs, branches = [], []
    for name, dim in block_dims.items():
        inp = layers.Input(shape=(dim,), name=name)
        x = layers.GaussianNoise(0.1, name=f"{name}_noise")(inp)
        x = layers.Dropout(0.2, name=f"{name}_indrop")(x)
        x = layers.Dense(hidden, kernel_regularizer=keras.regularizers.l2(1e-5), name=f"{name}_dense")(x)
        x = layers.LayerNormalization(name=f"{name}_ln")(x)
        x = layers.Activation("gelu", name=f"{name}_act")(x)
        x = layers.Dropout(drop, name=f"{name}_drop")(x)
        inputs.append(inp)
        branches.append(x)

    h = branches[0] if len(branches) == 1 else layers.Concatenate(name="fusion")(branches)
    h = layers.Dense(hidden, name="head_dense1")(h)
    h = layers.LayerNormalization(name="head_ln1")(h)
    h = layers.Activation("gelu", name="head_act1")(h)
    h = layers.Dropout(drop, name="head_drop1")(h)
    h = layers.Dense(64, activation="gelu", name="head_dense2")(h)
    out = layers.Dense(1, bias_initializer=keras.initializers.Constant(float(y_mean)), name="speed")(h)

    model = keras.Model(inputs, out)
    sched = keras.optimizers.schedules.CosineDecay(lr, decay_steps=max(total_steps, 1), alpha=0.02)
    model.compile(optimizer=keras.optimizers.AdamW(learning_rate=sched, weight_decay=wd, clipnorm=1.0), loss="mse")
    return model


def keras_fusion_oof(blocks: Dict[str, np.ndarray], y: np.ndarray, folds: np.ndarray, n_train: int,
                     epochs: int = 30, batch: int = 64, n_seeds: int = 3, on_fold=None, on_epoch=None, on_fit=None,
                     history_out=None, console: bool = True, **kw) -> Tuple[np.ndarray, np.ndarray]:
    # Мала MLP: забагато потоків/конфлікт OpenMP-рантаймів на Intel Mac може давати десятки секунд на епоху.
    # Кількість потоків керується змінною FP_TORCH_THREADS (за замовчуванням 4).
    try:
        torch.set_num_threads(int(os.getenv("FP_TORCH_THREADS", "4")))
    except Exception:  # noqa: BLE001
        pass
    names = list(blocks)
    dims = {n: blocks[n].shape[1] for n in names}
    Xtr = {n: blocks[n][:n_train] for n in names}
    Xte = {n: blocks[n][n_train:] for n in names}
    K = int(folds.max()) + 1
    oof, test = np.zeros(n_train), np.zeros(len(blocks[names[0]]) - n_train)
    try:
        _dev = getattr(keras.ops.convert_to_tensor([1.0]), "device", "?")
    except Exception:  # noqa: BLE001
        _dev = "?"
    logger.info(f"🧠 Keras {keras.__version__} ({keras.backend.backend()}) | пристрій тензорів: {_dev} | блоки: {dims}")

    for k in tqdm(range(K), desc="Keras fusion (фолди)", file=sys.stdout):
        tr, va = np.where(folds != k)[0], np.where(folds == k)[0]
        pv, pt = np.zeros(len(va)), np.zeros_like(test)
        for s in range(n_seeds):
            keras.utils.set_random_seed(1000 * k + s)
            steps = int(np.ceil(len(tr) / batch)) * epochs
            m = build_fusion_model(dims, float(y[tr].mean()), steps, **kw)
            # validation_data — ЛИШЕ для моніторингу кривих (без early-stopping/відбору) → OOF лишається чистим
            h = m.fit({n: Xtr[n][tr] for n in names}, y[tr], batch_size=batch, epochs=epochs, verbose=0, shuffle=True,
                      validation_data=({n: Xtr[n][va] for n in names}, y[va]),
                      callbacks=[_EpochHook(f"фолд {k + 1}/{K} · seed {s + 1}/{n_seeds}", epochs, on_epoch, console)])
            if history_out is not None:
                history_out.append({"fold": k, "seed": s, "loss": h.history["loss"], "val_loss": h.history["val_loss"]})
            if on_fit:
                on_fit(k, s, K, n_seeds)
            pv += m.predict({n: Xtr[n][va] for n in names}, batch_size=512, verbose=0).ravel() / n_seeds
            pt += m.predict(Xte, batch_size=512, verbose=0).ravel() / n_seeds
            del m
            keras.backend.clear_session()
            gc.collect()
        oof[va], test = pv, test + pt / K
        _rmse = float(np.sqrt(np.mean((pv - y[va]) ** 2)))
        logger.info(f"   fold {k}: RMSE={_rmse:.4f}")
        if on_fold:
            on_fold(k, K, _rmse)
    return oof, test
