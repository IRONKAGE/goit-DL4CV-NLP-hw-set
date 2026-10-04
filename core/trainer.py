# ==============================================================================
# 🏎️ MLOps CORE: Universal Model Trainer
# ==============================================================================
"""
Універсальний цикл навчання моделей (Оновлення: Жовтень 2026).
Забезпечує стандартизоване тренування PyTorch-моделей з автоматичним
логуванням у MLflow, механізмом Early Stopping та збереженням артефактів.
"""

import os
import time
import logging
from typing import Dict, Optional, Tuple, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
from tqdm import tqdm

logger = logging.getLogger("core.trainer")

# ------------------------------------------------------------------------------
# 1. EARLY STOPPING (Захист від перенавчання)
# ------------------------------------------------------------------------------
class EarlyStopping:
    """
    Зупиняє навчання, якщо метрика валідації не покращується протягом заданої кількості епох.
    """
    def __init__(self, patience: int = 5, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_metric = None
        self.early_stop = False

    def __call__(self, current_metric: float) -> bool:
        if self.best_metric is None:
            self.best_metric = current_metric
            return False

        if self.mode == "min":
            improvement = self.best_metric - current_metric > self.min_delta
        else:
            improvement = current_metric - self.best_metric > self.min_delta

        if improvement:
            self.best_metric = current_metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"🛑 Спрацював Early Stopping! Метрика не покращувалась {self.patience} епох.")

        return self.early_stop

# ------------------------------------------------------------------------------
# 2. УНІВЕРСАЛЬНИЙ ТРЕНЕР (BaseTrainer)
# ------------------------------------------------------------------------------
class BaseTrainer:
    """
    Апаратно-агностичний клас для навчання PyTorch моделей.
    Інтегровано з MLflow для трекінгу експериментів (Zero-Config).
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Callable,
        device: torch.device,
        experiment_name: str = "default_experiment",
        run_name: str = "base_run"
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.experiment_name = experiment_name
        self.run_name = run_name

        # Налаштування MLflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns/mlruns.db"))
        mlflow.set_experiment(self.experiment_name)

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Виконує одну епоху навчання."""
        self.model.train()
        total_loss = 0.0

        progress_bar = tqdm(dataloader, desc="Тренування", leave=False)
        for batch_x, batch_y in progress_bar:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

            self.optimizer.zero_grad()
            # Вирівнюємо розмірність [batch_size, 1] -> [batch_size] для MSELoss
            predictions = self.model(batch_x).squeeze(-1)
            loss = self.criterion(predictions, batch_y)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / len(dataloader)

    @torch.no_grad()
    def validate_epoch(self, dataloader: DataLoader) -> float:
        """Виконує одну епоху валідації (без обчислення градієнтів)."""
        self.model.eval()
        total_loss = 0.0

        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            predictions = self.model(batch_x).squeeze(-1)
            loss = self.criterion(predictions, batch_y)
            total_loss += loss.item()

        return total_loss / len(dataloader)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        patience: int = 5,
        model_save_path: str = "models/best_model.pth",
        epoch_callback: Callable = None
    ) -> Dict[str, list]:
        """
        Головний цикл навчання з автоматичним логуванням у MLflow та збереженням артефактів.
        """
        early_stopping = EarlyStopping(patience=patience, mode="min")
        history = {"train_loss": [], "val_loss": []}

        # Забезпечуємо існування директорії для моделей
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

        with mlflow.start_run(run_name=self.run_name):
            # Логування базових гіперпараметрів
            mlflow.log_params({
                "epochs": epochs,
                "batch_size": train_loader.batch_size,
                "optimizer": self.optimizer.__class__.__name__,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "device": self.device.type
            })

            logger.info(f"🚀 Початок навчання на пристрої: {self.device.type.upper()}")
            start_time = time.time()

            for epoch in range(1, epochs + 1):
                logger.info(f"Епоха {epoch}/{epochs}")

                # Тренування та Валідація
                train_loss = self.train_epoch(train_loader)
                val_loss = self.validate_epoch(val_loader)

                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)

                # Логування метрик у MLflow в реальному часі
                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "val_loss": val_loss
                }, step=epoch)

                logger.info(f"📉 Train Loss: {train_loss:.4f} | 📈 Val Loss: {val_loss:.4f}")

                # Передаємо метрики в Marimo UI, якщо callback передано
                if epoch_callback:
                    epoch_callback(epoch, train_loss, val_loss)

                # Перевірка Early Stopping та збереження найкращої моделі
                if early_stopping.best_metric is None or val_loss < early_stopping.best_metric:
                    logger.info(f"💾 Знайдено нову найкращу модель! Збереження у {model_save_path}")
                    torch.save(self.model.state_dict(), model_save_path)

                if early_stopping(val_loss):
                    break

            training_time = (time.time() - start_time) / 60
            logger.info(f"✅ Навчання завершено за {training_time:.2f} хвилин.")

            # Серіалізація фінальної моделі в MLflow (Graceful Degradation)
            try:
                self.model.load_state_dict(torch.load(model_save_path, weights_only=True))

                success = False
                # Перебираємо формати від найсучаснішого до базового
                for fmt in ["pt2", "pickle"]:
                    try:
                        mlflow.pytorch.log_model(
                            self.model,
                            artifact_path="model",
                            serialization_format=fmt
                        )
                        logger.info(f"📦 Модель успішно залогована в MLflow Registry (формат: {fmt}).")
                        success = True
                        break  # Виходимо з циклу при першому успішному збереженні
                    except Exception as fmt_err:
                        logger.debug(f"Фолбек серіалізації: формат '{fmt}' недоступний. Причина: {fmt_err}")

                if not success:
                    logger.warning("⚠ Не вдалося залогувати модель в MLflow: жоден формат серіалізації не підтримується у поточному середовищі.")

            except Exception as mlf_err:
                logger.warning(f"⚠ Критична помилка під час роботи з MLflow: {mlf_err}")

        return history
