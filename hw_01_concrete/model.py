# ==============================================================================
# 🧠 MODEL ARCHITECTURE: Concrete Strength Predictor
# ==============================================================================
"""
Архітектура глибокого багатошарового перцептрона (MLP) для задачі регресії.
Приймає 8 фізико-хімічних параметрів бетону і прогнозує його міцність (МПа).
"""

import logging
from typing import List

import torch
import torch.nn as nn

logger = logging.getLogger("hw_01.model")


class ConcreteStrengthMLP(nn.Module):
    """
    Глибока нейронна мережа прямого поширення (Feed-Forward NN).

    Параметри:
        input_dim (int): Кількість вхідних ознак (для набору даних бетону = 8).
        hidden_dims (List[int]): Розмірності прихованих шарів.
        dropout_rate (float): Ймовірність вимкнення нейрона (захист від перенавчання).
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: List[int] = [128, 64, 32],
        dropout_rate: float = 0.2
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Динамічна побудова шарів мережі
        layers = []
        in_features = input_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_features, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            in_features = h_dim

        # Фінальний шар (1 вихід для регресії, без функції активації)
        layers.append(nn.Linear(in_features, 1))

        self.network = nn.Sequential(*layers)

        # Ініціалізація ваг
        self._initialize_weights()
        logger.info(f"🏗️ Модель ініціалізовано: {input_dim} вхідних ознак -> {hidden_dims} -> 1 вихід.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямий прохід тензора через мережу.
        x shape: (batch_size, input_dim) -> output shape: (batch_size, 1)
        """
        return self.network(x)

    def _initialize_weights(self) -> None:
        """
        Kaiming (He) Initialization.
        Оптимальна стратегія ініціалізації ваг для мереж, що використовують ReLU.
        Запобігає згасанню або вибуху градієнтів на перших епохах.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
