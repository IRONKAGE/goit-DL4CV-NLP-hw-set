import logging
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

logger = logging.getLogger("hw_02.model")

class IntelSceneResNet(nn.Module):
    """
    Трансферне навчання на базі ResNet18 для класифікації 6 природних сцен.
    Заморожує ранні шари згортки та навчає лише фінальні блоки для швидкої адаптації.
    """
    def __init__(self, num_classes: int = 6, freeze_backbone: bool = True):
        super().__init__()

        # 1. Завантаження попередньо навченої моделі
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        # 2. Опціональне "заморожування" ранніх шарів (Feature Extractor)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Розморожуємо лише останній шар згортки (layer4) для Fine-Tuning
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True

        # 3. Модифікація останнього повнозв'язного шару (Linear) під 6 класів
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

        logger.info(f"🏗️️ Модель ініціалізовано: ResNet18 -> Fine-Tuning Layer 4 -> {num_classes} виходів.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямий прохід тензора через мережу (x shape: batch_size, 3, 150, 150)"""
        return self.backbone(x)
