import torch.nn as nn

class SpamDetectorMLP(nn.Module):
    """
    Універсальна нейромережа для класифікації тексту.
    Підтримує динамічний input_dim для роботи з розрідженими (BoW, TF-IDF)
    та щільними (Word2Vec) векторами.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout_rate: float = 0.3):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim // 2, 1) # Бінарна класифікація
        )

    def forward(self, x):
        return self.network(x)
