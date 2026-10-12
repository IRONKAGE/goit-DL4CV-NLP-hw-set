import os
import sys
import platform
import logging
import torch
import torch.nn as nn
from tqdm import tqdm

from core import BaseTrainer, get_hardware_config, set_global_seed, log_system_info, clear_vram
from hw_05_translation.data_adapter import get_translation_dataloaders, PAD_TOKEN
from hw_05_translation.model import EncoderRNN, AttentionDecoderRNN, Seq2Seq

logger = logging.getLogger("hw_05.trainer")

def get_console_tqdm(iterable, desc: str):
    """Фірмовий tqdm із виводом у sys.stdout"""
    return tqdm(
        iterable,
        desc=desc,
        leave=True,
        file=sys.stdout,
        bar_format="{l_bar}{bar:30}{r_bar}"
    )

class Seq2SeqTrainer(BaseTrainer):
    """Розширений тренажер для Seq2Seq із батч-прогрес-баром"""

    def train_epoch(self, dataloader) -> float:
        self.model.train()
        total_loss = 0.0

        pbar = get_console_tqdm(dataloader, desc="Тренування батчів")
        for src, trg in pbar:
            src, trg = src.to(self.device), trg.to(self.device)
            self.optimizer.zero_grad()

            output = self.model(src, trg, teacher_forcing_ratio=0.5)
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = self.criterion(output, trg)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / len(dataloader)

    @torch.no_grad()
    def validate_epoch(self, dataloader) -> float:
        self.model.eval()
        total_loss = 0.0

        pbar = get_console_tqdm(dataloader, desc="Валідація батчів")
        for src, trg in pbar:
            src, trg = src.to(self.device), trg.to(self.device)
            output = self.model(src, trg, teacher_forcing_ratio=0.0)
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = self.criterion(output, trg)
            total_loss += loss.item()
            pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

        return total_loss / len(dataloader)

def main():
    log_system_info()
    set_global_seed(42)
    device, device_name = get_hardware_config()

    # Селективний fallback: захист від оверхеду Metal на Intel Mac x86_64
    if device.type == "mps" and platform.machine() in ("x86_64", "AMD64"):
        device = torch.device("cpu")
        device_name = "CPU (Fallback з Intel Mac x86_64 для послідовного RNN-циклу)"

    logger.info(f"🚀 Запуск навчання на: {device_name}")

    train_loader, val_loader, vocab_en, vocab_el = get_translation_dataloaders(batch_size=128, subset_size=25000)

    HIDDEN_SIZE = 256
    encoder = EncoderRNN(vocab_en.n_words, HIDDEN_SIZE).to(device)
    decoder = AttentionDecoderRNN(HIDDEN_SIZE, vocab_el.n_words, max_length=35).to(device)
    model = Seq2Seq(encoder, decoder, device).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    trainer = Seq2SeqTrainer(
        model=model, optimizer=optimizer, criterion=criterion, device=device,
        experiment_name="hw_05_translation", run_name="seq2seq_bahdanau_cli"
    )

    trainer.fit(
        train_loader, val_loader,
        epochs=10, patience=3,
        model_save_path="models/hw_05/best_translation_model.pth"
    )

    clear_vram(device)
    logger.info("✅ CLI-пайплайн тренування успішно завершено!")

if __name__ == "__main__":
    main()
