import logging
from collections import Counter
from typing import Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import spacy

logger = logging.getLogger("hw_05.data_adapter")

try:
    spacy_en = spacy.load("en_core_web_sm")
    spacy_el = spacy.load("el_core_news_sm")
except OSError:
    logger.warning("Spacy-моделі не знайдено. Fallback до базового split().")
    spacy_en = spacy_el = None

SOS_TOKEN = 0
EOS_TOKEN = 1
PAD_TOKEN = 2
UNK_TOKEN = 3

class Vocab:
    """Будівельник словників з фільтрацією рідкісних слів (захист від Vocabulary Explosion)."""
    def __init__(self, max_vocab_size: int = 12000, min_freq: int = 2):
        self.word2index = {"<sos>": SOS_TOKEN, "<eos>": EOS_TOKEN, "<pad>": PAD_TOKEN, "<unk>": UNK_TOKEN}
        self.index2word = {0: "<sos>", 1: "<eos>", 2: "<pad>", 3: "<unk>"}
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.counter = Counter()

    def count_sentence(self, sentence: str, spacy_model):
        tokens = [tok.text.lower() for tok in spacy_model.tokenizer(sentence)] if spacy_model else sentence.lower().split()
        self.counter.update(tokens)

    def build_vocab(self):
        most_common = self.counter.most_common(self.max_vocab_size - len(self.word2index))
        for word, count in most_common:
            if count >= self.min_freq and word not in self.word2index:
                idx = len(self.word2index)
                self.word2index[word] = idx
                self.index2word[idx] = word
        self.n_words = len(self.word2index)

class EuroparlDataset(Dataset):
    def __init__(self, data, vocab_en, vocab_el, max_len=30):
        self.data = data
        self.vocab_en = vocab_en
        self.vocab_el = vocab_el
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def _numericalize(self, text, vocab, spacy_model):
        tokens = [tok.text.lower() for tok in spacy_model.tokenizer(text)] if spacy_model else text.lower().split()
        ids = [vocab.word2index.get(w, UNK_TOKEN) for w in tokens[:self.max_len - 2]]
        return [SOS_TOKEN] + ids + [EOS_TOKEN]

    def __getitem__(self, idx):
        pair = self.data[idx]['translation']
        en_seq = self._numericalize(pair['en'], self.vocab_en, spacy_en)
        el_seq = self._numericalize(pair['el'], self.vocab_el, spacy_el)
        return torch.tensor(en_seq, dtype=torch.long), torch.tensor(el_seq, dtype=torch.long)

def collate_fn(batch):
    en_batch, el_batch = zip(*batch)
    en_pad = torch.nn.utils.rnn.pad_sequence(en_batch, padding_value=PAD_TOKEN, batch_first=True)
    el_pad = torch.nn.utils.rnn.pad_sequence(el_batch, padding_value=PAD_TOKEN, batch_first=True)
    return en_pad, el_pad

def get_translation_dataloaders(batch_size: int = 128, subset_size: int = 25000):
    """
    Завантажує оптимізовану підмножину Europarl (en-el).
    25 000 пар речень є оптимальними для тренування без багатоденних очікувань.
    """
    logger.info(f"⏳ Завантаження набору даних Europarl (el-en) — вибірка: {subset_size} пар...")
    dataset = load_dataset("Helsinki-NLP/europarl", "el-en", split="train")

    subset = dataset.select(range(subset_size))
    train_size = int(0.85 * len(subset))
    train_data = subset.select(range(train_size))
    val_data = subset.select(range(train_size, len(subset)))

    logger.info("📚 Побудова словників...")
    vocab_en = Vocab(max_vocab_size=12000, min_freq=2)
    vocab_el = Vocab(max_vocab_size=15000, min_freq=2)

    for row in train_data:
        vocab_en.count_sentence(row['translation']['en'], spacy_en)
        vocab_el.count_sentence(row['translation']['el'], spacy_el)

    vocab_en.build_vocab()
    vocab_el.build_vocab()

    train_ds = EuroparlDataset(train_data, vocab_en, vocab_el)
    val_ds = EuroparlDataset(val_data, vocab_en, vocab_el)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    logger.info(f"✅ Дані готові! Train: {len(train_ds)}, Val: {len(val_ds)} | Vocab EN: {vocab_en.n_words}, EL: {vocab_el.n_words}")
    return train_loader, val_loader, vocab_en, vocab_el
