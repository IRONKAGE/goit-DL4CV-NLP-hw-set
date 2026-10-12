import sys
import logging
import numpy as np
import torch
import gensim.downloader as api
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

logger = logging.getLogger("hw_04.nlp_core")

# Налаштовуємо tqdm для гарантованого виводу в консоль
def get_tqdm(iterable, desc):
    return tqdm(
        iterable,
        desc=desc,
        leave=True,
        file=sys.stdout,
        bar_format="{l_bar}{bar:30}{r_bar}"
    )

class Word2VecEmbedder:
    """
    Векторизатор тексту на основі попередньо навчених ембедингів Word2Vec/GloVe.
    Використовує підхід Mean Pooling (усереднення векторів слів у повідомленні).
    """
    def __init__(self, model_name: str = "glove-twitter-50"):
        logger.info(f"⏳ Завантаження векторів {model_name}...")
        self.wv = api.load(model_name)
        self.vector_size = self.wv.vector_size
        logger.info("✅ Вектори GloVe успішно завантажено в пам'ять!")

    def transform(self, texts: list, desc: str = "GloVe Pooling") -> np.ndarray:
        """Перетворює список сирих текстів у матрицю ознак розмірності (N, vector_size)."""
        doc_vectors = []
        for text in get_tqdm(texts, desc=desc):
            words = str(text).split()
            word_vecs = [self.wv[w] for w in words if w in self.wv]
            if len(word_vecs) > 0:
                doc_vectors.append(np.mean(word_vecs, axis=0))
            else:
                doc_vectors.append(np.zeros(self.vector_size))
        return np.vstack(doc_vectors)


class FastTextEmbedder:
    """
    Векторизатор на базі FastText.
    Стійкий до спамерських помилок та спотворень (OOV) завдяки символьним n-грамам.
    """
    def __init__(self, model_name: str = "fasttext-wiki-news-subwords-300"):
        logger.info(f"⏳ Завантаження FastText {model_name} (може зайняти час)...")
        self.wv = api.load(model_name)
        self.vector_size = self.wv.vector_size
        logger.info("✅ Словник FastText успішно завантажено в пам'ять!")

    def transform(self, texts: list, desc: str = "FastText Pooling") -> np.ndarray:
        doc_vectors = []
        for text in get_tqdm(texts, desc=desc):
            words = str(text).split()
            # FastText генерує вектори навіть для незнайомих слів (OOV)
            word_vecs = [self.wv[w] for w in words if w in self.wv or self.wv.has_index_for(w)]
            if len(word_vecs) > 0:
                doc_vectors.append(np.mean(word_vecs, axis=0))
            else:
                doc_vectors.append(np.zeros(self.vector_size))
        return np.vstack(doc_vectors)


class TransformerEmbedder:
    """
    SOTA Векторизатор (2026). Генерує щільні контекстуальні вектори речень.
    Використовує HuggingFace Transformers та механізм Attention Pooling.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        logger.info(f"⏳ Ініціалізація Transformer-моделі {model_name} на {device.upper()}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device
        self.model.eval()
        logger.info("✅ Трансформер MiniLM успішно ініціалізовано!")

    def transform(self, texts: list, batch_size: int = 32, desc: str = "MiniLM Attention") -> np.ndarray:
        # Примусово конвертуємо вхідний масив у чистий Python-список рядків
        # (це необхідно для HuggingFace Tokenizer)
        texts = [str(t) for t in texts]

        all_embeddings = []
        with torch.no_grad():
            for i in get_tqdm(range(0, len(texts), batch_size), desc=desc):
                batch_texts = texts[i:i+batch_size]

                # Токенізація з паддінгом та обрізкою
                encoded = self.tokenizer(
                    batch_texts, padding=True, truncation=True,
                    max_length=128, return_tensors='pt'
                ).to(self.device)

                # Прохід через трансформер
                model_output = self.model(**encoded)

                # Mean Pooling (з урахуванням Attention Mask)
                attention_mask = encoded['attention_mask']
                token_embeddings = model_output[0]
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                batch_embeddings = sum_embeddings / sum_mask

                all_embeddings.append(batch_embeddings.cpu().numpy())

        return np.vstack(all_embeddings)
