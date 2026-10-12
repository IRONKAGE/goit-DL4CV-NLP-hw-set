import os
import re
import logging
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from core.etl_core import SecureDownloader

logger = logging.getLogger("hw_04.data_adapter")
DATA_DIR = "data/hw_04"

def clean_text(text: str) -> str:
    """Очищення тексту: видалення URL, спецсимволів, цифр та зайвих пробілів."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_spam_data(test_size: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Завантажує набір даних Email Spam Detection, нормалізує назви колонок,
    видаляє дублікати (Fix Data Leakage), очищує текст та виконує стратифікований Train/Val Split.
    """
    downloader = SecureDownloader(
        dataset_path="shantanudhakadd/email-spam-detection-dataset-classification",
        data_dir=DATA_DIR,
        zip_name="email_spam.zip"
    )
    downloader.download()
    csv_path = downloader.extract_atomically(target_extensions=('.csv',))[0]

    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='latin-1')

    if 'v1' in df.columns and 'v2' in df.columns:
        df = df.rename(columns={'v1': 'label', 'v2': 'text'})
    elif 'Category' in df.columns and 'Message' in df.columns:
        df = df.rename(columns={'Category': 'label', 'Message': 'text'})

    df = df[['label', 'text']].dropna()

    # Видаляємо дублікати ПЕРЕД обробкою тексту та сплітом (Data Leakage Prevention)
    df = df.drop_duplicates(subset=['text'], keep='first')

    df['clean_text'] = df['text'].apply(clean_text)
    df = df[df['clean_text'].str.len() > 0]
    df['target'] = df['label'].apply(lambda x: 1 if str(x).lower().strip() == 'spam' else 0)

    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df['target']
    )

    logger.info(f"✅ Набір даних підготовлено! Train: {len(train_df)} | Val: {len(val_df)}")
    return train_df, val_df
