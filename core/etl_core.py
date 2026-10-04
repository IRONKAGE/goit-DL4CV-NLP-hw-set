import os
import zipfile
import shutil
import urllib.request
import urllib.error
import sys
import logging
import pandas as pd
import time
import hashlib
from functools import wraps
from tqdm import tqdm

# =========================================================================
# 📝 НАЛАШТУВАННЯ ЛОГУВАННЯ (Console + File Audit Trail)
# =========================================================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False  # Заборона дублювання логів

# Форматувальник для логів
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Хендлер для виводу в консоль (sys.stdout, щоб не конфліктувати з tqdm)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(stream_handler)

# Хендлер для запису в файл (Audit Trail для нічних збоїв)
file_handler = logging.FileHandler('etl_pipeline.log', encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Спробуємо імпортувати Kaggle API. Якщо бібліотеки немає, ми просто перейдемо на urllib
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    KAGGLE_LIB_AVAILABLE = True
except ImportError:
    KAGGLE_LIB_AVAILABLE = False


# =========================================================================
# 🔄 ДЕКОРАТОР EXPONENTIAL BACKOFF (Захист від переривання інтернету)
# =========================================================================
def retry_with_backoff(retries=3, backoff_in_seconds=2):
    """Декоратор: повторює виконання функції при мережевих збоях зі збільшенням інтервалу."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Якщо це помилка 416 (Range Not Satisfiable), ми не робимо backoff,
                    # бо вона обробляється специфічно всередині функції (видаленням файлу)
                    if isinstance(e, urllib.error.HTTPError) and e.code == 416:
                        raise e

                    if attempt >= retries:
                        logger.error(f"❌ Ліміт спроб вичерпано ({retries}). Помилка: {e}")
                        raise e

                    sleep_time = (backoff_in_seconds * 2 ** attempt)
                    logger.warning(f"⏳ Мережевий збій: {e}. Повторна спроба {attempt+1}/{retries} через {sleep_time} сек...")
                    time.sleep(sleep_time)
                    attempt += 1
        return wrapper
    return decorator


# =========================================================================
# 📦 КЛАС ЗАВАНТАЖУВАЧА (SMART ROUTER 5-TIER)
# =========================================================================
class SecureDownloader:
    def __init__(self, dataset_path=None, dataset_url=None, kaggle_direct_url=None, gdrive_id=None, fallback_generator=None, data_dir=None, zip_name="dataset_archive.zip", expected_size=None, expected_sha256=None):
        """
        Ініціалізує гібридний завантажувач (Kaggle API -> GDrive -> urllib -> scikit-learn або інший аналог)
        dataset_path: шлях для API Kaggle (напр. 'Cornell-University/arxiv')
        dataset_url: пряме посилання для urllib (напр. S3 bucket url)
        kaggle_direct_url: пряме посилання Kaggle API без ключів
        gdrive_id: ID файлу на Google Drive
        fallback_generator: функція для локальної генерації даних (Fallback)
        expected_size: еталонний розмір архіву в байтах (для базового захисту від підміни)
        expected_sha256: криптографічний хеш для параноїдального рівня безпеки
        """
        self.data_dir = data_dir or os.getenv("DATA_DIR", "./data")
        self.dataset_path = dataset_path
        self.dataset_url = dataset_url
        self.kaggle_direct_url = kaggle_direct_url or (f"https://www.kaggle.com/api/v1/datasets/download/{dataset_path}" if dataset_path else None)
        self.gdrive_id = gdrive_id
        self.fallback_generator = fallback_generator
        self._internal_chunk_seed = 0x49524f4e4b414745
        self.zip_path = os.path.join(self.data_dir, zip_name)
        self.expected_size = expected_size
        self.expected_sha256 = expected_sha256
        self._is_zip_valid_cache = None     # Cache: щоб не рахувати CRC32/SHA256 двічі!

        if self.data_dir != ".":
            os.makedirs(self.data_dir, exist_ok=True)

    def _verify_sha256(self):
        """Глибока перевірка криптографічного підпису набору даних (захист від MITM та Spoofing)"""
        if not self.expected_sha256:
            return True # Якщо хеш не передано, ігноруємо

        logger.info("🔐 Перевірка криптографічного хешу SHA-256...")
        sha256_hash = hashlib.sha256()

        # Читаємо файл шматками по 4 МБ, щоб не забити оперативну пам'ять
        with open(self.zip_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        actual_hash = sha256_hash.hexdigest()
        if actual_hash != self.expected_sha256:
            logger.error(f"❌ SHA-256 не збігається!\nОчікувано: {self.expected_sha256}\nФакт: {actual_hash}")
            return False

        logger.info("✅ Криптографічний хеш ідеально збігається!")
        return True

    def is_valid_zip(self):
        """Комплексна перевірка цілісності ZIP-архіву..."""
        # Якщо ми вже перевіряли архів у цьому циклі, повертаємо результат миттєво
        if self._is_zip_valid_cache is not None:
            return self._is_zip_valid_cache

        # 1. Перевірка наявності та магічних байтів ZIP-формату
        if not os.path.exists(self.zip_path) or not zipfile.is_zipfile(self.zip_path):
            self._is_zip_valid_cache = False
            return False

        # 2. Перевірка фізичного розміру (Анти-жартівник)
        if self.expected_size and os.path.getsize(self.zip_path) != self.expected_size:
            logger.warning(f"⚠️ Розмір архіву не збігається з еталоном (Факт: {os.path.getsize(self.zip_path)}, Очікувалось: {self.expected_size})!")
            self._is_zip_valid_cache = False
            return False

        # 3. Перевірка цілісності CRC32 (Глибока перевірка на побиті файли)
        try:
            with zipfile.ZipFile(self.zip_path, 'r') as z:
                if z.testzip() is not None:
                    logger.warning("⚠️ Архів пошкоджено (не збігається контрольна сума CRC32)!")
                    self._is_zip_valid_cache = False
                    return False
        except Exception as e:
            logger.warning(f"⚠️ Помилка читання архіву: {e}")
            self._is_zip_valid_cache = False
            return False

        # 4. Перевірка криптографічного хешу SHA-256 (якщо задано)
        if not self._verify_sha256():
            self._is_zip_valid_cache = False
            return False

        self._is_zip_valid_cache = True
        return True

    def _download_via_api(self):
        """Рівень 1: Внутрішній метод для завантаження через офіційне API Kaggle"""
        logger.info("🤖 Ініціалізація офіційного Kaggle API...")
        logger.info(f"⏳ Завантаження набору даних '{self.dataset_path}' у '{self.data_dir}'...")

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(self.dataset_path, path=self.data_dir, unzip=False)

        # ⚡ БЕЗПЕЧНЕ ПЕРЕЙМЕНУВАННЯ (Знаходимо найновіший zip-архів)
        zip_files = [
            os.path.join(self.data_dir, f)
            for f in os.listdir(self.data_dir)
            if f.endswith('.zip') and f != os.path.basename(self.zip_path)
        ]

        if zip_files:
            # Знаходимо файл, який був створений/змінений останнім (наш щойно скачаний)
            latest_zip = max(zip_files, key=os.path.getmtime)

            # Захист від FileExistsError перед перейменуванням
            if os.path.exists(self.zip_path):
                os.remove(self.zip_path)

            os.rename(latest_zip, self.zip_path)

    def _download_via_gdrive(self):
        """Рівень 2: Внутрішній метод для завантаження з Google Drive"""
        logger.info(f"📁 Виявлено Google Drive ID. Ініціалізація gdown...")
        try:
            import gdown
            url = f'https://drive.google.com/uc?id={self.gdrive_id}'
            gdown.download(url, self.zip_path, quiet=False)
        except ImportError:
            logger.warning("⚠️ Пакет 'gdown' не встановлено. Запустіть 'pip install gdown'")
            raise Exception("Бібліотека gdown відсутня...")

    @retry_with_backoff(retries=4, backoff_in_seconds=3)
    def _download_via_urllib(self, target_url=None):
        """Рівень 3/4: Внутрішній метод для завантаження через urllib із підтримкою HTTP Range (дозавантаження)"""
        url_to_download = target_url or self.dataset_url
        if not url_to_download:
            raise Exception("Fallback URL не вказано!")

        existing_size = 0
        if os.path.exists(self.zip_path):
            existing_size = os.path.getsize(self.zip_path)

        req = urllib.request.Request(url_to_download, headers={'User-Agent': 'Mozilla/5.0'})

        # Якщо файл вже частково завантажено, просимо сервер віддати лише залишок
        if existing_size > 0:
            req.add_header('Range', f'bytes={existing_size}-')
            logger.info(f"🌐 Відновлення завантаження (urllib) з {existing_size / 1024 / 1024:.1f} MB...")
        else:
            logger.info("🌐 Ініціалізація прямого завантаження (urllib)...")

        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                content_length = int(response.headers.get('Content-Length', -1))

                # Обробка 206 Partial Content (Сервер підтримує дозавантаження)
                if response.status == 206:
                    expected_total = existing_size + content_length if content_length != -1 else -1
                    mode = 'ab'  # Append binary (дозапис)
                    downloaded_bytes = existing_size
                    logger.info("🔄 Сервер підтримує Range-запити. Продовжуємо завантаження...")
                else:
                    expected_total = content_length
                    mode = 'wb'  # Write binary (з нуля)
                    downloaded_bytes = 0
                    if existing_size > 0:
                        logger.warning("⚠️ Сервер не підтримує відновлення (Range). Починаємо завантаження з нуля...")

                block_size = 8192

                with open(self.zip_path, mode) as out_file:
                    # Ініціалізація tqdm (автоматично малює прогрес-бар, розраховує швидкість та ETA)
                    with tqdm(
                        total=expected_total if expected_total != -1 else None,
                        initial=downloaded_bytes,
                        unit='iB',
                        unit_scale=True,
                        unit_divisor=1024,
                        desc="📥 Завантаження",
                        file=sys.stdout
                    ) as pbar:
                        while True:
                            buffer = response.read(block_size)
                            if not buffer:
                                break
                            out_file.write(buffer)
                            pbar.update(len(buffer))

            # Завантаження завершено, скидаємо кеш перевірки
            self._is_zip_valid_cache = None

        except urllib.error.HTTPError as e:
            # 416 означає, що локальний файл більший або не збігається з сервером
            if e.code == 416:
                logger.warning("⚠️ Помилка 416 (Range Not Satisfiable). Локальний файл конфліктує із сервером. Видаляємо та качаємо наново...")
                if os.path.exists(self.zip_path):
                    os.remove(self.zip_path)
                self._download_via_urllib(target_url=url_to_download)  # Рекурсивний ретрай з нуля
            else:
                raise e

    def _execute_fallback_generator(self, target_filename):
        """Рівень 5: Виконання користувацької функції генерації даних (Dependency Injection)"""
        logger.info("🧪 Активація користувацького генератора даних (Fallback)...")
        try:
            target_path = os.path.join(self.data_dir, target_filename)
            # Викликаємо передану з ноутбука функцію
            self.fallback_generator(target_path)
            logger.info(f"✅ Локальний набір даних згенеровано успішно: {target_path}")
        except Exception as e:
            raise Exception(f"Помилка у користувацькому генераторі: {e}")

    def download(self, target_filename="data.csv"):
        """Головний метод завантаження з розумним маршрутизатором (Smart Router 5-Tier)"""
        logger.info("🔍 Перевірка локальних файлів...")

        # 1. Idempotency: Якщо цільовий файл вже є, нічого не качаємо
        target_file = os.path.join(self.data_dir, target_filename)
        if os.path.exists(target_file):
            logger.info(f"🔋 Знайдено готовий файл: {target_filename}. Пропускаємо завантаження...")
            return

        # 2. Idempotency: Якщо архів вже є і він цілий
        if os.path.exists(self.zip_path):
            if self.is_valid_zip():
                logger.info("🔋 Архів цілий. Пропускаємо мережевий запит...")
                return
            else:
                logger.warning("🪫 Архів неповний або пошкоджений. Спроба відновлення завантаження...")

        download_success = False
        provided_channels = 0  # Лічильник вказаних користувачем каналів

        # 🥇 Спроба 1: Офіційне API Kaggle (Найвищий пріоритет)
        if hasattr(self, 'dataset_path') and self.dataset_path:
            provided_channels += 1

            k_user = os.getenv("KAGGLE_USERNAME", "").strip()
            k_key = os.getenv("KAGGLE_KEY", "").strip()

            # 🛡️ СУВОРА ВАЛІДАЦІЯ КЛЮЧІВ (Sanity Check 2.0)
            # Справжній Kaggle Key - це завжди 32-символьний буквено-цифровий рядок
            is_valid_format = (
                bool(k_user) and
                len(k_key) == 32 and
                k_key.isalnum()
            )

            if is_valid_format and KAGGLE_LIB_AVAILABLE:
                try:
                    self._download_via_api()
                    self._is_zip_valid_cache = None # Скидаємо кеш
                    if self.is_valid_zip():
                        download_success = True
                except Exception as e:
                    logger.warning(f"⚠️ Помилка Kaggle API: {e}. Перехід на публічний ендпоінт...")
            else:
                if k_user or k_key:
                    logger.info("⏩ Пропуск офіційного Kaggle API (ключі відсутні або мають недійсний формат)...")

        # 🥈 Спроба 2: Пряме посилання Kaggle (v1, без ключів) - Логічне продовження Спроби 1
        if not download_success and hasattr(self, 'kaggle_direct_url') and self.kaggle_direct_url:
            provided_channels += 1
            try:
                logger.info("⚡ Спроба анонімного завантаження через прямий ендпоінт Kaggle URL (v1)...")
                self._download_via_urllib(target_url=self.kaggle_direct_url)
                if self.is_valid_zip():
                    download_success = True
            except Exception as e:
                logger.warning(f"⚠️ Помилка прямого Kaggle URL: {e}. Перехід на інші хмарні сервіси...")

        # 🥉 Спроба 3: Google Drive (Зовнішня хмара)
        if not download_success and hasattr(self, 'gdrive_id') and self.gdrive_id:
            provided_channels += 1
            try:
                self._download_via_gdrive()
                self._is_zip_valid_cache = None
                if self.is_valid_zip():
                    download_success = True
            except Exception as e:
                logger.warning(f"⚠️ Помилка завантаження з GDrive: {e}. Перехід на Fallback...")

        # 🛡️ Спроба 4: Fallback URL (S3 / GitHub / GroupLens)
        if not download_success and hasattr(self, 'dataset_url') and self.dataset_url:
            provided_channels += 1
            try:
                logger.info("🏛️ Спроба завантаження з резервного джерела (Fallback)...")
                self._download_via_urllib(target_url=self.dataset_url)
                if self.is_valid_zip():
                    download_success = True
            except Exception as e:
                logger.warning(f"⚠️ Помилка резервного джерела: {e}")

        # 🛟 Спроба 5: Користувацький генератор (Dependency Injection)
        if not download_success and hasattr(self, 'fallback_generator') and self.fallback_generator:
            provided_channels += 1
            logger.warning("🛟 Всі мережеві завантаження провалилися. Активація користувацького генератора...")
            try:
                self._execute_fallback_generator(target_filename)
                download_success = True
            except Exception as e:
                logger.warning(f"⚠️ Помилка генерації через кастомну функцію: {e}")

        # 🛑 РОЗУМНІ ПОМИЛКИ ТА ФІНАЛЬНА ПЕРЕВІРКА:
        if provided_channels == 0:
            raise ValueError("❌ Жодного джерела даних не вказано! Передайте хоча б один параметр (dataset_path, dataset_url, kaggle_direct_url, gdrive_id, fallback_generator) в SecureDownloader")

        if not download_success:
            # 🧹 Видаляємо пошкоджений файл перед тим, як впасти
            if os.path.exists(self.zip_path):
                os.remove(self.zip_path)
            raise Exception(f"❌ Критична помилка: Усі вказані канали ({provided_channels} шт.) виявилися недоступними...")
        elif os.path.exists(self.zip_path) and self.is_valid_zip():
            logger.info("✅ Завантаження архіву завершено успішно!")

    def extract_atomically(self, target_extensions=('.csv', '.json', '.dat'), expected_filename="data.csv"):
        """Атомарне розпакування з Flattening (вирівнюванням директорій) та TQDM прогресом"""
        # Логіка ідемпотентності
        target_file = os.path.join(self.data_dir, expected_filename)
        if os.path.exists(target_file):
            return [target_file]

        if not self.is_valid_zip():
            raise Exception("❌ Критична помилка: Архів відсутній або пошкоджений")

        extracted_files = []
        logger.info(f"📦 Аналізуємо вміст архіву...")

        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            data_files = [f for f in zip_ref.namelist() if f.endswith(target_extensions)]

            if not data_files:
                raise Exception(f"В архіві немає файлів з розширеннями {target_extensions}!")

            for file_in_zip in data_files:
                # Flattening: ігноруємо вкладені папки всередині ZIP
                final_path = os.path.join(self.data_dir, os.path.basename(file_in_zip))
                tmp_extract_path = final_path + ".tmp_extract"

                if os.path.exists(final_path):
                    logger.info(f"⚡ Файл '{os.path.basename(final_path)}' вже готовий. Пропускаємо...")
                    extracted_files.append(final_path)
                    continue

                try:
                    # Отримуємо розмір стиснутого файлу для прогрес-бару
                    file_size = zip_ref.getinfo(file_in_zip).file_size

                    with zip_ref.open(file_in_zip) as source, open(tmp_extract_path, "wb") as target:
                        # Інтеграція TQDM для процесу розпакування
                        with tqdm(total=file_size, unit='iB', unit_scale=True, desc=f"   ⚙️ Витягуємо {os.path.basename(file_in_zip)}", file=sys.stdout) as pbar:
                            while True:
                                buffer = source.read(8192)
                                if not buffer:
                                    break
                                target.write(buffer)
                                pbar.update(len(buffer))

                    os.replace(tmp_extract_path, final_path) # Атомарний коміт на диск
                    extracted_files.append(final_path)

                except Exception as extract_err:
                    raise Exception(f"Помилка фізичного запису на диск: {extract_err}")
                finally:
                    if os.path.exists(tmp_extract_path):
                        os.remove(tmp_extract_path)

        logger.info("✅ Успіх! Файли витягнуто безпечно!")
        return extracted_files


# =========================================================================
# 🧬 SMART READER: ПАТЕРН GRACEFUL DEGRADATION & SELF-HEALING (DRY)
# =========================================================================
def smart_read_csv(file_path, desc_name, **kwargs):
    """
    Універсальний DRY-рідер (Enterprise MLOps Standard)
    Забезпечує 3-рівневу стійкість: Engine Routing -> Encoding Fallback -> Data Healing
    """
    # У Pandas існує лише 3 офіційні рушії, сортуємо від найшвидшого до найгнучкішого
    engines = ["pyarrow", "c", "python"]
    last_error = None

    for engine in engines:
        attempt_kwargs = kwargs.copy()
        attempt_kwargs["engine"] = engine

        # 🛡️ 1. Parameter Sanitization (Очищення несумісних аргументів)
        if engine == "python":
            attempt_kwargs.pop("float_precision", None)
            attempt_kwargs.pop("low_memory", None)
        elif engine == "pyarrow":
            attempt_kwargs.pop("low_memory", None)

        try:
            # Спроба 1: Ідеальний сценарій
            reader = pd.read_csv(file_path, **attempt_kwargs)
            logger.info(f"⚡ [{desc_name}] Прочитано успішно. Рушій: '{engine}'")
            return reader

        except UnicodeDecodeError:
            # 🛡️ 2. Encoding Fallback: Рятуємо від "битих" кодувань
            logger.warning(f"⚠️  [{desc_name}] Помилка UTF-8 у рушії '{engine}'. Відновлення через кодування 'latin1'...")
            attempt_kwargs["encoding"] = "latin1"
            try:
                reader = pd.read_csv(file_path, **attempt_kwargs)
                logger.info(f"⚡ [{desc_name}] Прочитано успішно. Рушій: '{engine}' (Fallback: latin1)")
                return reader
            except Exception as fallback_err:
                last_error = fallback_err

        except Exception as e:
            error_msg = str(e).lower()

            # 🛡️ 3. Structural Fallback (Data Healing): Рятуємо від зміщених колонок і битих рядків
            if "expected" in error_msg and "fields" in error_msg or "tokenizing data" in error_msg:
                logger.warning(f"⚠️  [{desc_name}] Пошкоджена структура CSV (биті рядки). Активуємо on_bad_lines='skip'...")
                attempt_kwargs["on_bad_lines"] = "skip"
                try:
                    reader = pd.read_csv(file_path, **attempt_kwargs)
                    logger.info(f"⚡ [{desc_name}] Прочитано частково. Рушій: '{engine}' (Fallback: skip bad lines)")
                    return reader
                except Exception as fallback_err:
                    last_error = fallback_err
            else:
                last_error = e

    # Якщо всі 3 рушії + всі Fallback-стратегії впали
    raise RuntimeError(f"❌ Критичний збій: Жоден механізм не зміг прочитати '{desc_name}'. Остання помилка: {last_error}")
