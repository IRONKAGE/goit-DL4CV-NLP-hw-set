# ==============================================================================
# 🚀 Deep Learning & MLOps Enterprise Makefile (Zero-Config)
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Експорт змінних середовища (якщо файл існує)
# ------------------------------------------------------------------------------
ifneq (,$(wildcard ./.env))
	include .env
	export $(shell awk -F= '/^[a-zA-Z_]/ {print $$1}' .env)
endif

# ------------------------------------------------------------------------------
# 2. Системні змінні
# ------------------------------------------------------------------------------
PY_VER := 3.12
VENV := .venv
PYTHON_CMD := $(shell command -v python3 2>/dev/null || command -v python 2>/dev/null || echo "python3")
UV := $(PYTHON_CMD) -m uv
MIN_RAM_GB := 8

# ------------------------------------------------------------------------------
# 3. Кросплатформна підтримка ОС та збір інформації про залізо (OS, Arch, RAM)
# ------------------------------------------------------------------------------
ifeq ($(OS),Windows_NT)
	OPEN_CMD := start ""
	SYS_OS_NAME := Windows
	SYS_CPU_ARCH := $(PROCESSOR_ARCHITECTURE)
	SYS_RAM_GB := $(shell powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)")
	VENV_PY := $(VENV)/Scripts/python
else
	UNAME_S := $(shell uname -s)
	SYS_CPU_ARCH := $(shell uname -m)
	VENV_PY := $(VENV)/bin/python
	ifeq ($(UNAME_S),Linux)
		OPEN_CMD := xdg-open
		SYS_OS_NAME := $(shell grep '^PRETTY_NAME=' /etc/os-release | cut -d '"' -f 2 || echo "Linux")
		SYS_RAM_GB := $(shell awk '/MemTotal/ {printf "%.0f", $$2/1024/1024}' /proc/meminfo)
	endif
	ifeq ($(UNAME_S),Darwin)
		OPEN_CMD := open
		SYS_OS_NAME := $(shell sw_vers -productName) $(shell sw_vers -productVersion)
		SYS_RAM_GB := $(shell expr $$(sysctl -n hw.memsize) / 1073741824)
	endif
endif

# ------------------------------------------------------------------------------
# 4. Кольори для термінала
# ------------------------------------------------------------------------------
CYAN := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
PURPLE := \033[35m
RESET := \033[0m

.PHONY: help check-resources setup clean deep-clean mlflow run-hw1 run-hw2 run-hw3 run-hw4 run-hw5 run-fp

# Дефолтна ціль
help:
	@echo "$(CYAN)================================================================================$(RESET)"
	@echo "$(GREEN)        🧠 Deep Learning (CV & NLP) MLOps Portfolio by IRONKAGE$(RESET)"
	@echo "$(PURPLE)        🔮 ОС: $(SYS_OS_NAME) | Архітектура ЦП: $(SYS_CPU_ARCH) | ОЗП: $(SYS_RAM_GB) Гб$(RESET)"
	@echo "$(CYAN)================================================================================$(RESET)"
	@echo "$(YELLOW)🛠  Команди інфраструктури:$(RESET)"
	@echo "  $(GREEN)make setup$(RESET)        - 📦 Налаштувати середовище та залежності через uv sync"
	@echo "  $(GREEN)make mlflow$(RESET)       - 📈 Запустити MLflow Tracking Server на порту 5000"
	@echo "  $(GREEN)make clean$(RESET)        - 🧹 Очистити кеш (pycache, ruff, marimo)"
	@echo "  $(GREEN)make deep-clean$(RESET)   - 🧨 ПОВНЕ очищення: видалити середовище та артефакти"
	@echo ""
	@echo "$(YELLOW)🚀 Універсальні Дашборди (Навчання + Аналітика):$(RESET)"
	@echo "  $(PURPLE)make run-hw1$(RESET)      - 🏗️  ДЗ 1: Прогнозування міцності бетону (MLP Регресія)"
	@echo "  $(PURPLE)make run-hw2$(RESET)      - 🖼️  ДЗ 2: Класифікація пейзажів Intel (ResNet18)"
	@echo "  $(PURPLE)make run-hw3$(RESET)      - 🛋️  ДЗ 3: Детекція кімнатних об'єктів (YOLO26s NMS-Free)"
	@echo "  $(PURPLE)make run-hw4$(RESET)      - 📧 ДЗ 4: Фільтрація спаму (GloVe / TF-IDF)"
	@echo "  $(PURPLE)make run-hw5$(RESET)      - 🌍 ДЗ 5: Машинний переклад EN-EL (Seq2Seq Attention)"
	@echo "  $(PURPLE)make run-fp$(RESET)       - 🐶🐱 ФП: PetFinder (TF Data + Keras Late Fusion)"
	@echo "$(CYAN)================================================================================$(RESET)"

# ------------------------------------------------------------------------------
# 5. ІНІЦІАЛІЗАЦІЯ ІНФРАСТРУКТУРИ ТА ПЕРЕВІРКА ЗАЛІЗА
# ------------------------------------------------------------------------------
check-resources:
	@echo "$(CYAN)🔍 Аналіз апаратного забезпечення (RAM Validator)...$(RESET)"
	@echo "import platform, subprocess, sys" > .hw_check.py
	@echo "gb = 0" >> .hw_check.py
	@echo "try:" >> .hw_check.py
	@echo "    s = platform.system()" >> .hw_check.py
	@echo "    if s == 'Windows':" >> .hw_check.py
	@echo "        import ctypes" >> .hw_check.py
	@echo "        class M(ctypes.Structure): _fields_=[('l',ctypes.c_ulong),('m',ctypes.c_ulong),('t',ctypes.c_ulonglong)]" >> .hw_check.py
	@echo "        mem = M(); mem.l = ctypes.sizeof(M); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)); gb = mem.t / (1024**3)" >> .hw_check.py
	@echo "    elif s == 'Darwin':" >> .hw_check.py
	@echo "        gb = int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'])) / (1024**3)" >> .hw_check.py
	@echo "    else:" >> .hw_check.py
	@echo "        gb = int([l.split()[1] for l in open('/proc/meminfo') if 'MemTotal' in l][0]) / (1024**2)" >> .hw_check.py
	@echo "except Exception as e: print(f'Помилка детектора: {e}'); sys.exit(0)" >> .hw_check.py
	@echo "req = int('$(MIN_RAM_GB)')" >> .hw_check.py
	@echo "if gb < req: print(f'FAIL:{gb:.1f}')" >> .hw_check.py
	@echo "else: print(f'OK:{gb:.1f}')" >> .hw_check.py
	@HW_RESULT=$$(python3 .hw_check.py 2>/dev/null || python .hw_check.py 2>/dev/null); \
	rm -f .hw_check.py; \
	if echo "$$HW_RESULT" | grep -q "FAIL"; then \
		ACTUAL_RAM=$$(echo "$$HW_RESULT" | cut -d':' -f2); \
		echo "$(RED)❌ Небезпечно мало ОЗП ($$ACTUAL_RAM ГБ)! Для Deep Learning потрібно мінімум $(MIN_RAM_GB) ГБ.$(RESET)"; \
	elif echo "$$HW_RESULT" | grep -q "OK"; then \
		echo "$(GREEN)✅ Залізо схвалено (>= $(MIN_RAM_GB) ГБ ОЗП).$(RESET)"; \
	fi

setup: check-resources
	@echo "$(CYAN)🚀 Ініціалізація MLOps інфраструктури...$(RESET)"
	@echo "$(YELLOW)📁 Створення системних директорій...$(RESET)"
	@mkdir -p data models mlruns
	@echo "$(YELLOW)📦 Встановлення пакетного менеджера uv...$(RESET)"
	@$(PYTHON_CMD) -m pip install --upgrade pip --break-system-packages > /dev/null 2>&1 || true
	@$(PYTHON_CMD) -m pip install uv --break-system-packages > /dev/null 2>&1 || $(PYTHON_CMD) -m pip install --user uv > /dev/null 2>&1 || true
	@echo "$(YELLOW)⚡ Синхронізація залежностей (Python $(PY_VER)) через uv...$(RESET)"
	@$(UV) sync --python $(PY_VER)
	@if [ ! -f .env ]; then \
		echo "$(RED)🔐 Файл .env не знайдено. Створюю безпечний конфіг з .env.example...$(RESET)"; \
		cp .env.example .env 2>/dev/null || touch .env; \
	fi
	@echo "$(GREEN)✅ Setup завершено! Введіть 'make help' для перегляду списку команд.$(RESET)"

mlflow:
	@echo "$(CYAN)📈 Запуск MLflow сервера експериментів...$(RESET)"
	@echo "$(YELLOW)🌐 UI доступний за адресою: http://127.0.0.1:5000$(RESET)"
	PYTHONPATH=. $(UV) run mlflow ui --host 127.0.0.1 --port 5000

# ------------------------------------------------------------------------------
# 6. ЗАПУСК MLOPS ДАШБОРДІВ (MARIMO) - Тренування + EDA в одному вікні
# ------------------------------------------------------------------------------
run-hw1:
	@echo "$(PURPLE)📊 Відкриття дашборду HW 01 (Міцність бетону)...$(RESET)"
	PYTHONPATH=. $(UV) run marimo edit notebooks_eda/hw_01_eda.py

run-hw2:
	@echo "$(PURPLE)📊 Відкриття дашборду HW 02 (Intel Image Classification)...$(RESET)"
	PYTHONPATH=. $(UV) run marimo edit notebooks_eda/hw_02_eda.py

run-hw3:
	@echo "$(PURPLE)📊 Відкриття дашборду HW 03 (YOLO Indoor Object Detection)...$(RESET)"
	PYTHONPATH=. $(UV) run marimo edit notebooks_eda/hw_03_eda.py

run-hw4:
	@echo "$(PURPLE)📊 Відкриття дашборду HW 04 (NLP Spam Detection)...$(RESET)"
	PYTHONPATH=. $(UV) run marimo edit notebooks_eda/hw_04_eda.py

run-hw5:
	@echo "$(PURPLE)📊 Відкриття дашборду HW 05 (Machine Translation Seq2Seq)...$(RESET)"
	PYTHONPATH=. $(UV) run marimo edit notebooks_eda/hw_05_eda.py

run-fp:
	@echo "$(PURPLE)📊 Відкриття дашборду Фінального Проєкту (PetFinder)...$(RESET)"
	PYTHONPATH=. $(UV) run marimo edit notebooks_eda/hw_fp_eda.py

# ------------------------------------------------------------------------------
# 7. ОЧИЩЕННЯ СМІТТЯ ТА КЕШІВ
# ------------------------------------------------------------------------------
clean:
	@echo "$(YELLOW)🧹 Очищення тимчасових файлів та кешу...$(RESET)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "__marimo__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "$(GREEN)✨ Системний кеш очищено.$(RESET)"

deep-clean: clean
	@echo "$(RED)🧨 ПОВНЕ ОЧИЩЕННЯ: Видалення середовища .venv та ML-артефактів...$(RESET)"
	rm -rf $(VENV)
	rm -rf mlruns/
	rm -rf .uv_cache/
	@echo "$(GREEN)✅ Репозиторій повернуто до первозданного вигляду (Zero-State)!$(RESET)"

# Хак для ігнорування невідомих аргументів термінала
%:
	@:
