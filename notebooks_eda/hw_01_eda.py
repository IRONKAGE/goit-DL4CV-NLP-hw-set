import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def title_head_hw(mo):
    css_no_scroll = mo.md(
        """
        <div class="marimo-noscroll-override"></div>
        <style>
            marimo-cell-output:has(.marimo-noscroll-override),
            .output-area:has(.marimo-noscroll-override) {
                max-height: none !important;
                overflow-y: visible !important;
            }
        </style>
        """
    )

    mo.md("""
    <div style="text-align: center; font-size: 2.2em; font-weight: bold; margin-top: 0.67em; margin-bottom: 0.67em;">
        🏗️ ДЗ №1: ️Прогнозування міцності бетону <i>(Deep Learning)</i>
    </div>

    <h3 align="center"><b><u>Пайплайн</u>: EDA ➔ Train/Val Split ➔ StandardScaler ➔ MLP Architecture ➔ Early Stopping ➔ Evaluation (MSE, MAE, R²)</b></h3>

    <p align="center"><i>© Oleh Hatsenko (IRONKAGE) | Deep Learning for Computer Vision and NLP [09.2026]</i></p>
    """)
    return (css_no_scroll,)


@app.cell(hide_code=True)
def imports():
    import marimo as mo
    import polars as pl
    import matplotlib.pyplot as plt
    import seaborn as sns
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.metrics import (
        mean_squared_error,
        mean_absolute_error,
        r2_score,
    )
    import numpy as np
    import plotly.express as px
    import scipy.cluster.hierarchy as sch
    import scipy.spatial.distance as ssd
    import os
    import urllib.request
    from hw_01_concrete.ui_labels import UA_COLUMNS

    # Імпорт нашого MLOps-ядра
    from core.etl_core import SecureDownloader
    from core import get_hardware_config, set_global_seed
    from hw_01_concrete.data_adapter import get_dataloaders
    from hw_01_concrete.model import ConcreteStrengthMLP
    from core.trainer import BaseTrainer

    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)

    mo.center(
        mo.md(
            "✅ **Бібліотеки, Локальні Адаптери та Ядро MLOps успішно імпортовано!**"
        )
    )
    return (
        BaseTrainer,
        ConcreteStrengthMLP,
        SecureDownloader,
        UA_COLUMNS,
        get_dataloaders,
        get_hardware_config,
        mean_absolute_error,
        mean_squared_error,
        mo,
        nn,
        np,
        optim,
        pl,
        plt,
        px,
        r2_score,
        sch,
        set_global_seed,
        sns,
        ssd,
        torch,
        urllib,
    )


@app.cell(hide_code=True)
def load_data(SecureDownloader, UA_COLUMNS, mo, pl, urllib):
    DATA_DIR = "data/hw_01"
    CSV_FILENAME = "concrete.csv"
    FALLBACK_URL = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/concrete.csv"

    def download_concrete_csv(target_path: str) -> None:
        urllib.request.urlretrieve(FALLBACK_URL, target_path)

    # Використовуємо всю міць Smart Router
    downloader = SecureDownloader(
        dataset_path="mchilamwar/predict-concrete-strength",  # Спроба 1: Kaggle API
        fallback_generator=download_concrete_csv,              # Спроба 5: Сирий CSV
        data_dir=DATA_DIR,
        zip_name="concrete_archive.zip"
    )

    downloader.download(target_filename=CSV_FILENAME)

    # Якщо відпрацював Kaggle, тут відбудеться розпакування ZIP.
    # Якщо відпрацював fallback, файл concrete.csv вже лежить готовий,
    # і extract_atomically просто миттєво поверне його шлях!
    csv_path = downloader.extract_atomically(
        target_extensions=('.csv',),
        expected_filename=CSV_FILENAME
    )[0]

    # Читаємо файл і одразу перейменовуємо колонки наявними ключами зі словника
    df_raw = pl.read_csv(csv_path).rename({
        col: UA_COLUMNS[col] for col in pl.read_csv(csv_path).columns if col in UA_COLUMNS
    })

    mo.center(mo.md(f"📊 Сирі дані (Джерело: *Kaggle / Yeh Repository*): ➡️ Завантажено рядків: **{df_raw.shape[0]}**, колонок: **{df_raw.shape[1]}**"))
    return (df_raw,)


@app.cell(hide_code=True)
def summary_statistics(css_no_scroll, df_raw, mo):
    summary_df = df_raw.describe()

    mo.vstack([
        css_no_scroll,
        mo.center(mo.md("### 📈 Описова статистика (Summary Statistics)")),
        mo.ui.table(summary_df),
        mo.center(mo.md("### 📋 Перегляд даних")),
        mo.ui.table(df_raw, pagination=True, page_size=5)
    ])
    return


@app.cell(hide_code=True)
def correlation_heatmap(
    UA_COLUMNS,
    css_no_scroll,
    df_raw,
    mo,
    np,
    px,
    sch,
    ssd,
):
    # Конвертуємо Polars у Pandas
    df_pd = df_raw.to_pandas()

    # 1. Рахуємо кореляцію тільки для числових колонок
    numeric_df = df_pd.select_dtypes(
        include=["float64", "int64", "float32", "int32"]
    )
    corr_matrix = numeric_df.corr(method="pearson")

    # Безпечно витягуємо кореляцію (з урахуванням можливої українізації)
    col_strength_ua = UA_COLUMNS.get("strength", "Міцність (МПа)")
    col_cement_ua = UA_COLUMNS.get("cement", "Цемент (кг/м³)")

    if (
        col_strength_ua in corr_matrix.columns
        and col_cement_ua in corr_matrix.columns
    ):
        # Дані вже перекладені
        _top_corr_val = corr_matrix.loc[col_strength_ua, col_cement_ua]
        _top_corr_str = f"~{_top_corr_val:.2f}"
    elif "strength" in corr_matrix.columns and "cement" in corr_matrix.columns:
        # Дані ще в оригінальному англійському форматі
        _top_corr_val = corr_matrix.loc["strength", "cement"]
        _top_corr_str = f"~{_top_corr_val:.2f}"
    else:
        _top_corr_str = "[Дані відсутні]"

    # ==========================================================
    # Matrix Seriation (Yan Holtz Clustered Correlogram)
    # ==========================================================
    # Перетворюємо матрицю кореляцій на матрицю математичних відстаней:
    # Сильна позитивна кореляція (1) -> Відстань 0 (Дуже близько)
    # Сильна негативна кореляція (-1) -> Відстань 2 (Дуже далеко)
    dists = 1 - corr_matrix.values
    np.fill_diagonal(dists, 0)  # Діагональ завжди має відстань 0
    dists = np.clip(
        (dists + dists.T) / 2, 0, 2
    )  # Захист від float-помилок округлення

    # Виконуємо ієрархічну кластеризацію (Метод Варда)
    linkage = sch.linkage(ssd.squareform(dists), method="ward")

    # Отримуємо новий, ідеально згрупований порядок індексів
    optimal_order = sch.leaves_list(linkage)

    # Пересортовуємо рядки та стовпці нашої матриці за новим порядком
    corr_matrix_sorted = corr_matrix.iloc[optimal_order, optimal_order]

    # 2. Переклад назв колонок і індексів для красивого UI
    corr_matrix_sorted = corr_matrix_sorted.rename(
        columns=UA_COLUMNS, index=UA_COLUMNS
    )

    # 3. Налаштування кольорів під тему Marimo
    _theme = mo.app_meta().theme
    _text_color = "white" if _theme == "dark" else "#1f2937"

    # 4. Будуємо інтерактивну теплову карту (Heatmap) через Plotly
    fig_corr = px.imshow(
        corr_matrix_sorted,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",  # Червоний - позитивна, Синій - негативна
        zmin=-1,
        zmax=1,
        labels=dict(color="Коеф. Пірсона"),
    )

    fig_corr.update_traces(
        hovertemplate="<b>Ознака X:</b> %{x}<br><b>Ознака Y:</b> %{y}<br><b>Кореляція:</b> %{z:.5f}<extra></extra>"
    )

    fig_corr.update_layout(
        title=dict(
            text="<b>Теплова карта кореляцій (Smart Clustered)</b>",
            x=0.5,
            font=dict(color=_text_color, size=18),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=_text_color),
        height=750,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    # 5. MLOps Інсайт для бізнесу (Адаптовано під бетон)
    _insight_ui = mo.md(
        f"""
        <div class="corr-expanded-cell"></div>
        <style>
            marimo-cell-output:has(.corr-expanded-cell),
            .output-area:has(.corr-expanded-cell) {{
                max-height: none !important;
                overflow: visible !important;
            }}
        </style>
        > **💡 Tech Lead Insight (Аналіз матриці):**<br/>
        > *За порадою Yan Holtz (Data to Viz), ми застосували ієрархічну кластеризацію матриці. Тепер фічі з подібною поведінкою автоматично згруповані у візуальні "блоки", відкриваючи приховану хімію набору даних.*<br/>

        > - **Головний драйвер:** Перетин `Міцність (МПа)` та `Цемент (кг/м³)` має коефіцієнт `{_top_corr_str}`. Це найпотужніший позитивний сигнал, що повністю підтверджує базовий закон бетонознавства.
        > - **Хімічний конфлікт (Вода vs Пластифікатор):** Зверніть увагу на негативну кореляцію між суперпластифікатором та водою. Суперпластифікатор додають саме для того, щоб зменшити кількість води без втрати текучості, і алгоритм кластеризації ідеально розпізнав цей фізичний взаємозв'язок, відкинувши їх одне від одного!
        """
    )

    mo.vstack([css_no_scroll, fig_corr, _insight_ui])
    return


@app.cell(hide_code=True)
def feature_dropdown(UA_COLUMNS, css_no_scroll, df_raw, mo):
    _head = mo.center(mo.md("### 🔬 Інтерактивний аналіз ознак\nОберіть будь-який компонент із випадаючого списку, щоб побачити його вплив на міцність бетону."))

    # Динамічно визначаємо українські назви для цільової змінної та дефолтного вибору
    _target_col = UA_COLUMNS.get("strength", "Міцність (МПа)")
    default_feature = UA_COLUMNS.get("cement", "Цемент (кг/м³)")

    # Відсікаємо цільову змінну, використовуючи її актуальну (перекладену) назву
    feature_options = [col for col in df_raw.columns if col != _target_col]

    # Захист: якщо default_feature відсутній (наприклад, ще не переклали), беремо першу доступну
    if default_feature not in feature_options and feature_options:
        default_feature = feature_options[0]

    dropdown = mo.ui.dropdown(
        options=feature_options,
        value=default_feature,
        label="🔍 Оберіть ознаку (Feature):"
    )

    mo.vstack([
        css_no_scroll,
        _head,
        mo.center(dropdown)
    ])
    return (dropdown,)


@app.cell(hide_code=True)
def interactive_scatter(
    UA_COLUMNS,
    css_no_scroll,
    df_raw,
    dropdown,
    mo,
    plt,
    sns,
):
    # Витягуємо локалізовані назви осей
    _target_col = UA_COLUMNS.get("strength", "Міцність (МПа)")
    hue_col = UA_COLUMNS.get("age", "Вік (дні)")

    if dropdown.value:
        fig_scatter_eda, ax_scatter_eda = plt.subplots(figsize=(10, 6))

        sns.scatterplot(
            data=df_raw.to_pandas(),
            x=dropdown.value,
            y=_target_col,
            hue=hue_col if hue_col in df_raw.columns else None,
            palette="viridis",
            alpha=0.7,
            ax=ax_scatter_eda
        )

        plt.title(f"Вплив компонента «{dropdown.value}» на міцність бетону", fontsize=14)
        plt.tight_layout()

        display_scatter_eda = mo.as_html(fig_scatter_eda)
    else:
        display_scatter_eda = mo.md("*Оберіть ознаку для побудови графіка.*")

    mo.vstack([
        css_no_scroll,
        mo.center(display_scatter_eda)
    ])
    return


@app.cell(hide_code=True)
def theory_markdown(css_no_scroll, mo):
    _theory_ui = mo.md("""
    <h2 align="center" style="margin-bottom: 25px;">2. Теоретичне обґрунтування архітектури</h2>

    ### 🧠 Архітектура мережі
    Ми використовуємо багатошаровий перцептрон (**MLP**) з конфігурацією прихованих шарів `[128, 64, 32]`. Така «лійкоподібна» архітектура дозволяє моделі поступово виділяти складні нелінійні залежності між хімічними компонентами суміші. Для запобігання перенавчанню додано шари `BatchNorm1d` (нормалізація батчу) та `Dropout(0.2)` (регуляризація).

    ---

    ### 📉 Вибір функції втрат (Loss Function)
    Обрано **MSELoss** *(Mean Squared Error)*. У будівництві сильні відхилення у прогнозі міцності є критичними (це питання безпеки конструкцій). MSE квадратично штрафує великі помилки, змушуючи модель уникати екстремальних промахів.

    ---

    ### 🚀 Вибір оптимізатора
    Замість класичного SGD ми використовуємо **AdamW** *(Adam with Weight Decay)*. Він адаптує швидкість навчання для кожного параметра окремо, що дозволяє значно швидше знаходити глобальний мінімум. Наявність `weight_decay` додає L2-регуляризацію безпосередньо в крок оптимізації, що є критично важливим для табличних даних, які схильні до перенавчання.

    ---

    ### 📊 Вибір метрик
    *   **MSE:** Базова метрика для оптимізації (штрафує за великі промахи).
    *   **MAE (Mean Absolute Error):** Дає фізичне розуміння помилки. Показує середнє відхилення у мегапаскалях (**МПа**), що легко пояснити інженерам-технологам.
    *   **R² (Коефіцієнт детермінації):** Показує відсоток дисперсії цільової змінної, яку змогла пояснити наша модель.
    """)

    mo.vstack([
        css_no_scroll,
        mo.center(_theory_ui)
    ])
    return


@app.cell(hide_code=True)
def train_model(
    BaseTrainer,
    ConcreteStrengthMLP,
    get_dataloaders,
    get_hardware_config,
    mo,
    nn,
    optim,
    set_global_seed,
):
    _header = mo.md("<h2 align='center' style='margin-bottom: 20px;'>3. Тренування моделі (MLP)</h2>")
    mo.output.append(_header)

    # 1. Підготовка DataLoaders (зі стандартизацією StandardScaler)
    train_loader, val_loader, scaler = get_dataloaders(batch_size=32, seed=42)

    # 2. Ініціалізація
    set_global_seed(42)
    device, device_name = get_hardware_config()

    model = ConcreteStrengthMLP(input_dim=8, hidden_dims=[128, 64, 32], dropout_rate=0.2)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    epochs_count = 150
    # Інтеграція експерименту MLflow у новий BaseTrainer
    trainer = BaseTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        experiment_name="concrete_strength",
        run_name="marimo_run"
    )

    # Інтерактивний Progress Bar у стилі Windows XP
    with mo.status.progress_bar(
        title=f"Тренування нейромережі ({epochs_count} епох)...",
        subtitle=f"💎 <b>Engine:</b> {device_name} <br/>⏳ <b>Ініціалізація...</b>",
        remove_on_exit=True,
        total=epochs_count,
    ) as bar:

        # Функція зворотного виклику для оновлення UI
        def update_ui(epoch, t_loss, v_loss):
            bar.update(
                increment=1,
                subtitle=f"💎 <b>Engine:</b> {device_name} <br/>☣️ <b>Епоха:</b> {epoch}/{epochs_count} | 📉 <b>Train MSE:</b> {t_loss:.1f} | 📈 <b>Val MSE:</b> {v_loss:.1f}"
            )

        # Прибрано verbose_console=False, щоб відповідати еталонному BaseTrainer
        history = trainer.fit(
            train_loader,
            val_loader,
            epochs=epochs_count,
            patience=20,
            model_save_path="models/hw_01/best_marimo_model.pth",
            epoch_callback=update_ui
        )

    _footer = mo.md(f"✅ **Тренування завершено на пристрої {device_name}!** Дані успішно розбиті на батчі та нормалізовані.")
    mo.output.append(mo.center(_footer))
    return device, history, model, val_loader


@app.cell(hide_code=True)
def plot_loss(css_no_scroll, history, mo, plt, sns):
    _header = mo.md("<h2 align='center' style='margin-bottom: 20px;'>4. Аналіз процесу навчання (Loss Curve)</h2>")

    fig_loss, ax_loss = plt.subplots(figsize=(10, 5))

    epochs = range(1, len(history["train_loss"]) + 1)
    # Робимо лінії трохи товщими (linewidth=2.5) і задаємо приємні контрастні кольори
    sns.lineplot(x=epochs, y=history["train_loss"], label="Train Loss (MSE)", ax=ax_loss, linewidth=2.5, color="#2E86AB")
    sns.lineplot(x=epochs, y=history["val_loss"], label="Validation Loss (MSE)", ax=ax_loss, linewidth=2.5, color="#D64933")

    ax_loss.set_title("Динаміка падіння помилки під час тренування", fontsize=14, pad=15)
    ax_loss.set_xlabel("Епоха", fontsize=12)
    ax_loss.set_ylabel("MSE Loss", fontsize=12)

    # Додаємо легку пунктирну сітку для зручності читання значень
    ax_loss.grid(True, linestyle='--', alpha=0.7)
    ax_loss.legend(fontsize=11)

    plt.tight_layout()
    display_loss = mo.as_html(fig_loss)

    _insight_ui = mo.md(
        """
        > **💡 Tech Lead Insight (Аналіз збіжності):**<br/>
        > Крива демонструє класичне "здорове" навчання: `Train Loss` та `Validation Loss` синхронно спадають без різкого розходження (ножиць). Це означає, що модель не запам'ятовує сирі дані напам'ять (відсутність жорсткого overfitting'у), а ефективно виділяє загальні закономірності суміші. Завдяки механізму `Early Stopping`, тренування автоматично зупинилося, зберігши найкращі ваги в точці мінімуму валідаційної помилки.
        """
    )

    mo.vstack([
        css_no_scroll,
        _header,
        mo.center(display_loss),
        _insight_ui
    ])
    return


@app.cell(hide_code=True)
def evaluate_predictions(
    css_no_scroll,
    device,
    mean_absolute_error,
    mean_squared_error,
    mo,
    model,
    plt,
    r2_score,
    sns,
    torch,
    val_loader,
):
    # 1. Відцентрований заголовок
    _header = mo.md("<h2 align='center' style='margin-bottom: 20px;'>5. Оцінка на тестовій вибірці (Actual vs Predicted)</h2>")

    # Завантаження найкращих ваг
    model.load_state_dict(torch.load("models/hw_01/best_marimo_model.pth", weights_only=True))
    model.eval()

    preds_list = []
    targets_list = []

    # Отримання прогнозів
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            preds = model(x_batch)
            preds_list.extend(preds.cpu().numpy().flatten())
            targets_list.extend(y_batch.numpy().flatten())

    # Розрахунок метрик
    mse = mean_squared_error(targets_list, preds_list)
    mae = mean_absolute_error(targets_list, preds_list)
    r2 = r2_score(targets_list, preds_list)

    # 2. Побудова графіка
    fig_scatter_pred, ax_scatter_pred = plt.subplots(figsize=(8, 8))
    sns.scatterplot(x=targets_list, y=preds_list, alpha=0.7, ax=ax_scatter_pred, color="#2E86AB", s=60)

    min_val = min(min(targets_list), min(preds_list))
    max_val = max(max(targets_list), max(preds_list))
    ax_scatter_pred.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2.5, label="Ідеальний прогноз")

    ax_scatter_pred.set_title("Справжня міцність vs Прогнозована", fontsize=14, pad=15)
    ax_scatter_pred.set_xlabel("Справжня міцність (МПа)", fontsize=12)
    ax_scatter_pred.set_ylabel("Прогнозована міцність (МПа)", fontsize=12)
    ax_scatter_pred.grid(True, linestyle='--', alpha=0.6)
    ax_scatter_pred.legend(fontsize=11)

    display_scatter_pred = mo.as_html(fig_scatter_pred)

    # 3. Формування блоку метрик та графіка
    _metrics_and_plot = mo.vstack([
        mo.md(f"### 🏆 Метрики найкращої моделі:\n- **MSE:** {mse:.2f}\n- **MAE:** {mae:.2f} МПа\n- **R² Score:** {r2:.3f}"),
        mo.center(display_scatter_pred)
    ])

    _insight_ui = mo.md(
        """
        > **💡 Tech Lead Insight (Аналіз результатів):**<br/>
        > Графік *Actual vs Predicted* дозволяє візуально оцінити якість моделі без занурення в сухі цифри. Точки щільно групуються вздовж ідеальної діагоналі (червоної пунктирної лінії), що свідчить про високу точність прогнозів. Значення **R²** підтверджує, що наша мережа здатна пояснити переважну частку дисперсії у даних, а **MAE** дає бізнесу пряме розуміння середньої похибки в реальних фізичних величинах (МПа).
        """
    )

    mo.vstack([
        css_no_scroll,
        _header,
        _metrics_and_plot,
        _insight_ui
    ])
    return mae, r2


@app.cell(hide_code=True)
def final_conclusions(css_no_scroll, mae, mo, r2):
    _conclusions_ui = mo.md(f"""
    <h2 align='center' style='margin-bottom: 25px;'>6. Загальні висновки та напрямки оптимізації</h2>

    **1. 🎯 Аналіз метрик:**
    Модель продемонструвала стабільні результати. Коефіцієнт детермінації **$R^2 = {r2:.2f}$** вказує на те, що нейромережа здатна пояснити переважну більшість дисперсії у даних. Середня абсолютна похибка складає близько **{mae:.2f} МПа**, що є цілком прийнятним показником для попередніх інженерних розрахунків.

    **2. 🧠 Аналіз архітектури:**
    Обрана архітектура багатошарового перцептрона (MLP) з використанням `BatchNorm1d` довела свою ефективність. Нормалізація батчів допомогла стабілізувати градієнти, а `Dropout` дозволив уникнути жорсткого перенавчання на тренувальній вибірці.

    **3. 🚧 Обмеження підходу:**
    * **Об'єм даних:** Набір містить відносно малу кількість спостережень (~1000 рядків). Глибокі нейронні мережі зазвичай потребують значно більших вибірок для розкриття свого повного потенціалу.
    * **Сліпі плями:** У наборі даних відсутні критично важливі контекстні змінні, такі як: температура затвердіння, вологість повітря, а також початкова марка та якість самого цементу.

    **4. 🚀 Шляхи покращення (Оптимізація):**
    * Впровадження **K-Fold Cross Validation** для більш об'єктивної оцінки моделі на малих даних.
    * Застосування алгоритмів градієнтного бустингу (**XGBoost** або **Explainable Boosting Machine**). Для табличних даних такого розміру ансамблі дерев часто перевершують класичні нейромережі.
    * **MLOps-інтеграція:** Налаштування автоматичного підбору гіперпараметрів (архітектура шарів, learning rate) за допомогою фреймворку **Optuna**.
    """)

    mo.vstack([
        css_no_scroll,
        _conclusions_ui
    ])
    return


if __name__ == "__main__":
    app.run()
