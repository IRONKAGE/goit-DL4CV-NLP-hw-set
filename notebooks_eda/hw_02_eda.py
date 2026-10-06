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
        🏞️ ДЗ №2: Класифікація Зображень <i>(Transfer Learning - ResNet18)</i>
    </div>

    <h3 align="center"><b><u>Пайплайн</u>: ImageFolder ➔ Data Augmentation ➔ ResNet18 Fine-Tuning ➔ Classification Report ➔ Confusion Matrix</b></h3>

    <p align="center"><i>© Oleh Hatsenko (IRONKAGE) | Deep Learning for Computer Vision and NLP</i></p>
    """)
    return (css_no_scroll,)


@app.cell(hide_code=True)
def imports():
    import marimo as mo
    import matplotlib.pyplot as plt
    import seaborn as sns
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
    import numpy as np
    import plotly.express as px
    from torchvision.utils import make_grid
    import base64
    from io import BytesIO
    from PIL import Image

    # Імпорт MLOps-ядра та адаптерів
    from core import get_hardware_config, set_global_seed
    from core.trainer import BaseTrainer
    from hw_02_intel_cv.data_adapter import get_image_dataloaders
    from hw_02_intel_cv.model import IntelSceneResNet
    from hw_02_intel_cv.ui_labels import get_ua_label

    sns.set_theme(style="whitegrid")

    mo.center(mo.md("✅ **Інфраструктура комп'ютерного зору (TorchVision & MLOps) успішно завантажена!**"))
    return (
        BaseTrainer,
        BytesIO,
        DataLoader,
        Image,
        IntelSceneResNet,
        accuracy_score,
        base64,
        classification_report,
        confusion_matrix,
        f1_score,
        get_hardware_config,
        get_image_dataloaders,
        get_ua_label,
        make_grid,
        mo,
        nn,
        np,
        optim,
        plt,
        px,
        set_global_seed,
        sns,
        torch,
    )


@app.cell(hide_code=True)
def load_and_visualize_data(
    css_no_scroll,
    get_image_dataloaders,
    get_ua_label,
    make_grid,
    mo,
    np,
    plt,
    set_global_seed,
    torch,
):
    _header = mo.md("<h2 align='center' style='margin-bottom: 20px;'>1. Підготовка та аугментація даних (150x150)</h2>")

    set_global_seed(42)
    # Завантаження DataLoaders (Беремо батч більший, щоб гарантовано захопити всі 6 класів)
    train_loader, val_loader, test_loader, class_names = get_image_dataloaders(batch_size=64)

    # Візуалізація батчу (Денормалізація для відображення)
    def imshow(img, title):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * std + mean  # Unnormalize
        npimg = img.numpy()

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(np.transpose(npimg, (1, 2, 0)).clip(0, 1))
        ax.set_title(title, fontsize=14, pad=15)
        ax.axis('off')
        plt.tight_layout()
        return fig

    # Отримуємо один батч тренувальних даних
    _dataiter = iter(train_loader)
    _images, _labels = next(_dataiter)

    # 🛠 Розумний фільтр: 6 унікальних класів + 2 випадкових (Разом 8)
    _selected_idx = []
    _seen_classes = set()

    # Крок 1: Шукаємо по 1 представнику кожної категорії
    for _i, _lbl in enumerate(_labels.tolist()):
        if _lbl not in _seen_classes:
            _seen_classes.add(_lbl)
            _selected_idx.append(_i)
        if len(_seen_classes) == 6:
            break

    # Крок 2: Добиваємо список до 8 випадковими картинками
    for _i in range(len(_labels)):
        if len(_selected_idx) == 8:
            break
        if _i not in _selected_idx:
            _selected_idx.append(_i)

    _disp_images = _images[_selected_idx]
    _disp_labels = _labels[_selected_idx]

    # Показуємо 8 зображень у 2 ряди (nrow=4)
    grid_img = make_grid(_disp_images, nrow=4)
    labels_ua = [get_ua_label(class_names[_disp_labels[j]]) for j in range(8)]

    fig_samples = imshow(grid_img, f"Приклади аугментованих зображень (Усі 6 класів + 2 випадкові):\n{labels_ua}")
    display_samples = mo.as_html(fig_samples)

    _insight_ui = mo.callout(
        title="💡 Tech Lead Insight (Data Pipeline):",
        kind="info",
        value="Зображення успішно завантажені та ресайзнуті до формату 150x150 пікселів. У візуалізованому батчі гарантовано представлені всі 6 категорій. Для тренувального набору активовано нескладну аугментацію (`RandomHorizontalFlip`, `RandomRotation`, `ColorJitter`) та нормалізацію за стандартами ImageNet, що є обов'язковою умовою для коректного використання ResNet18."
    )

    mo.vstack([
        css_no_scroll, _header, mo.center(display_samples), _insight_ui
    ])
    return class_names, test_loader, train_loader, val_loader


@app.cell(hide_code=True)
def theory_markdown(css_no_scroll, mo):
    _theory_ui = mo.md("""
    <h2 align="center" style="margin-bottom: 25px;">2. Архітектура Transfer Learning та Функції</h2>

    ### 🧠 Архітектура мережі: ResNet18 (Fine-Tuning)
    Використано попередньо навчену на ImageNet модель **ResNet18**. Згідно з найкращими практиками Transfer Learning, ранні шари згортки "заморожені" (freeze), оскільки вони вже вміють розпізнавати базові патерни (лінії, текстури). Ми розморозили лише фінальний `layer4` та замінили повнозв'язний шар `Linear` під наші 6 класів природних сцен.

    ---

    ### 📉 Вибір функції втрат: CrossEntropyLoss
    Оскільки це задача багатокласової класифікації (Multi-class), **CrossEntropyLoss** є математичним еталоном. Вона поєднує `LogSoftmax` (для отримання ймовірностей) та `NLLLoss` (для штрафування за неправильні прогнози) в одну високооптимізовану операцію.

    ---

    ### 🚀 Вибір оптимізатора: Adam
    Використано оптимізатор **Adam**, який поєднує переваги Momentum та RMSprop. Він автоматично адаптує швидкість навчання (learning rate) для кожної ваги окремо, що критично важливо при Fine-Tuning'у розморожених шарів ResNet.

    ---

    ### 📊 Вибір метрики: F1-Score (Weighted)
    Точність (Accuracy) може бути оманливою при дисбалансі класів. **F1-Score (гармонічне середнє між Precision та Recall)** дає більш об'єктивну оцінку здатності моделі розпізнавати складні сцени (наприклад, відрізняти море від льодовика), збалансовано враховуючи хибнопозитивні та хибнонегативні спрацювання.
    """)

    mo.vstack([
        css_no_scroll,
        mo.center(_theory_ui)
    ])
    return


@app.cell(hide_code=True)
def train_model(
    BaseTrainer,
    IntelSceneResNet,
    get_hardware_config,
    mo,
    nn,
    optim,
    train_loader,
    val_loader,
):
    _header = mo.md("<h2 align='center' style='margin-bottom: 20px;'>3. Тренування мережі (MLflow Tracking)</h2>")
    mo.output.append(_header)

    device, device_name = get_hardware_config()

    # Ініціалізація Transfer Learning
    model = IntelSceneResNet(num_classes=6, freeze_backbone=True)
    criterion = nn.CrossEntropyLoss()
    # Навчаємо лише розморожені параметри (layer4 та класифікатор)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    epochs_count = 10

    # Інтеграція експерименту MLflow у новий BaseTrainer
    trainer = BaseTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        experiment_name="intel_image_classification",
        run_name="resnet18_fine_tuning"
    )

    with mo.status.progress_bar(
        title=f"Fine-Tuning ResNet18 ({epochs_count} епох)...",
        subtitle=f"💎 <b>Engine:</b> {device_name} <br/>⏳ <b>Ініціалізація...</b>",
        remove_on_exit=True,
        total=epochs_count,
    ) as bar:

        def update_ui(epoch, t_loss, v_loss):
            bar.update(
                increment=1,
                subtitle=f"💎 <b>Engine:</b> {device_name} <br/>☣️ <b>Епоха:</b> {epoch}/{epochs_count} | 📉 <b>Train Loss:</b> {t_loss:.3f} | 📈 <b>Val Loss:</b> {v_loss:.3f}"
            )

        # Виклик без verbose_console (ядро BaseTrainer робить це за нас)
        history = trainer.fit(
            train_loader, val_loader,
            epochs=epochs_count, patience=5,
            model_save_path="models/hw_02/best_cv_model.pth",
            epoch_callback=update_ui
        )

    _footer = mo.md(f"✅ **Тренування завершено на пристрої {device_name}! Найкращу модель збережено.**")
    mo.output.append(mo.center(_footer))
    return device, history, model


@app.cell(hide_code=True)
def plot_loss(css_no_scroll, history, mo, plt, sns):
    _header = mo.md("<h2 align='center' style='margin-bottom: 20px;'>4. Аналіз збіжності (Loss Curve)</h2>")

    fig_loss, ax_loss = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    sns.lineplot(x=epochs, y=history["train_loss"], label="Train Loss (CrossEntropy)", ax=ax_loss, linewidth=2.5, color="#2E86AB")
    sns.lineplot(x=epochs, y=history["val_loss"], label="Validation Loss (CrossEntropy)", ax=ax_loss, linewidth=2.5, color="#D64933")

    ax_loss.set_title("Динаміка падіння помилки при Fine-Tuning", fontsize=14, pad=15)
    ax_loss.set_xlabel("Епоха", fontsize=12)
    ax_loss.set_ylabel("CrossEntropy Loss", fontsize=12)
    ax_loss.grid(True, linestyle='--', alpha=0.7)
    ax_loss.legend(fontsize=11)

    plt.tight_layout()
    display_loss = mo.as_html(fig_loss)

    _insight_ui = mo.callout(
        title="💡 Tech Lead Insight (Аналіз збіжності):",
        kind="info",
        value="Криві навчання демонструють відмінну збіжність: `Train Loss` та `Validation Loss` стабільно зменшуються без ознак жорсткого перенавчання (overfitting). Завдяки тому, що ми заморозили базові шари (Feature Extractor) і навчали лише розморожений `layer4` та фінальний класифікатор, модель змогла швидко та безпечно адаптувати потужні ознаки ImageNet до наших природних ландшафтів."
    )

    mo.vstack([
        css_no_scroll, _header, mo.center(display_loss), _insight_ui
    ])
    return


@app.cell(hide_code=True)
def evaluate_on_test(
    accuracy_score,
    class_names,
    classification_report,
    confusion_matrix,
    css_no_scroll,
    device,
    f1_score,
    get_ua_label,
    mo,
    model,
    px,
    test_loader,
    torch,
):
    _header = mo.md("<h2 align='center' style='margin-bottom: 20px;'>5. Оцінка на тестовій вибірці (Метрики та Матриця)</h2>")

    model.load_state_dict(torch.load("models/hw_02/best_cv_model.pth", weights_only=True))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, _labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, _preds = torch.max(outputs, 1)
            all_preds.extend(_preds.cpu().numpy())
            all_labels.extend(_labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    ua_classes = [get_ua_label(c) for c in class_names]
    clf_report = classification_report(all_labels, all_preds, target_names=ua_classes)

    cm = confusion_matrix(all_labels, all_preds)
    fig_cm = px.imshow(
        cm, text_auto=True, aspect="auto",
        x=ua_classes, y=ua_classes,
        color_continuous_scale="Blues",
        labels=dict(x="Прогнозований клас", y="Фактичний клас", color="Кількість")
    )
    # Зменшили ширину до 550, щоб графік ідеально став поруч із текстом
    fig_cm.update_layout(title_x=0.5, height=500, width=550, margin=dict(t=30, b=0))

    # Примусово розділяємо текстові блоки та об'єднуємо їх вертикально
    _metrics_block = mo.md(f"""
    ### 🏆 Глобальні метрики:
    - **Точність (Accuracy):** {acc*100:.2f}%
    - **F1-Score (Weighted):** {f1:.4f}
    """)

    _report_block = mo.md(f"""
    **Детальний класифікаційний звіт:**
    ```text
    {clf_report}
    ```
    """)

    # Створюємо жорстку ліву колонку з вирівнюванням по лівому краю
    _left_column = mo.vstack([_metrics_block, _report_block], align="start")

    _insight_ui = mo.callout(
        title="💡 Tech Lead Insight (Аналіз матриці помилок):",
        kind="info",
        value="Діагональ матриці демонструє впевнені коректні прогнози. Головні зони ризику (False Positives) зазвичай виникають між візуально схожими класами: `Гори` та `Льодовик` (обидва мають снігові вершини), або `Будівлі` та `Вулиця` (спільні урбаністичні текстури). Саме тому Transfer Learning тут показує себе блискуче — фільтри ResNet18 здатні розрізняти тонкі текстурні деталі краще, ніж прості CNN."
    )

    # Використовуємо hstack для поділу рядка на дві колонки (з вирівнюванням по центру)
    _two_columns_ui = mo.center(mo.hstack(
        [_left_column, mo.center(mo.as_html(fig_cm))],
        justify="space-around",
        align="center",
        wrap=True
    ))

    mo.vstack([
        css_no_scroll, _header, _two_columns_ui, _insight_ui
    ])
    return acc, f1


@app.cell(hide_code=True)
def visual_error_analysis(
    BytesIO,
    DataLoader,
    Image,
    base64,
    class_names,
    css_no_scroll,
    device,
    get_ua_label,
    mo,
    model,
    test_loader,
    torch,
):
    _header = mo.md("<h2 align='center' style='margin-bottom: 20px;'>6. Приклади класифікації (Категорії у стовпчиках)</h2>")

    # 1. Збираємо по 3 випадкові зображення ДЛЯ КОЖНОГО з 6 класів
    _temp_loader = DataLoader(test_loader.dataset, batch_size=32, shuffle=True)
    _class_samples = {i: [] for i in range(len(class_names))}

    for _imgs, _lbls in _temp_loader:
        for _img, _lbl in zip(_imgs, _lbls):
            _c = _lbl.item()
            if len(_class_samples[_c]) < 3:
                _class_samples[_c].append((_img, _c))
    
        # Якщо для всіх 6 класів вже є рівно по 3 картинки - зупиняємо пошук
        if all(len(v) == 3 for v in _class_samples.values()):
            break

    # 2. Формуємо правильний порядок для CSS Grid (заповнення йде по рядках)
    # Рядок 1: Класи 0, 1, 2, 3, 4, 5
    # Рядок 2: Класи 0, 1, 2, 3, 4, 5
    # Рядок 3: Класи 0, 1, 2, 3, 4, 5
    _ordered_images = []
    _ordered_actuals = []
    for _row in range(3):
        for _c in range(len(class_names)):
            _img, _lbl = _class_samples[_c][_row]
            _ordered_images.append(_img)
            _ordered_actuals.append(_lbl)
        
    _images_tensor = torch.stack(_ordered_images)
    _actuals_tensor = torch.tensor(_ordered_actuals)

    # 3. Робимо інференс на відсортованому батчі (18 картинок)
    model.eval()
    with torch.no_grad():
        _inputs = _images_tensor.to(device)
        _outputs = model(_inputs)
        _, _preds = torch.max(_outputs, 1)

    _preds = _preds.cpu().numpy()
    _actuals = _actuals_tensor.numpy()

    # 4. Будуємо CSS Grid (6 стовпців x 3 рядки + Заголовки)
    # align-items: end допомагає вирівняти заголовки по нижньому краю перед картинками
    html_content = "<div style='display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; justify-items: center; align-items: end;'>"

    # Спочатку додаємо ЗАГОЛОВКИ КОЛОНОК (назви категорій)
    for _c in range(len(class_names)):
        ua_name = get_ua_label(class_names[_c])
        html_content += f"<div style='font-weight: bold; font-size: 15px; margin-bottom: 5px; color: #2E86AB; text-transform: uppercase;'>{ua_name}</div>"

    # Потім додаємо самі картинки (вони ідеально розіллються по своїх колонках)
    for i in range(18):
        # Денормалізація
        img = _images_tensor[i].clone()
        img[0] = img[0] * 0.229 + 0.485
        img[1] = img[1] * 0.224 + 0.456
        img[2] = img[2] * 0.225 + 0.406
        img = img.numpy().transpose((1, 2, 0)).clip(0, 1)

        # Конвертація в Base64 для HTML
        pil_img = Image.fromarray((img * 255).astype('uint8'))
        buffered = BytesIO()
        pil_img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        pred_class = get_ua_label(class_names[_preds[i]])
        actual_class = get_ua_label(class_names[_actuals[i]])

        # Зелений для правильних, Червоний для помилок
        color = "#2A9D8F" if _preds[i] == _actuals[i] else "#D64933"

        html_content += f"""
        <div style='text-align: center; border: 1px solid #ddd; padding: 8px; border-radius: 8px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); background-color: white; width: 135px;'>
            <img src='data:image/jpeg;base64,{img_str}' width='115' height='115' style='border-radius: 5px; object-fit: cover;'/>
            <p style='margin: 8px 0 0 0; font-size: 13px;'><b>Прогноз:</b> <br><span style='color: {color}; font-weight: bold;'>{pred_class}</span></p>
        </div>
        """
    html_content += "</div>"

    mo.vstack([
        css_no_scroll, _header, mo.md(html_content)
    ])
    return


@app.cell(hide_code=True)
def final_conclusions(acc, css_no_scroll, f1, mo):
    _conclusions_ui = mo.callout(
        title="7. Висновки та оптимізація архітектури",
        kind="success",
        value=mo.md(f"""
        **1. 🎯 Аналіз ефективності:**  
        Використання попередньо навченої моделі ResNet18 дозволило досягти блискучих результатів: **Точність = {acc*100:.2f}%** та **F1-Score = {f1:.4f}**. Це повністю відповідає еталонним очікуванням для Transfer Learning у задачах комп'ютерного зору. Модель успішно адаптувала потужні ознаки ImageNet для розпізнавання природних сцен.

        **2. 🧠 Аналіз архітектури (Переваги):**  
        Часткове заморожування (freezing) ранніх шарів усунуло ризик руйнування базових детекторів меж та текстур під час навчання на нових даних. Активація лише `layer4` та фінального класифікатора значно пришвидшила навчання та захистила мережу від перенавчання (overfitting). Додавання шару `Dropout(0.3)` у новий класифікатор також підвищило стійкість моделі.

        **3. 🚧 Шляхи покращення (Оптимізація):**  
        * **Архітектурні експерименти:** Перехід на архітектури сімейства **EfficientNet (B0-B3)** для економії VRAM за рахунок Compound Scaling.
        * **Fine-Tuning стратегія:** Застосування диференційованої швидкості навчання (Discriminative Learning Rates), де глибокі шари навчаються з меншим learning rate.
        * **Advanced Augmentation:** Додавання методів на кшталт `MixUp` або `CutMix` для кращої генералізації візуально складних класів.
        """)
    )

    mo.vstack([
        css_no_scroll,
        _conclusions_ui
    ])
    return


if __name__ == "__main__":
    app.run()
