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
        🛋️ ДЗ №3: Детекція об'єктів у приміщенні <i>(YOLO26)</i>
    </div>
    <h3 align="center"><b><u>Пайплайн</u>: Аудит YOLO-набору даних ➔ YOLO26s (NMS-Free) ➔ Криві навчання ➔ Метрики по класах ➔ Матриця помилок ➔ Візуалізація детекції</b></h3>
    <p align="center"><i>© Oleh Hatsenko (IRONKAGE) | Deep Learning for Computer Vision and NLP</i></p>
    """)
    return (css_no_scroll,)


@app.cell(hide_code=True)
def imports():
    import os
    import sys
    import time
    import platform

    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import marimo as mo
    from PIL import Image
    import torch
    import ultralytics
    from ultralytics import YOLO

    # --- ДИНАМІЧНИЙ ШЛЯХ ---
    _root = os.path.abspath(os.path.join(os.getcwd(), "..")) if "notebooks_eda" in os.getcwd() else os.getcwd()
    if _root not in sys.path:
        sys.path.insert(0, _root)

    from core import get_hardware_config, log_system_info, set_global_seed
    from hw_03_indoor_objects import analysis as an
    from hw_03_indoor_objects import data_adapter as da
    from hw_03_indoor_objects import ui_labels as ui

    PROJECT_ROOT = da.PROJECT_ROOT
    # Клас вважаємо «розпізнаним», якщо AP@0.5 ≥ порогу = нижня межа очікуваного mAP@0.5 з ТЗ (40%)
    AP50_THRESHOLD = 0.40

    set_global_seed(42)
    log_system_info()
    device, device_name = get_hardware_config()

    mo.center(mo.md(f"✅ **Інфраструктура ініціалізована! Апаратне прискорення:** `{device_name}`"))
    return (
        AP50_THRESHOLD,
        Image,
        PROJECT_ROOT,
        YOLO,
        an,
        da,
        device,
        device_name,
        go,
        make_subplots,
        mo,
        np,
        os,
        pd,
        platform,
        px,
        time,
        torch,
        ui,
        ultralytics,
    )


@app.cell(hide_code=True)
def environment(
    css_no_scroll,
    device,
    device_name,
    mo,
    pd,
    platform,
    torch,
    ultralytics,
):
    _header = mo.md("<h2 align='center' style='margin-bottom: 20px;'>0. Робоче середовище</h2>")

    _type = getattr(device, "type", str(device))
    _rows = pd.DataFrame({
        "Параметр": ["Операційна система", "Python", "PyTorch", "Ultralytics", "Пристрій обчислень (core.hardware)"],
        "Значення": [platform.platform(), platform.python_version(), torch.__version__, ultralytics.__version__, device_name],
    })

    if _type == "cuda":
        _kind, _text = "success", "**NVIDIA GPU (CUDA)** — відповідає вимозі ТЗ щодо власної машини з NVIDIA GPU."
    elif _type == "mps":
        _kind, _text = "warn", (
            "Навчання виконується на графічному прискорювачі, точну конфігурацію вказано в таблиці нижче:"
        )
    else:
        _kind, _text = "danger", "**GPU не використовується (CPU).** Вимога ТЗ щодо середовища з GPU не виконана; навчання буде значно повільнішим."

    mo.vstack([css_no_scroll, _header, mo.callout(mo.md(_text), kind=_kind), mo.ui.table(_rows, selection=None)])
    return


@app.cell(hide_code=True)
def data_ops(an, css_no_scroll, da, mo, pd, time, ui):
    _t0 = time.perf_counter()
    _header = mo.md("<h2 align='center' style='margin-bottom: 20px;'>1. Data Ops: Набір даних, аудит розмітки та YAML конфіг</h2>")

    with mo.status.spinner("Перевірка / завантаження набору даних та аудит розмітки..."):
        ds = da.prepare_dataset()

    _st = ds.splits_table().set_index("split").reindex([s for s in ("train", "val", "test") if s in ds.splits]).reset_index()
    _splits_ua = pd.DataFrame({
        "Вибірка": _st["split"].map(ui.get_split_label),
        "Зображень": _st["images"],
        "Файлів розмітки": _st["label_files"],
        "Зображень без розмітки": _st["no_label"],
        "Розмітка без зображення": _st["orphan_labels"],
        "Об'єктів (рамок)": _st["boxes"],
        "Некоректних рядків": _st["bad_lines"],
    })

    _ct = ds.class_table()
    _classes_ua = pd.DataFrame({"ID": _ct["id"], "Назва в YAML (англ.)": _ct["name"], "Українською": _ct["name"].map(ui.get_indoor_label)})
    for _s in ("train", "val", "test"):
        if _s in _ct.columns:
            _classes_ua[f"Об'єктів: {ui.get_split_label(_s)}"] = _ct[_s]

    with open(ds.yaml_path, "r", encoding="utf-8") as _f:
        _yaml_content = _f.read()

    _insight_loc = mo.callout(
        title="💡 Tech Lead Insight (Localization):",
        kind="info",
        value=mo.md(
            "У файлі `yaml` назви класів **свідомо лишаються англійськими**: внутрішні графіки Ultralytics (OpenCV) не вміють кирилицю й "
            "малюють «биті» шрифти `[][][]`. Тому всі таблиці та графіки дашборду українізуємо «на льоту», а рамки на зображеннях малюємо "
            "власним рендером (PIL + шрифт DejaVu з matplotlib)."
        ),
    )

    if ds.issues:
        _issues_ui = mo.callout(mo.md("**Зауваження до набору даних:**\n\n" + "\n".join(f"- {i}" for i in ds.issues)), kind="warn")
    else:
        _issues_ui = mo.callout(mo.md("✅ **Аудит пройдено:** кожне зображення має відповідний файл розмітки, рядки YOLO коректні, класи в межах діапазону."), kind="success")

    _footer = mo.md(
        f"✅ **Набір даних готовий за {an.fmt_elapsed(time.perf_counter() - _t0)}.** Класів: **{ds.nc}** "
        f"(імена взято з: {ds.names_source}). Порядок класів у YAML збігається з розміткою набору даних."
    )

    mo.vstack([
        css_no_scroll, _header, _insight_loc, _issues_ui,
        mo.md("**Вибірки (train / validation / test):**"), mo.ui.table(_splits_ua, selection=None),
        mo.md("**Класи (порядок = індекси в розмітці):**"), mo.ui.table(_classes_ua, selection=None),
        mo.md(f"**Згенерований конфіг (`{ds.yaml_path}`):**"),
        mo.ui.code_editor(value=_yaml_content, language="yaml", disabled=True),
        mo.center(_footer),
    ])
    return (ds,)


@app.cell(hide_code=True)
def class_distribution(css_no_scroll, ds, mo, np, pd, px, ui):
    _header = mo.md("<h2 align='center' style='margin-bottom: 20px;'>2. EDA: Розподіл класів та розмірів об'єктів</h2>")

    _rows = [{"Клас": ui.get_indoor_label(_n), "Вибірка": ui.get_split_label(_s), "Кількість": _st.class_counts.get(_i, 0)}
             for _s, _st in ds.splits.items() for _i, _n in enumerate(ds.names)]
    _long = pd.DataFrame(_rows)
    _fig_cls = px.bar(_long, y="Клас", x="Кількість", color="Вибірка", orientation="h", barmode="group",
                      title="Кількість об'єктів за класами та вибірками",
                      labels={"Кількість": "Кількість об'єктів", "Клас": "Клас", "Вибірка": "Вибірка"})
    _fig_cls.update_layout(height=520, title_x=0.5)

    _sz = ds.splits["train"].sizes
    if len(_sz):
        _rel = np.sqrt(_sz[:, 0] * _sz[:, 1])
        _small = float((_rel < 0.10).mean())
        _fig_sz = px.histogram(x=_rel, nbins=40, title="Відносний розмір рамок у навчальній вибірці",
                               labels={"x": "Відносний розмір рамки, √(ширина·висота)", "count": "Кількість"})
        _fig_sz.update_layout(height=420, title_x=0.5, yaxis_title="Кількість об'єктів")
        _size_ui = mo.as_html(_fig_sz)
    else:
        _small, _size_ui = float("nan"), mo.md("*Розмірів рамок немає.*")

    _train = ds.splits["train"].class_counts
    _tot = sum(_train.values()) or 1
    _top = max(range(ds.nc), key=lambda i: _train.get(i, 0))
    _bot = min(range(ds.nc), key=lambda i: _train.get(i, 0))
    _ratio = _train.get(_top, 0) / max(_train.get(_bot, 0), 1)
    _insight = mo.callout(
        title="💡 Tech Lead Insight (Data):",
        kind="info",
        value=mo.md(
            f"У навчальній вибірці найчастіший клас — **«{ui.get_indoor_label(ds.names[_top])}»** ({100 * _train.get(_top, 0) / _tot:.1f}% об'єктів), "
            f"найрідший — **«{ui.get_indoor_label(ds.names[_bot])}»** ({100 * _train.get(_bot, 0) / _tot:.1f}%); дисбаланс ≈ **{_ratio:.0f}×**. "
            f"Малі об'єкти (сторона < 10% зображення) становлять **{100 * _small:.0f}%** розмітки — вони зазвичай найскладніші для детекції."
        ),
    )

    mo.vstack([css_no_scroll, _header, _insight, mo.as_html(_fig_cls), _size_ui])
    return


@app.cell(hide_code=True)
def gt_controls(css_no_scroll, ds, mo):
    _header = mo.md("<h2 align='center' style='margin-bottom: 20px;'>2b. Візуалізація розмітки (Ground Truth)</h2>")
    _n = len(ds.splits["train"].image_paths)
    gt_slider = mo.ui.slider(start=0, stop=max(_n - 1, 0), step=1, value=0,
                             label=f"Зображення навчальної вибірки (№ з {_n})", full_width=True)
    mo.vstack([css_no_scroll, _header, gt_slider])
    return (gt_slider,)


@app.cell(hide_code=True)
def gt_viewer(Image, an, da, ds, gt_slider, mo, pd, ui):
    _p = ds.splits["train"].image_paths[int(gt_slider.value)]
    _img = Image.open(_p).convert("RGB")
    _boxes = da.read_yolo_labels(ds.splits["train"].labels_dir / (_p.stem + ".txt"), *_img.size)
    _drawn = an.draw_boxes(_img, _boxes, ds.names, ui.get_indoor_label)

    _counts = pd.Series([ui.get_indoor_label(ds.names[b["cls"]]) if 0 <= b["cls"] < ds.nc else f"клас {b['cls']}" for b in _boxes]).value_counts()
    _caption = (f"**{_p.name}** — {len(_boxes)} об'єктів: " + ", ".join(f"{k} × {v}" for k, v in _counts.items())) if len(_boxes) else f"**{_p.name}** — розмітки немає (фонове зображення)"

    mo.vstack([mo.center(mo.Html(an.pil_to_html(_drawn, max_width=760))), mo.center(mo.md(_caption))])
    return


@app.cell(hide_code=True)
def training_controls(css_no_scroll, mo):
    _header = mo.md("<h2 align='center' style='margin-bottom: 20px;'>3. Тренування моделі YOLO26 (NMS-Free)</h2>")

    model_choice = mo.ui.dropdown(
        options={"YOLO26s (рекомендовано, NMS-free)": "yolo26s.pt", "YOLO26n (найлегша)": "yolo26n.pt",
                 "YOLO11s (запасний варіант)": "yolo11s.pt", "YOLOv9s (як у ТЗ)": "yolov9s.pt"},
        value="YOLO26s (рекомендовано, NMS-free)", label="Модель")
    mode_choice = mo.ui.dropdown(
        options={"Швидка перевірка (1 епоха, 10% даних)": "smoke", "Повне навчання": "full"},
        value="Повне навчання", label="Режим")
    epochs_input = mo.ui.number(start=1, stop=300, step=1, value=10, label="Епох (повний режим)")
    optimizer_choice = mo.ui.dropdown(
        options={"auto (для YOLO26 — MuSGD)": "auto", "AdamW (запасний)": "AdamW", "SGD": "SGD"},
        value="auto (для YOLO26 — MuSGD)", label="Оптимізатор")
    device_choice = mo.ui.dropdown(options={"авто (core.hardware)": "auto", "CPU": "cpu"}, value="авто (core.hardware)", label="Пристрій")
    run_train = mo.ui.run_button(label="🚀 Запустити навчання", kind="success")

    _tip = mo.callout(mo.md(
        "**Порада:** при першому запуску обери «Швидка перевірка» — за хвилини стане видно, що пайплайн працює (завантаження ваг, MPS/CUDA, оптимізатор). "
        "Якщо `auto`-оптимізатор (MuSGD) або `yolo26s.pt` не працюють на твоїй версії torch/MPS — вибери AdamW або запасну модель. "
        "Ваги `best.pt` / `last.pt` зберігаються автоматично, а за наявності збережених ваг повторне навчання не потрібне."
    ), kind="info")

    mo.vstack([css_no_scroll, _header, _tip,
               mo.hstack([model_choice, mode_choice, epochs_input], justify="start", gap=2),
               mo.hstack([optimizer_choice, device_choice, run_train], justify="start", gap=2, align="end")])
    return (
        device_choice,
        epochs_input,
        mode_choice,
        model_choice,
        optimizer_choice,
        run_train,
    )


@app.cell(hide_code=True)
def training(
    PROJECT_ROOT,
    YOLO,
    an,
    device,
    device_choice,
    ds,
    epochs_input,
    mo,
    mode_choice,
    model_choice,
    optimizer_choice,
    os,
    run_train,
    time,
):
    _smoke = mode_choice.value == "smoke"
    _run = an.resolve_run(PROJECT_ROOT, model_choice.value, _smoke)
    runs_dir, run_name, weights_path, results_csv = _run["runs_dir"], _run["run_name"], _run["weights"], _run["results_csv"]
    yolo_dev = an.yolo_device(device, device_choice.value)
    _epochs = 1 if _smoke else int(epochs_input.value)

    train_stamp = None
    if run_train.value:
        _t0 = time.perf_counter()
        try:
            with mo.status.progress_bar(title=f"Навчання {model_choice.value} ({_epochs} епох) на {yolo_dev.upper()}", total=_epochs,
                                        remove_on_exit=True) as _bar:

                def _on_epoch_end(trainer):
                    _m = getattr(trainer, "metrics", None) or {}
                    _bar.update(subtitle=(f"Епоха {trainer.epoch + 1}/{trainer.epochs} · mAP@0.5: {_m.get('metrics/mAP50(B)', float('nan')):.3f} "
                                          f"· mAP@0.5:0.95: {_m.get('metrics/mAP50-95(B)', float('nan')):.3f}"))

                _model = YOLO(model_choice.value)
                _model.add_callback("on_fit_epoch_end", _on_epoch_end)
                _args = dict(data=ds.yaml_path, epochs=_epochs, imgsz=640, batch=16, device=yolo_dev, project=runs_dir,
                             name=run_name, exist_ok=True, plots=True, seed=42, workers=4)
                if _smoke:
                    _args["fraction"] = 0.1
                if optimizer_choice.value != "auto":
                    _args["optimizer"] = optimizer_choice.value
                _model.train(**_args)
            train_stamp = f"{time.time():.0f}"
            _status = mo.callout(mo.md(f"✅ **Тренування завершено за {an.fmt_elapsed(time.perf_counter() - _t0)}.** "
                                       f"Артефакти та ваги збережено у `{_run['run_dir']}`."), kind="success")
        except Exception as _e:  # noqa: BLE001
            _status = mo.callout(mo.md(
                f"❌ **Навчання не вдалося:** `{type(_e).__name__}: {_e}`\n\n"
                "Що спробувати: оптимізатор **AdamW** (якщо проблема в MuSGD), інша модель (**YOLO11s** / **YOLOv9s**), пристрій **CPU**, "
                "режим «Швидка перевірка» для діагностики. Надішли повний traceback з терміналу."), kind="danger")
    elif os.path.exists(weights_path):
        _status = mo.callout(mo.md(f"ℹ️ Знайдено збережені ваги `{weights_path}` — тренування пропущено, аналіз нижче працює з ними."), kind="info")
    else:
        _status = mo.md("*Натисни «Запустити навчання». Логи Ultralytics відображатимуться в терміналі, прогрес — тут.*")

    _status
    return results_csv, run_name, runs_dir, weights_path, yolo_dev


@app.cell(hide_code=True)
def learning_curves(
    an,
    css_no_scroll,
    go,
    make_subplots,
    mo,
    px,
    results_csv,
    ui,
):
    _header = mo.md("<h2 align='center' style='margin-bottom: 20px;'>4. Криві навчання (Learning Curves)</h2>")

    _df = an.read_results(results_csv)
    fit = None
    if _df is None:
        _body = mo.md("*Графіки будуть доступні після завершення тренування...*")
    else:
        fit = an.fit_diagnostics(_df)
        _pairs = an.loss_pairs(_df)
        _fig_loss = make_subplots(rows=1, cols=max(len(_pairs), 1), subplot_titles=[ui.get_metric_label(p) for p in _pairs])
        for _k, _p in enumerate(_pairs, start=1):
            _fig_loss.add_trace(go.Scatter(x=_df["epoch"], y=_df[f"train/{_p}"], mode="lines+markers", name="Навчання (train)",
                                           line=dict(color="#636EFA"), showlegend=(_k == 1)), row=1, col=_k)
            _fig_loss.add_trace(go.Scatter(x=_df["epoch"], y=_df[f"val/{_p}"], mode="lines+markers", name="Валідація (val)",
                                           line=dict(color="#EF553B", dash="dash"), showlegend=(_k == 1)), row=1, col=_k)
            _fig_loss.update_xaxes(title_text="Епоха", row=1, col=_k)
        _fig_loss.update_layout(height=420, title_text="Втрати: навчання vs валідація (розбіжність → перенавчання)", title_x=0.5,
                                legend_title_text="Вибірка")

        _cols = [c for c in ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)") if c in _df.columns]
        _dfm = _df[["epoch"] + _cols].rename(columns={c: ui.get_metric_label(c) for c in _cols})
        _fig_map = px.line(_dfm, x="epoch", y=[ui.get_metric_label(c) for c in _cols], markers=True,
                           title="Динаміка метрик на валідації (Precision, Recall, mAP)",
                           labels={"value": "Значення метрики", "variable": "Метрика", "epoch": "Епоха"})
        _fig_map.update_layout(height=420, title_x=0.5, legend_title_text="Метрика")

        _kind = {"overfit": "warn", "underfit": "warn", "ok": "success"}.get(fit["verdict"], "info")
        _diag = mo.callout(mo.md(f"**Діагностика кривих:** {an.describe_fit(fit)}"), kind=_kind)
        _body = mo.vstack([_diag, mo.as_html(_fig_loss), mo.as_html(_fig_map)])

    mo.vstack([css_no_scroll, _header, _body])
    return (fit,)


@app.cell(hide_code=True)
def evaluation(
    YOLO,
    an,
    css_no_scroll,
    ds,
    go,
    mo,
    np,
    os,
    px,
    run_name,
    runs_dir,
    time,
    ui,
    weights_path,
    yolo_dev,
):
    _header = mo.md("<h2 align='center' style='margin-bottom: 20px;'>5. Оцінка моделі на валідаційній вибірці: метрики по класах</h2>")

    val_df, overall, cm_matrix = None, None, None
    if not os.path.exists(weights_path):
        _body = mo.md("*Метрики з'являться після тренування (потрібні ваги `best.pt`)...*")
    else:
        _t0 = time.perf_counter()
        with mo.status.spinner("Валідація найкращої моделі (best.pt) на валідаційній вибірці..."):
            _m = YOLO(weights_path).val(data=ds.yaml_path, split="val", imgsz=640, batch=16, device=yolo_dev, plots=True,
                                        project=runs_dir, name=f"{run_name}_val", exist_ok=True, verbose=False)
        val_df = an.per_class_table(_m.box, ds.names, ui.get_indoor_label, ds.support("val"))
        overall = an.overall_metrics(_m.box, val_df)
        cm_matrix = np.asarray(_m.confusion_matrix.matrix)
        _acc = an.classification_accuracy(cm_matrix, ds.nc)
        _speed = dict(getattr(_m, "speed", {}) or {})

        _stats = mo.hstack([
            mo.stat(value=an.pct(overall["map50"]), label="mAP@0.5", caption="орієнтир ТЗ: 40–60%", bordered=True),
            mo.stat(value=an.pct(overall["map95"]), label="mAP@0.5:0.95", caption="орієнтир ТЗ: 25–40%", bordered=True),
            mo.stat(value=an.pct(overall["mp"]), label="Precision (середня)", caption="за класами", bordered=True),
            mo.stat(value=an.pct(overall["mr"]), label="Recall (середня)", caption="за класами: 50–70%", bordered=True),
            mo.stat(value=an.pct(overall["f1"]), label="F1 (середній)", caption="орієнтир ТЗ: 45–60%", bordered=True),
            mo.stat(value=an.pct(_acc), label="Точність класифікації", caption="за матрицею помилок; ТЗ: 55–70%", bordered=True),
        ], justify="center", gap=1)

        _ranked = val_df.dropna(subset=[an.COL_F1]).sort_values(an.COL_F1, ascending=False)
        _long = _ranked.melt(id_vars=[an.COL_CLASS], value_vars=[an.COL_P, an.COL_R, an.COL_F1], var_name="Метрика", value_name="Значення")
        _fig_prf = px.bar(_long, y=an.COL_CLASS, x="Значення", color="Метрика", barmode="group", orientation="h",
                          category_orders={an.COL_CLASS: _ranked[an.COL_CLASS].tolist()},
                          title="Precision, Recall та F1 за класами (від найкращого до найгіршого за F1)",
                          labels={an.COL_CLASS: "Клас", "Значення": "Значення метрики"})
        _fig_prf.update_yaxes(autorange="reversed")
        _fig_prf.update_layout(height=160 + 60 * len(_ranked), title_x=0.5, legend_title_text="Метрика", xaxis_range=[0, 1])

        _fig_ap = go.Figure()
        _fig_ap.add_bar(y=_ranked[an.COL_CLASS], x=_ranked[an.COL_AP50], name="AP@0.5", orientation="h")
        _fig_ap.add_bar(y=_ranked[an.COL_CLASS], x=_ranked[an.COL_AP], name="AP@0.5:0.95", orientation="h")
        _fig_ap.add_vline(x=0.40, line_dash="dash", annotation_text="поріг «розпізнаного» класу (40%)")
        _fig_ap.update_layout(barmode="group", title="Average Precision за класами", title_x=0.5, height=160 + 60 * len(_ranked),
                              xaxis_title="Значення AP", yaxis_title="Клас", legend_title_text="Метрика", xaxis_range=[0, 1])
        _fig_ap.update_yaxes(autorange="reversed")

        _absent = val_df[val_df[an.COL_F1].isna()][an.COL_CLASS].tolist()
        _best, _worst = _ranked.iloc[0], _ranked.iloc[-1]
        _note = (f"Найкраще визначається **«{_best[an.COL_CLASS]}»** (F1 {an.pct(_best[an.COL_F1])}, AP@0.5 {an.pct(_best[an.COL_AP50])}), "
                 f"найгірше — **«{_worst[an.COL_CLASS]}»** (F1 {an.pct(_worst[an.COL_F1])}, AP@0.5 {an.pct(_worst[an.COL_AP50])}). "
                 "Precision/Recall/F1 по класах Ultralytics обчислює при порозі впевненості, що максимізує F1.")
        if _absent:
            _note += " Класи без об'єктів у валідаційній вибірці (метрики недоступні): " + ", ".join(f"«{c}»" for c in _absent) + "."
        if _speed:
            _note += (f" Швидкість на зображення: препроцес {_speed.get('preprocess', float('nan')):.1f} мс, "
                      f"інференс {_speed.get('inference', float('nan')):.1f} мс, постпроцес {_speed.get('postprocess', float('nan')):.1f} мс.")

        _table = mo.ui.table(val_df.drop(columns=[an.COL_ID]).round(3), selection=None, page_size=12)
        _footer = mo.md(f"✅ **Оцінку виконано за {an.fmt_elapsed(time.perf_counter() - _t0)}.**")
        _body = mo.vstack([mo.callout(mo.md(_note), kind="info"), _stats, _table, mo.as_html(_fig_prf), mo.as_html(_fig_ap), mo.center(_footer)])

    mo.vstack([css_no_scroll, _header, _body])
    return cm_matrix, overall, val_df


@app.cell(hide_code=True)
def confusion(an, cm_matrix, css_no_scroll, ds, go, mo, np, pd, ui):
    _header = mo.md("<h2 align='center' style='margin-bottom: 20px;'>6. Матриця помилок (Confusion Matrix) та міжкласові плутанини</h2>")

    top_conf = []
    if cm_matrix is None:
        _body = mo.md("*Матриця помилок з'явиться після оцінки моделі...*")
    else:
        _labels = [ui.get_indoor_label(n) for n in ds.names] + ["Фон / пропуск"]
        _z = an.confusion_true_pred(cm_matrix, normalize=True)
        _text = np.where(_z > 0.005, np.round(_z, 2).astype(str), "")
        _fig = go.Figure(go.Heatmap(z=_z, x=_labels, y=_labels, colorscale="Blues", zmin=0, zmax=1, text=_text, texttemplate="%{text}",
                                    hovertemplate="Справжній: %{y}<br>Передбачено: %{x}<br>Частка: %{z:.2f}<extra></extra>",
                                    colorbar=dict(title=dict(text="Частка"))))
        _fig.update_yaxes(autorange="reversed", title_text="Справжній клас")
        _fig.update_xaxes(title_text="Передбачений клас", tickangle=-35)
        _fig.update_layout(title="Нормалізована матриця помилок (рядок = справжній клас; остання колонка — пропущені об'єкти)",
                           title_x=0.5, height=720)

        top_conf = an.top_confusions(cm_matrix, ds.names, k=8)
        _conf_df = pd.DataFrame({
            "Справжній клас": [ui.get_indoor_label(t["true"]) for t in top_conf],
            "Модель передбачила": [ui.get_indoor_label(t["pred"]) for t in top_conf],
            "Частка від справжнього класу, %": [round(100 * t["fraction"], 1) for t in top_conf],
            "Кількість": [t["count"] for t in top_conf],
            "Можлива причина (гіпотеза)": [ui.get_confusion_reason(t["true"], t["pred"]) for t in top_conf],
        })
        _bg = an.background_rates(cm_matrix, ds.names)
        _bg_df = pd.DataFrame({
            "Клас": _bg["name"].map(ui.get_indoor_label),
            "Пропущено (FN), %": (100 * _bg["missed"]).round(1),
            "Хибні спрацювання (FP), % від передбачень": (100 * _bg["false_pos"]).round(1),
        })
        _hint = mo.callout(mo.md(
            "**Як читати:** діагональ — правильні відповіді; поза діагоналлю — міжкласові помилки; остання колонка — об'єкти, яких модель "
            "**не знайшла** (FN), останній рядок — **хибні спрацювання** без реального об'єкта (FP). Причини в таблиці — **гіпотези** на основі природи класів, "
            "а не доведені факти: їх варто перевірити, переглянувши приклади помилок."), kind="info")
        _body = mo.vstack([_hint, mo.as_html(_fig),
                           mo.md("**Найчастіші міжкласові помилки:**"), mo.ui.table(_conf_df, selection=None),
                           mo.md("**Пропуски (FN) та хибні спрацювання (FP) за класами:**"), mo.ui.table(_bg_df, selection=None)])

    mo.vstack([css_no_scroll, _header, _body])
    return (top_conf,)


@app.cell(hide_code=True)
def detection_gallery(
    Image,
    YOLO,
    an,
    css_no_scroll,
    da,
    ds,
    mo,
    np,
    os,
    ui,
    weights_path,
    yolo_dev,
):
    _header = mo.md("<h2 align='center' style='margin-bottom: 20px;'>7. Візуалізація детекції: зображення та розпізнані об'єкти</h2>")

    _split = "test" if "test" in ds.splits else "val"
    _paths = ds.splits[_split].image_paths
    if not os.path.exists(weights_path) or not _paths:
        _body = mo.md("*Візуалізація з'явиться після тренування моделі...*")
    else:
        _model = YOLO(weights_path)
        _idx = np.random.default_rng(42).choice(len(_paths), size=min(4, len(_paths)), replace=False)
        _rows = []
        for _i in _idx:
            _p = _paths[int(_i)]
            _img = Image.open(_p).convert("RGB")
            _gt = da.read_yolo_labels(ds.splits[_split].labels_dir / (_p.stem + ".txt"), *_img.size)
            _res = _model.predict(source=str(_p), conf=0.25, imgsz=640, device=yolo_dev, verbose=False)[0]
            _xyxy = _res.boxes.xyxy.cpu().numpy()
            _cls = _res.boxes.cls.cpu().numpy().astype(int)
            _conf = _res.boxes.conf.cpu().numpy()
            _pred = [{"cls": int(c), "xyxy": tuple(b), "conf": float(s)} for b, c, s in zip(_xyxy, _cls, _conf)]
            _left = an.draw_boxes(_img, _gt, ds.names, ui.get_indoor_label, color=(34, 160, 60))
            _right = an.draw_boxes(_img, _pred, ds.names, ui.get_indoor_label, show_conf=True)
            _rows.append(mo.vstack([
                mo.md(f"**{_p.name}** — розмітка: {len(_gt)} об'єктів · модель знайшла: {len(_pred)}"),
                mo.hstack([mo.Html(an.pil_to_html(_left, max_width=560)), mo.Html(an.pil_to_html(_right, max_width=560))], justify="center", gap=1),
            ]))

        _note = mo.callout(mo.md(
            f"Зображення з **{'тестової' if _split == 'test' else 'валідаційної'}** вибірки (фіксований випадковий вибір). "
            "Ліворуч — **розмітка (зелені рамки)**, праворуч — **передбачення моделі** з підписом класу українською та впевненістю "
            "(поріг `conf = 0.25`). Для моделей YOLO26 end-to-end інференс не потребує NMS (за документацією Ultralytics)."), kind="success")
        _body = mo.vstack([_note] + _rows)

    mo.vstack([css_no_scroll, _header, _body])
    return


@app.cell(hide_code=True)
def criteria_check(
    AP50_THRESHOLD,
    an,
    cm_matrix,
    css_no_scroll,
    device,
    ds,
    fit,
    mo,
    os,
    pd,
    val_df,
    weights_path,
):
    _header = mo.md("<h2 align='center' style='margin-bottom: 20px;'>8. Чек-ліст критеріїв прийняття ДЗ</h2>")

    _type = getattr(device, "type", str(device))
    _n_ok = len(an.classes_detected(val_df, AP50_THRESHOLD)) if val_df is not None else 0
    _rows = [
        ("1. Код виконується", "✅", "ноутбук дійшов до цього кроку"),
        ("2. Набір даних завантажено", "✅", f"{sum(s.n_images for s in ds.splits.values())} зображень у {len(ds.splits)} вибірках"),
        ("3. Робоче середовище з GPU", {"cuda": "✅", "mps": "⚠️"}.get(_type, "❌"),
        {"cuda": "NVIDIA GPU (CUDA)", "mps": "GPU через MPS (не NVIDIA/Kaggle/Colab) - на ваш вибір"}.get(_type, "CPU — GPU не використано")),
        ("4. YAML-конфіг та навчання YOLO", "✅" if os.path.exists(ds.yaml_path) and os.path.exists(weights_path) else "❌", "yaml створено; ваги best.pt збережено" if os.path.exists(weights_path) else "ваг ще немає — запусти навчання"),
        ("5. Оцінка на валідації (mAP, precision, recall)", "✅" if val_df is not None else "❌", "метрики розраховано" if val_df is not None else "ще не виконано"),
        ("6. Аналіз результатів (криві та класи)", "✅" if (fit is not None and cm_matrix is not None) else "❌", "криві, метрики по класах, матриця помилок" if fit is not None else "ще не виконано"),
        (f"Умова: ≥ 3 з 10 класів розпізнаються (AP@0.5 ≥ {AP50_THRESHOLD:.0%})", "✅" if _n_ok >= 3 else "❌", f"розпізнано класів: {_n_ok}"),
        ("Умова: аналіз та ≥ 1 ідея покращення у звіті", "✅" if val_df is not None else "❌", "див. розділ «Висновки»"),
    ]
    _df = pd.DataFrame(_rows, columns=["Критерій", "Статус", "Деталі"])
    mo.vstack([css_no_scroll, _header, mo.ui.table(_df, selection=None)])
    return


@app.cell(hide_code=True)
def conclusions(
    AP50_THRESHOLD,
    an,
    css_no_scroll,
    fit,
    mo,
    model_choice,
    overall,
    top_conf,
    ui,
    val_df,
):
    if val_df is None or fit is None:
        _ui_out = mo.callout(mo.md("*Висновки сформуються автоматично після тренування та оцінки моделі.*"), kind="neutral")
    else:
        _model = model_choice.value.replace(".pt", "")
        _sentences = an.build_summary(_model, fit["n_epochs"], overall, val_df, top_conf, fit, AP50_THRESHOLD, ui.get_indoor_label, ui.get_confusion_reason)
        _recs = an.build_recommendations(val_df, fit, top_conf, ui.get_indoor_label)

        _summary_md = "\n".join(f"{i}. {s}" for i, s in enumerate(_sentences, start=1))
        _recs_md = "\n".join(f"- {r}" for r in _recs)
        _ui_out = mo.callout(
            title="9. Захист ДЗ: Аналіз та Висновки",
            kind="warn",
            value=mo.md(f"""
    **🎯 Висновки за отриманими результатами:**

    {_summary_md}

    **🏗️ Вибір архітектури та її обмеження:** взято pretrained `{_model}` (transfer learning з COCO): невеликий варіант `s` — розумний баланс швидкості й точності
    для навчального прототипу на обмеженому залізі, а end-to-end архітектура YOLO26 не потребує NMS на інференсі (за документацією Ultralytics).
    Обмеження: модель відносно нова, порівняння з іншими версіями (YOLOv9s, YOLO11s) у цьому ноутбуці **не проводилось**; навчання тривало лише {fit['n_epochs']} епох;
    якість обмежена обсягом та узгодженістю розмітки набору даних.

    **🚀 Пропозиції щодо покращення:**

    {_recs_md}

    *Висновки та рекомендації сформовано автоматично з фактичних метрик цього запуску; причини плутанини — гіпотези, які потребують перевірки.*
    """),
        )

    mo.vstack([css_no_scroll, _ui_out])
    return


if __name__ == "__main__":
    app.run()
