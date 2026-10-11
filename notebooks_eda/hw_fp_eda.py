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
        🏆 Фінальний Проект: Прогнозування Швидкості Прилаштування Тварин</br><i>(Tri-Framework Multimodal Stack)</i>
    </div>
    <h3 align="center"><b><u>Пайплайн</u>: 🔶 TensorFlow (tf.data + EfficientNetV2) ┃ 🔥 PyTorch (CLIP + Multilingual Transformers) ┃ 🧠 Keras 3 (Late-Fusion) ➔ Stacking ➔ Threshold Tuning (QWK)</b></h3>
    <p align="center"><i>© Oleh Hatsenko (IRONKAGE) | Deep Learning for Computer Vision and NLP</i></p>
    """)
    return (css_no_scroll,)


@app.cell(hide_code=True)
def imports():
    import os
    import time

    os.environ["KERAS_BACKEND"] = "torch"          # Keras-голова працює на PyTorch
    os.environ.setdefault("KERAS_TORCH_DEVICE", "cpu")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.express as px

    from core import get_hardware_config, log_system_info, set_global_seed
    from hw_fp_kaggle_petfinder import data_adapter as da
    from hw_fp_kaggle_petfinder import extractors as ex
    from hw_fp_kaggle_petfinder import model as fm
    from hw_fp_kaggle_petfinder import stacking as st
    from hw_fp_kaggle_petfinder.timing import fmt_elapsed
    from hw_fp_kaggle_petfinder import ui_labels as ui
    from hw_fp_kaggle_petfinder.ui_labels import get_adoption_label, get_column_label

    log_system_info()
    _device, _device_name = get_hardware_config()   # маршрутизатор заліза з core.hardware (поважає ML_BACKEND з .env)
    mo.center(mo.md(f"✅ **Omni-Інфраструктура (TF ⟷ PyTorch ⟷ Keras 3) ініціалізована! Пристрій PyTorch: {_device_name}**"))
    return (
        da,
        ex,
        fm,
        fmt_elapsed,
        get_adoption_label,
        get_column_label,
        mo,
        np,
        os,
        pd,
        px,
        set_global_seed,
        st,
        time,
        ui,
    )


@app.cell(hide_code=True)
def data_ops(
    css_no_scroll,
    da,
    get_adoption_label,
    get_column_label,
    mo,
    pd,
    px,
    set_global_seed,
):
    _header = mo.md("<h2 align='center' style='margin-bottom: 20px;'>1. Data Ops: Приватне змагання (лише train/test цього курсу)</h2>")

    set_global_seed(42)
    with mo.status.spinner("Перевірка / завантаження даних приватного змагання..."):
        da.ensure_data()
        comp = da.load_competition(use_core_reader=True)   # smart_read_csv із core.etl_core
    tab_tree, tab_nn = da.build_tabular(comp)

    _schema = pd.DataFrame({
        "колонка": list(comp.train.columns),
        "назва (UA)": [get_column_label(c) for c in comp.train.columns],
        "тип": [str(t) for t in comp.train.dtypes],
        "унікальних": [int(comp.train[c].nunique()) for c in comp.train.columns],
        "пропусків, %": [round(100 * float(comp.train[c].isna().mean()), 2) for c in comp.train.columns],
    })

    _dist = comp.train[comp.target_col].value_counts().sort_index().reset_index()
    _dist.columns = [comp.target_col, "Кількість"]
    _target_ua = get_column_label(comp.target_col)
    _dist["Категорія"] = _dist[comp.target_col].map(get_adoption_label)
    _fig = px.bar(_dist, x="Категорія", y="Кількість", text="Кількість", color=comp.target_col,
                  title=f"Розподіл швидкості прилаштування (train, {comp.n_train} тварин)",
                  labels={"Категорія": _target_ua, comp.target_col: _target_ua})
    _fig.update_layout(height=380, title_x=0.5)

    _n_img = tab_tree["n_images"]
    _info = mo.callout(
        kind="info",
        title="💡 Tech Lead Insight (Data):",
        value=mo.md(f"""Train: **{comp.n_train}**, Test: **{comp.n_test}**. Фото: у середньому **{_n_img.mean():.1f} на тварину** (0 фото у {(int((_n_img == 0).sum()))} тварин).\n
                Використовуємо ВСІ фото профілю (усереднення ембедингів), а не лише `-1.jpg`.\n
                Табличні колонки визначаються автоматично: **{tab_tree.shape[1]}** ознак для дерев / **{tab_nn.shape[1]}** для нейромережі."""),
    )
    mo.vstack([css_no_scroll, _header, _info, mo.ui.table(_schema, selection=None), mo.center(mo.as_html(_fig))])
    return comp, tab_nn, tab_tree


@app.cell(hide_code=True)
def features(comp, css_no_scroll, da, ex, fmt_elapsed, mo, pd, time, ui):
    _header = mo.md("<h2 align='center' style='margin-bottom: 20px;'>2. Feature Factory: кожен фреймворк у своїй зоні</h2>")

    _start_time = time.perf_counter()
    # Кожен екстрактор — окремий процес зі своїм бекендом; результати кешуються (*.npz) → повторні запуски миттєві.
    # Перший запуск (≈30–60 хв на i9) зручніше бачити в терміналі: python -m hw_fp_kaggle_petfinder.extractors all
    with mo.status.progress_bar(title="Екстракція ознак (TF → PyTorch CLIP → PyTorch Transformers)",
                                total=len(ex.JOBS), remove_on_exit=True) as _bar:
        def _on_job(_name, _phase, _status):
            if _phase == "start":
                _bar.update(increment=0, subtitle=f"{_name} ...")
            elif _phase == "progress":
                _bar.update(increment=0, subtitle=f"{_name} · {_status}")      # напр.: 45% (290/640) · 05:12<06:20, 1.1it/s
            else:
                _bar.update(increment=1, subtitle=f"{_name}: {_status}")

        _report = ex.ensure_all(da.DATA_DIR, on_job=_on_job)
    emb_blocks = da.load_embeddings(comp, da.DATA_DIR)
    _extraction_time_min = (time.perf_counter() - _start_time) / 60

    _rows = pd.DataFrame({"Блок ознак": [ui.get_block_label(k) for k in _report], "Статус": list(_report.values())})
    _dims = pd.DataFrame({"Блок ознак": [ui.get_block_label(k) for k in emb_blocks],
                         "Форма (тварин × ознак)": [str(v.shape) for v in emb_blocks.values()]})

    _insight = mo.callout(
        kind="info",
        title="💡 Tech Lead Insight (Frameworks):",
        value=mo.md(f"""🔶 **TensorFlow** — `tf.data` (паралелізоване декодування + prefetch) та EfficientNetV2 із **коректним діапазоном 0..255** (EfficientNet має вбудований rescaling: подавати пікселі 0..1 було б помилкою).\n
               🔥 **PyTorch** — CLIP ViT для картинок та multilingual-E5/MiniLM для тексту (MPS/CPU). **У цьому запуску реально завантажено блоки:** {', '.join(emb_blocks) or 'жодного'}.\n
               ***Ізоляція процесів усуває конфлікти libomp між TF і Torch на macOS.***"""),
    )

    _footer = mo.md(f"✅ **Екстракцію ознак завершено за {fmt_elapsed(_extraction_time_min * 60)}!** Знайдено {len(emb_blocks)} блоків даних.")

    mo.vstack([
        css_no_scroll,
        _header,
        _insight,
        mo.hstack([mo.ui.table(_rows, selection=None), mo.ui.table(_dims, selection=None)]),
        mo.center(_footer)
    ])
    return (emb_blocks,)


@app.cell(hide_code=True)
def level1_models(
    comp,
    css_no_scroll,
    da,
    emb_blocks,
    fmt_elapsed,
    mo,
    np,
    px,
    st,
    tab_tree,
    time,
    ui,
):
    _header = mo.md("<h2 align='center' style='margin-bottom: 20px;'>3a. Level-1: діагностика сигналу по модальностях (OOF, 5 фолдів)</h2>")

    folds = da.make_folds(comp)
    _n, _y = comp.n_train, comp.y

    l1_oof, l1_test = {}, {}
    _start_time = time.perf_counter()

    # Динамічно рахуємо кількість моделей для прогрес-бару
    # 2 (TF-IDF) + 2*N (Ridge+kNN для кожного emb_block) + 2 (HGB Tabular/All)
    _total_steps = 2 + (len(emb_blocks) * 2) + 2

    with mo.status.progress_bar(title="Тренування Level-1 моделей...", total=_total_steps, remove_on_exit=True) as _bar:

        _state = {"label": ""}

        def _step(_label):
            _state["label"] = _label.rstrip(".")
            _bar.update(increment=0, subtitle=_label)

        st.set_fold_hook(lambda _k, _K: _bar.update(increment=0, subtitle=f"{_state['label']} · фолд {_k}/{_K}"))

        _step("TF-IDF Ridge...")
        X_tfidf = st.build_tfidf(comp)
        l1_oof["tfidf_ridge"], l1_test["tfidf_ridge"] = st.ridge_sparse_oof(X_tfidf[:_n], _y, folds, X_tfidf[_n:])
        _bar.update(increment=1)

        _step("TF-IDF kNN...")
        svd_text = st.tfidf_svd(X_tfidf, 128)
        l1_oof["knn_tfidf"], l1_test["knn_tfidf"] = st.knn_oof(X_tfidf[:_n], _y, folds, X_tfidf[_n:])
        _bar.update(increment=1)

        for _name, _E in emb_blocks.items():
            _step(f"Ridge ({_name})...")
            l1_oof[f"ridge_{_name}"], l1_test[f"ridge_{_name}"] = st.ridge_dense_oof(_E[:_n], _y, folds, _E[_n:])
            _bar.update(increment=1)

            _step(f"kNN ({_name})...")
            l1_oof[f"knn_{_name}"], l1_test[f"knn_{_name}"] = st.knn_oof(_E[:_n], _y, folds, _E[_n:])
            _bar.update(increment=1)

        _step("HGB Tabular...")
        _tab = tab_tree.to_numpy(dtype=np.float32)
        l1_oof["hgb_tabular"], l1_test["hgb_tabular"] = st.hgb_oof(_tab[:_n], _y, folds, _tab[_n:])
        _bar.update(increment=1)

        _step("HGB All (Ensemble)...")
        _all = np.hstack([_tab] + [st.pca_block(_E, 24) for _E in emb_blocks.values()] + [svd_text[:, :24]])
        l1_oof["hgb_all"], l1_test["hgb_all"] = st.hgb_oof(_all[:_n], _y, folds, _all[_n:])
        _bar.update(increment=1)
        st.set_fold_hook(None)

    _training_time_min = (time.perf_counter() - _start_time) / 60

    _rep = ui.localize_report(st.model_report(l1_oof, _y, comp.lo, comp.hi))
    _fig = px.bar(_rep, y="Модель", x=["QWK (округлення)", "QWK (оптим. пороги)"], barmode="group", orientation="h",
                  title="QWK за модальностями (OOF, 5 фолдів)",
                  labels={"value": "QWK", "variable": "Метрика", "Модель": "Модель"})
    _fig.add_vline(x=0.5, line_dash="dash", annotation_text="0.5 = 100 балів")
    _fig.update_yaxes(autorange="reversed")            # найкраща модель — зверху
    _fig.update_layout(height=160 + 46 * len(_rep), title_x=0.5, legend_title_text="Метрика")

    _insight = mo.callout(
        kind="warn",
        title="🔎 Як читати діагностику:",
        value=mo.md(f"""Це показує, ЯКА модальність несе сигнал у цьому приватному наборі (топ лідерборду ≈0.8 — сигнал точно є).\n
               - Якщо домінує `tfidf_ridge`/`ridge_txt_*` — виграш у тексті;
               - якщо `hgb_tabular` — у метаданих;
               - якщо `ridge_img_*` — у фото."""),
    )

    _footer = mo.md(f"✅ **Level-1 моделі навчено за {fmt_elapsed(_training_time_min * 60)}!** Згенеровано {len(l1_oof)} OOF-прогнозів.")

    mo.vstack([
        css_no_scroll,
        _header,
        _insight,
        mo.ui.table(_rep, selection=None),
        mo.center(mo.as_html(_fig)),
        mo.center(_footer)
    ])
    return X_tfidf, folds, l1_oof, l1_test, svd_text


@app.cell(hide_code=True)
def signal_diagnostics(
    X_tfidf,
    comp,
    css_no_scroll,
    emb_blocks,
    fmt_elapsed,
    mo,
    pd,
    st,
    tab_tree,
    time,
    ui,
):
    _header = mo.md("<h2 align='center' style='margin-bottom: 20px;'>3b. Діагностика: де ховається сигнал?</h2>")

    _start_time = time.perf_counter()
    _n = comp.n_train

    _diag_blocks = {"tfidf": X_tfidf, **emb_blocks}
    with mo.status.progress_bar(title="Діагностика сусідів (косинусні матриці)", total=len(_diag_blocks),
                                remove_on_exit=True) as _bar:
        _corr = (tab_tree.iloc[:_n].corrwith(pd.Series(comp.y), method="spearman")
                 .dropna().sort_values(key=abs, ascending=False).head(12).round(3)
                 .rename("Кореляція Спірмена з класом").reset_index().rename(columns={"index": "Ознака"}))
        _corr["Ознака"] = _corr["Ознака"].map(ui.get_feature_label)

        _nb = st.neighbor_diagnostics(
            _diag_blocks, comp.y, _n,
            on_block=lambda _i, _tot, _name: _bar.update(subtitle=f"{ui.get_block_label(_name)} готово ({_i}/{_tot})"),
        )
        _nb = ui.localize_neighbors(_nb)

    _diag_time_min = (time.perf_counter() - _start_time) / 60

    _insight = mo.callout(
        kind="info",
        title="🔎 Як читати:",
        value=mo.md(f"""**Spearman** — чи несуть сигнал `n_images`, довжина тексту тощо (у цьому наборі лише PetID/Description/AdoptionSpeed + фото).\n
               **Сусіди:**
               - якщо багато train-тварин мають майже-дублікат (cos ≥ 0.95) і мітки сусідів збігаються значно частіше за випадок (~30–40%), сигнал у шаблонних описах/однакових притулках → працюють kNN-моделі.
               - Якщо сусідів мало, а збігів немає — сигнал семантичний (потрібен fine-tuning)."""),
    )

    _footer = mo.md(f"✅ **Діагностику завершено за {fmt_elapsed(_diag_time_min * 60)}!**")

    mo.vstack([
        css_no_scroll,
        _header,
        _insight,
        mo.hstack([mo.ui.table(_corr, selection=None), mo.ui.table(_nb, selection=None)]),
        mo.center(_footer)
    ])
    return


@app.cell(hide_code=True)
def keras_training(
    comp,
    css_no_scroll,
    emb_blocks,
    fm,
    fmt_elapsed,
    folds,
    mo,
    np,
    pd,
    px,
    svd_text,
    tab_nn,
    time,
):
    _header = mo.md("<h2 align='center' style='margin-bottom: 20px;'>4. Keras 3 (PyTorch backend): Late-Fusion по всіх модальностях</h2>")

    _blocks = fm.prepare_blocks(emb_blocks, tab_nn, svd_text)

    _start_time = time.perf_counter()
    _nf, _seeds, _epochs = int(folds.max()) + 1, 3, 7     # 7-епоховий запуск: мінімум val MSE на ~3-й епосі, далі перенавчання
    _hist = []
    with mo.status.progress_bar(title=f"Keras fusion: {_nf} фолдів × {_seeds} seed × {_epochs} епох ({', '.join(_blocks)})",
                                total=_nf * _seeds, remove_on_exit=True) as _bar:

        def _on_epoch(_tag, _ep, _total, _logs):          # живий стан усередині кожного навчання (як у старому _MarimoCallback)
            _bar.update(
                increment=0,
                subtitle=(f"{_tag} · епоха {_ep + 1}/{_total} | 📉 train loss: {_logs.get('loss', float('nan')):.4f} "
                          f"| 📈 val MSE: {_logs.get('val_loss', float('nan')):.4f}"),
            )

        def _on_fit(_k, _s, _nfolds, _nseeds):
            _bar.update(increment=1, subtitle=f"фолд {_k + 1}/{_nfolds} · seed {_s + 1}/{_nseeds} готово")

        keras_oof, keras_test = fm.keras_fusion_oof(
            _blocks, comp.y, folds, comp.n_train, epochs=_epochs, n_seeds=_seeds,
            on_epoch=_on_epoch, on_fit=_on_fit, history_out=_hist,
        )
    _training_time_min = (time.perf_counter() - _start_time) / 60

    _rmse = float(((keras_oof - comp.y) ** 2).mean() ** 0.5)

    _insight = mo.callout(
        kind="success",
        title="💡 Tech Lead Insight (Late-Fusion):",
        value="Усі модальності успішно злиті через багатовхідну мережу Keras. Навчання OOF (Out-Of-Fold) на 5 фолдах дає чесні OOF-прогнози для стекінгу (кожен об'єкт передбачений моделлю, яка його не бачила). Валідаційна крива на графіку — лише для моніторингу (без early-stopping і відбору ваг), тому OOF-прогнози лишаються чесними."
    )

    _footer = mo.md(f"✅ **Тренування завершено за {fmt_elapsed(_training_time_min * 60)}! OOF RMSE: {_rmse:.4f}**")

    _curves = pd.DataFrame({
        "Епоха": np.arange(1, _epochs + 1),
        "Навчання (train loss)": np.mean([h["loss"] for h in _hist], axis=0),
        "Валідація (val MSE)": np.mean([h["val_loss"] for h in _hist], axis=0),
    })
    _fig = px.line(_curves, x="Епоха", y=["Навчання (train loss)", "Валідація (val MSE)"],
                   title="Криві навчання (середнє по фолдах × seed)",
                   labels={"value": "Втрата (MSE)", "variable": "Крива"})
    _fig.update_layout(legend_title_text="Крива")
    _fig.update_layout(height=380, title_x=0.5)

    mo.vstack([css_no_scroll, _header, _insight, mo.center(mo.as_html(_fig)), mo.center(_footer)])
    return keras_oof, keras_test


@app.cell(hide_code=True)
def stack_and_submit(
    comp,
    css_no_scroll,
    da,
    fmt_elapsed,
    keras_oof,
    keras_test,
    l1_oof,
    l1_test,
    mo,
    os,
    pd,
    px,
    st,
    time,
    ui,
):
    _header = mo.md("<h2 align='center' style='margin-bottom: 20px;'>5. Стекінг, Threshold Tuning (QWK) та Submission</h2>")

    _oofs = {**l1_oof, "keras_fusion": keras_oof}
    _tests = {**l1_test, "keras_fusion": keras_test}

    _start_time = time.perf_counter()
    with mo.status.spinner("Стекінг + оптимізація порогів..."):
        _rep = st.model_report(_oofs, comp.y, comp.lo, comp.hi)
        final = st.stack(_oofs, _tests, comp.y, comp.lo, comp.hi)
    _stack_time_sec = time.perf_counter() - _start_time

    _submission = da.make_submission(comp, final["test_labels"])

    # MLOps: трекінг експерименту тим самим MLflow-налаштуванням, що й core.trainer.BaseTrainer
    try:
        import mlflow

        os.makedirs("mlruns", exist_ok=True)
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns/mlruns.db"))
        mlflow.set_experiment("hw_fp_petfinder_private")
        with mlflow.start_run(run_name="tri_framework_stack"):
            mlflow.log_params({"n_models": len(_oofs), "thresholds": [round(float(t), 4) for t in final["thresholds"]]})
            mlflow.log_metrics({"qwk_stack_oof": float(final["qwk_oof_thr"]), "qwk_stack_honest": float(final["qwk_honest"][0])})
            for _, _r in _rep.iterrows():
                mlflow.log_metric(f"qwk_{_r['модель']}", float(_r["QWK (пороги)"]))
            mlflow.log_artifact(os.path.join(comp.data_dir, "results", "submission.csv"))
    except Exception as _e:  # noqa: BLE001
        print(f"MLflow пропущено: {_e}")

    _sub_path = os.path.join(comp.data_dir, "results", "submission.csv")
    _btn = mo.download(data=_submission.to_csv(index=False).encode("utf-8"), filename="submission.csv",
                       label="📥 Завантажити submission.csv", mimetype="text/csv")

    _h_mean, _h_std = final["qwk_honest"]
    _score = min(100.0, max(0.0, _h_mean) * 200)
    _rep_ua = ui.localize_report(_rep)
    _w = pd.DataFrame({"Модель": [ui.get_model_label(k) for k in final["weights"]],
                       "Вага у стеку": list(final["weights"].values())})
    _cmp = pd.DataFrame({
        "Клас": range(comp.lo, comp.hi + 1),
        "Категорія": [ui.get_adoption_label(c) for c in range(comp.lo, comp.hi + 1)],
        "Train, %": [round(100 * float((comp.y == c).mean()), 1) for c in range(comp.lo, comp.hi + 1)],
        "Прогноз на test, %": [round(100 * float((final["test_labels"] == c).mean()), 1) for c in range(comp.lo, comp.hi + 1)],
    })
    _fig = px.bar(_rep_ua, y="Модель", x="QWK (оптим. пороги)", orientation="h", title="QWK окремих моделей і стеку (OOF)",
                  labels={"QWK (оптим. пороги)": "QWK (оптим. пороги)", "Модель": "Модель"})
    _fig.add_vline(x=final["qwk_oof_thr"], line_dash="dot", annotation_text=f"Стек: {final['qwk_oof_thr']:.3f}")
    _fig.add_vline(x=0.5, line_dash="dash", annotation_text="0.5 = 100 балів")
    _fig.update_yaxes(autorange="reversed")
    _fig.update_layout(height=160 + 46 * len(_rep_ua), title_x=0.5)

    _insight = mo.callout(
        kind="success",
        title="💡 Tech Lead Insight (Stacking & Thresholds):",
        value=mo.md(f"""Пороги (Nelder-Mead, 8 рестартів): **{[round(float(t), 3) for t in final['thresholds']]}**.\n
               QWK стеку: з округленням **{final['qwk_round_only']:.4f}**, з порогами **{final['qwk_oof_thr']:.4f}**, **чесна оцінка** (пороги на одній половині OOF, метрика на іншій) = **{_h_mean:.4f} ± {_h_std:.4f}** → орієнтовний бал LMS ≈ **{_score:.0f}/100** (за формулою ТЗ: 100 при QWK ≥ 0.5)."""),
    )

    _footer = mo.md(f"✅ **Стекінг та фінальну генерацію завершено за {fmt_elapsed(_stack_time_sec)}!**")

    mo.vstack([
        css_no_scroll, _header, _insight,
        mo.hstack([_btn, mo.md(f"*(або локально: `{os.path.abspath(_sub_path)}`)*")]),
        mo.ui.table(_rep_ua, selection=None),
        mo.hstack([mo.ui.table(_w, selection=None), mo.ui.table(_cmp, selection=None)]),
        mo.center(mo.as_html(_fig)),
        mo.ui.table(_submission.head(10), selection=None),
        mo.center(_footer)
    ])
    return (final,)


@app.cell(hide_code=True)
def final_fp_conclusions(css_no_scroll, emb_blocks, final, mo):
    _has_torch_blocks = any(k.startswith(("img_clip", "txt_")) for k in emb_blocks)
    _torch_item = (
        "<b>PyTorch</b> — CLIP ViT для зображень та multilingual E5/MiniLM для тексту (прискорення MPS/CUDA з автофолбеком на CPU)"
        if _has_torch_blocks else
        "<b>PyTorch</b> — бекенд Keras 3 для навчання fusion-голови; екстрактори PyTorch (CLIP, E5/MiniLM) у цьому запуску "
        "недоступні через несумісність версій transformers і torch, тому використані ембединги TF та TF-IDF"
    )
    _conclusions_ui = mo.callout(
        title="6. Захист Фінального Проекту: Аналіз та Оптимізація",
        kind="warn",
        value=mo.md(f"""
        **1. 🔶🔥🧠 Три фреймворки — кожен у своїй зоні.**

        <ul style="margin-top: 5px; margin-bottom: 10px; padding-left: 20px;">
            <li><b>TensorFlow</b> — <code>tf.data</code> (паралельне декодування, <code>ignore_errors</code> для битих файлів, prefetch) і EfficientNetV2 з коректним діапазоном пікселів 0..255;</li>
            <li>{_torch_item};</li>
            <li><b>Keras 3 (бекенд PyTorch)</b> — багатовхідна Late-Fusion мережа (гілка на модальність → конкатенація → MLP) із CosineDecay, AdamW, шумом/дропаутом, 5 фолдів × 3 seed.</li>
        </ul>

        **Graceful Degradation:** биті/відсутні фото пропускаються, тварини без фото отримують середній вектор, збій одного екстрактора не зупиняє пайплайн.

        **2. 🧠 Що змінено порівняно з baseline.** EfficientNet отримує пікселі 0..255 (rescaling усередині моделі); замість LSTM «з нуля» — попередньо навчені моделі та TF-IDF;
        використано ВСІ фото тварини та табличні ознаки; усі моделі валідуються OOF, а фінальний прогноз — усереднення 5 фолдів
        (еквівалент навчання на всіх даних). Приватний набір має класи {{1..4}} — пороги оптимізуються саме під них.

        **3. 🎯 QWK.** Регресія + `OptimizedRounder`; чесна оцінка QWK = **{final['qwk_honest'][0]:.3f}**.

        **4. 🚀 Подальші кроки.** Fine-tuning трансформера у PyTorch (якщо діагностика покаже, що текст — головний сигнал), pseudo-labeling на test,
        TTA для зображень, Optuna для гіперпараметрів стек-моделей.
        """),
    )
    mo.vstack([css_no_scroll, _conclusions_ui])
    return


if __name__ == "__main__":
    app.run()
