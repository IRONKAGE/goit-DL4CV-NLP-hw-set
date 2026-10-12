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
        📧 ДЗ №4: Фільтрація Спаму <i>(5 NLP Approaches Comparative Study)</i>
    </div>

    <h3 align="center"><b><u>Пайплайн</u>: Class Imbalance ➔ BoW / TF-IDF / GloVe / FastText / Transformers ➔ 🚀 Interactive Demo</b></h3>

    <p align="center"><i>© Oleh Hatsenko (IRONKAGE) | Deep Learning for Computer Vision and NLP</i></p>
    """)
    return (css_no_scroll,)


@app.cell(hide_code=True)
def imports():
    import time
    import logging

    import marimo as mo
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, precision_score,
        recall_score, f1_score, roc_curve
    )

    from core.hardware import get_hardware_config
    from hw_04_spam_nlp.data_adapter import get_spam_data, clean_text
    from hw_04_spam_nlp.nlp_core import Word2VecEmbedder, FastTextEmbedder, TransformerEmbedder
    from hw_04_spam_nlp.ui_labels import get_spam_label

    device, device_name = get_hardware_config()

    mo.center(mo.md(f"✅ **Екосистема NLP (Scikit-Learn, Gensim, Transformers) успішно ініціалізована! Пристрій: {device_name}**"))
    return (
        CountVectorizer,
        FastTextEmbedder,
        LogisticRegression,
        TfidfVectorizer,
        TransformerEmbedder,
        Word2VecEmbedder,
        accuracy_score,
        clean_text,
        device_name,
        f1_score,
        get_spam_data,
        get_spam_label,
        go,
        logging,
        mo,
        pd,
        precision_score,
        px,
        recall_score,
        roc_auc_score,
        roc_curve,
        time,
    )


@app.cell(hide_code=True)
def load_and_explore_data(
    css_no_scroll,
    get_spam_data,
    get_spam_label,
    mo,
    px,
):
    _header = mo.md("<h2 align='center' style='margin-bottom: 20px;'>1. Аналіз балансу класів та попередня обробка</h2>")

    with mo.status.spinner("Завантаження та очищення тексту..."):
        train_df, val_df = get_spam_data()

    # Аналіз співвідношення класів з локалізацією
    class_counts = train_df['target'].value_counts().reset_index()
    class_counts.columns = ['Клас', 'Кількість']
    class_counts['Категорія'] = class_counts['Клас'].map(get_spam_label)

    fig_dist = px.pie(
        class_counts, values='Кількість', names='Категорія', hole=0.45,
        color='Клас', color_discrete_map={0: '#2E86AB', 1: '#D64933'},
        title="<b>Співвідношення класів у навчальній вибірці (Train Set)</b>"
    )
    fig_dist.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#FFFFFF', width=2)))
    fig_dist.update_layout(height=400, title_x=0.5, legend_title_text="Тип повідомлення")

    display_dist = mo.as_html(fig_dist)

    _insight_ui = mo.callout(
        kind="info",
        title="💡 Tech Lead Insight (Аналіз даних):",
        value=mo.md("""
        - **Дисбаланс:** Легітимні повідомлення (Ham) становлять понад **86%** набору даних, тоді як спам — лише **~14%**.
        - **Стратегія оцінки:** Через таку асиметрію показник `Accuracy` не може бути єдиним орієнтиром. Ми розраховуємо **ROC-AUC, Precision, Recall та F1-score**, щоб контролювати як пропуск спаму, так і помилкове блокування важливих листів.
        """)
    )

    mo.vstack([
        css_no_scroll, _header, _insight_ui, mo.center(display_dist)
    ])
    return train_df, val_df


@app.cell(hide_code=True)
def train_and_evaluate_models(
    CountVectorizer,
    FastTextEmbedder,
    LogisticRegression,
    TfidfVectorizer,
    TransformerEmbedder,
    Word2VecEmbedder,
    accuracy_score,
    css_no_scroll,
    device_name,
    f1_score,
    go,
    logging,
    mo,
    pd,
    precision_score,
    px,
    recall_score,
    roc_auc_score,
    roc_curve,
    time,
    train_df,
    val_df,
):
    logger = logging.getLogger("hw_04.pipeline")

    _header = mo.md("<h2 align='center' style='margin-bottom: 20px;'>2. R&D Тренування: BoW / TF-IDF / GloVe / FastText / Transformers</h2>")
    mo.output.append(_header)

    X_train_txt = train_df['clean_text'].values
    y_train = train_df['target'].values
    X_val_txt = val_df['clean_text'].values
    y_val = val_df['target'].values

    results = []
    roc_curves = {}

    _start_time = time.perf_counter()

    with mo.status.progress_bar(title="Тренування ансамблю NLP-класифікаторів...", total=5, remove_on_exit=True) as _bar:

        # 1. Bag-of-Words
        msg = "Модель 1/5: Bag-of-Words (BoW) · Векторизація..."
        logger.info(msg)
        _bar.update(increment=0, subtitle=msg)

        bow_vec = CountVectorizer(max_features=3000, ngram_range=(1, 2))
        X_train_bow = bow_vec.fit_transform(X_train_txt)
        X_val_bow = bow_vec.transform(X_val_txt)

        clf_bow = LogisticRegression(max_iter=1000, random_state=42)
        clf_bow.fit(X_train_bow, y_train)
        pred_bow = clf_bow.predict(X_val_bow)
        prob_bow = clf_bow.predict_proba(X_val_bow)[:, 1]

        results.append({
            "Підхід": "Bag-of-Words (BoW)",
            "Accuracy": round(accuracy_score(y_val, pred_bow), 4),
            "AUC": round(roc_auc_score(y_val, prob_bow), 4),
            "Precision": round(precision_score(y_val, pred_bow), 4),
            "Recall": round(recall_score(y_val, pred_bow), 4),
            "F1-Score": round(f1_score(y_val, pred_bow), 4)
        })
        roc_curves["Bag-of-Words"] = roc_curve(y_val, prob_bow)

        msg_done = "✅ Модель 1/5: Bag-of-Words (BoW) · Готово!"
        logger.info(msg_done)
        _bar.update(increment=1, subtitle=msg_done)

        # 2. TF-IDF
        msg = "Модель 2/5: TF-IDF · Векторизація..."
        logger.info(msg)
        _bar.update(increment=0, subtitle=msg)

        tfidf_vec = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
        X_train_tfidf = tfidf_vec.fit_transform(X_train_txt)
        X_val_tfidf = tfidf_vec.transform(X_val_txt)

        clf_tfidf = LogisticRegression(max_iter=1000, random_state=42)
        clf_tfidf.fit(X_train_tfidf, y_train)
        pred_tfidf = clf_tfidf.predict(X_val_tfidf)
        prob_tfidf = clf_tfidf.predict_proba(X_val_tfidf)[:, 1]

        results.append({
            "Підхід": "TF-IDF",
            "Accuracy": round(accuracy_score(y_val, pred_tfidf), 4),
            "AUC": round(roc_auc_score(y_val, prob_tfidf), 4),
            "Precision": round(precision_score(y_val, pred_tfidf), 4),
            "Recall": round(recall_score(y_val, pred_tfidf), 4),
            "F1-Score": round(f1_score(y_val, pred_tfidf), 4)
        })
        roc_curves["TF-IDF"] = roc_curve(y_val, prob_tfidf)

        msg_done = "✅ Модель 2/5: TF-IDF · Готово!"
        logger.info(msg_done)
        _bar.update(increment=1, subtitle=msg_done)

        # 3. Word2Vec (GloVe)
        msg = "Модель 3/5: GloVe Embeddings · Mean Pooling..."
        logger.info(msg)
        _bar.update(increment=0, subtitle=msg)

        w2v_emb = Word2VecEmbedder(model_name="glove-twitter-50")
        X_tr_w2v = w2v_emb.transform(X_train_txt, desc="Train: GloVe Pooling")
        X_v_w2v = w2v_emb.transform(X_val_txt, desc="Val: GloVe Pooling")

        clf_w2v = LogisticRegression(max_iter=1000, random_state=42)
        clf_w2v.fit(X_tr_w2v, y_train)
        prob_w2v = clf_w2v.predict_proba(X_v_w2v)[:, 1]

        results.append({
            "Підхід": "GloVe (50d)",
            "Accuracy": round(accuracy_score(y_val, clf_w2v.predict(X_v_w2v)), 4),
            "AUC": round(roc_auc_score(y_val, prob_w2v), 4),
            "Precision": round(precision_score(y_val, clf_w2v.predict(X_v_w2v)), 4),
            "Recall": round(recall_score(y_val, clf_w2v.predict(X_v_w2v)), 4),
            "F1-Score": round(f1_score(y_val, clf_w2v.predict(X_v_w2v)), 4)
        })
        roc_curves["GloVe"] = roc_curve(y_val, prob_w2v)

        msg_done = "✅ Модель 3/5: GloVe Embeddings · Готово!"
        logger.info(msg_done)
        _bar.update(increment=1, subtitle=msg_done)

        # 4. FastText
        msg = "Модель 4/5: FastText · Subword N-grams..."
        logger.info(msg)
        _bar.update(increment=0, subtitle=msg)

        ft_emb = FastTextEmbedder(model_name="fasttext-wiki-news-subwords-300")
        X_tr_ft = ft_emb.transform(X_train_txt, desc="Train: FastText Pooling")
        X_v_ft = ft_emb.transform(X_val_txt, desc="Val: FastText Pooling")

        clf_ft = LogisticRegression(max_iter=1000, random_state=42)
        clf_ft.fit(X_tr_ft, y_train)
        prob_ft = clf_ft.predict_proba(X_v_ft)[:, 1]

        results.append({
            "Підхід": "FastText (300d)",
            "Accuracy": round(accuracy_score(y_val, clf_ft.predict(X_v_ft)), 4),
            "AUC": round(roc_auc_score(y_val, prob_ft), 4),
            "Precision": round(precision_score(y_val, clf_ft.predict(X_v_ft)), 4),
            "Recall": round(recall_score(y_val, clf_ft.predict(X_v_ft)), 4),
            "F1-Score": round(f1_score(y_val, clf_ft.predict(X_v_ft)), 4)
        })
        roc_curves["FastText"] = roc_curve(y_val, prob_ft)

        msg_done = "✅ Модель 4/5: FastText · Готово!"
        logger.info(msg_done)
        _bar.update(increment=1, subtitle=msg_done)

        # 5. Transformers (MiniLM)
        msg = f"Модель 5/5: Transformers (MiniLM) · {device_name.upper()} Acceleration..."
        logger.info(msg)
        _bar.update(increment=0, subtitle=msg)

        tr_emb = TransformerEmbedder(device="cpu")
        X_tr_tr = tr_emb.transform(X_train_txt, desc="Train: MiniLM Attention")
        X_v_tr = tr_emb.transform(X_val_txt, desc="Val: MiniLM Attention")

        clf_tr = LogisticRegression(max_iter=1000, random_state=42)
        clf_tr.fit(X_tr_tr, y_train)
        prob_tr = clf_tr.predict_proba(X_v_tr)[:, 1]

        results.append({
            "Підхід": "Transformers (MiniLM)",
            "Accuracy": round(accuracy_score(y_val, clf_tr.predict(X_v_tr)), 4),
            "AUC": round(roc_auc_score(y_val, prob_tr), 4),
            "Precision": round(precision_score(y_val, clf_tr.predict(X_v_tr)), 4),
            "Recall": round(recall_score(y_val, clf_tr.predict(X_v_tr)), 4),
            "F1-Score": round(f1_score(y_val, clf_tr.predict(X_v_tr)), 4)
        })
        roc_curves["Transformers"] = roc_curve(y_val, prob_tr)

        msg_done = "✅ Модель 5/5: Transformers (MiniLM) · Готово!"
        logger.info(msg_done)
        _bar.update(increment=1, subtitle=msg_done)

    _elapsed = time.perf_counter() - _start_time
    _mins, _secs = divmod(_elapsed, 60)
    _time_str = f"{int(_mins)} хв {int(_secs)} сек" if _mins > 0 else f"{_elapsed:.1f} сек"
    logger.info(f"✅ Всі 5 моделей успішно натреновано за {_time_str}!")

    res_df = pd.DataFrame(results)

    # Побудова графіка порівняння метрик (Grouped Bar Chart)
    metrics_melted = res_df.melt(id_vars=["Підхід"], var_name="Метрика", value_name="Значення")
    fig_metrics = px.bar(
        metrics_melted, x="Метрика", y="Значення", color="Підхід", barmode="group",
        color_discrete_sequence=px.colors.qualitative.Prism,
        title="<b>Комплексне порівняння метрик 5 NLP-підходів</b>",
        labels={"Підхід": "Модель"}
    )
    fig_metrics.update_layout(yaxis_range=[0.7, 1.0], height=450, title_x=0.5, legend_title_text="Модель")

    # Візуалізація ROC-кривих
    fig_roc = go.Figure()
    _colors = px.colors.qualitative.Prism

    for i, (name, (fpr, tpr, _)) in enumerate(roc_curves.items()):
        auc_val = res_df.loc[res_df["Підхід"].str.contains(name, regex=False), "AUC"].values[0]
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode='lines',
            name=f"{name} (AUC = {auc_val:.4f})",
            line=dict(width=2.5, color=_colors[i])
        ))

    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Випадкове вгадування', line=dict(dash='dash', color='gray')))
    fig_roc.update_layout(
        title="<b>ROC-криві класифікаторів</b>",
        xaxis_title="False Positive Rate (Хибнопозитивні)", yaxis_title="True Positive Rate (Істинно позитивні)",
        height=450, title_x=0.5, legend_title_text="Модель"
    )

    _footer = mo.md(f"✅ **Тренування ансамблю моделей завершено за {_time_str}!**")

    # Зберігаємо всі навчені моделі та векторизатори у словники для Demo
    models_dict = {
        "BoW": clf_bow,
        "TF-IDF": clf_tfidf,
        "Word2Vec": clf_w2v,
        "FastText": clf_ft,
        "Transformers": clf_tr
    }
    vectorizers_dict = {
        "BoW": bow_vec,
        "TF-IDF": tfidf_vec,
        "Word2Vec": w2v_emb,
        "FastText": ft_emb,
        "Transformers": tr_emb
    }

    _display_ui = mo.vstack([
        css_no_scroll,
        mo.ui.table(results, selection=None),
        mo.center(mo.as_html(fig_metrics)),
        mo.center(mo.as_html(fig_roc)),
        mo.center(_footer)
    ])
    mo.output.append(_display_ui)
    return models_dict, results, vectorizers_dict


@app.cell(hide_code=True)
def interactive_demo_input(
    clean_text,
    css_no_scroll,
    mo,
    models_dict,
    vectorizers_dict,
):
    _header = mo.md("<h2 align='center' style='margin-bottom: 20px;'>🚀 3. Інтерактивний Інференс (Demo)</h2>")

    _demo_text = mo.md(
        "Вставте будь-який текст листа (англійською) у поле нижче та оберіть модель для перевірки на спам."
    )

    # Випадаючий список для вибору моделі
    model_selector = mo.ui.dropdown(
        options={
            "TF-IDF (Рекомендована)": "TF-IDF",
            "Bag-of-Words": "BoW",
            "Word2Vec (GloVe)": "Word2Vec",
            "FastText": "FastText",
            "Transformers (MiniLM)": "Transformers",
            "🔥 Порівняти всі 5 моделей одночасно": "ALL"
        },
        value="TF-IDF (Рекомендована)",
        label="Оберіть NLP-модель:"
    )

    # Текстове поле для введення
    user_input = mo.ui.text_area(
        placeholder="Введіть текст повідомлення (напр., 'URGENT! You have won a 1 week FREE membership in our $100,000 Prize Jackpot!')...",
        label="Текст листа:",
        full_width=True,
        rows=4
    )

    def predict_single_model(text: str, model_name: str) -> tuple:
        """Повертає кортеж (is_spam_boolean, probability_float) для конкретної моделі."""
        cleaned = clean_text(text)

        # Векторизація
        vec_text = vectorizers_dict[model_name].transform([cleaned])

        # Інференс
        prob = models_dict[model_name].predict_proba(vec_text)[0][1]
        return prob > 0.5, prob

    def predict_spam(text: str, selected_mode: str) -> str:
        if not text.strip():
            return "Будь ласка, введіть текст для аналізу."

        # Якщо обрано одну модель
        if selected_mode != "ALL":
            is_spam, prob = predict_single_model(text, selected_mode)
            color = "#D64933" if is_spam else "#2A9D8F"
            result_text = "🚨 **УВАГА: Це СПАМ!**" if is_spam else "✅ **Легітимний лист (Ham)**"

            return f"""
            <div style="padding: 10px; border-left: 5px solid {color}; background-color: #f8f9fa; margin-bottom: 5px;">
                <span style="font-size: 1.1em;">{result_text}</span> <br/>
                <span style="color: #666; font-size: 0.9em;">Ймовірність спаму: {prob*100:.2f}%</span>
            </div>
            """

        # Якщо обрано порівняння всіх 5 моделей
        html_output = "<div>"
        for name in ["TF-IDF", "BoW", "Word2Vec", "FastText", "Transformers"]:
            is_spam, prob = predict_single_model(text, name)
            color = "#D64933" if is_spam else "#2A9D8F"
            result_text = "🚨 **УВАГА: Це СПАМ!**" if is_spam else "✅ **Легітимний лист (Ham)**"

            html_output += f"""
            <div style="padding: 8px; border-left: 4px solid {color}; background-color: #f8f9fa; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;">
                <b>{name}:</b>
                <span>{result_text} (Ймовірність: {prob*100:.1f}%)</span>
            </div>
            """
        html_output += "</div>"
        return html_output

    _display_ui = mo.vstack([
        css_no_scroll, _header, _demo_text,
        mo.hstack([model_selector]),
        user_input
    ])
    mo.output.append(_display_ui)
    return model_selector, predict_spam, user_input


@app.cell(hide_code=True)
def interactive_demo_output(mo, model_selector, predict_spam, user_input):
    # Реактивне зчитування (доступ до value в іншій клітинці)
    _prediction_output = mo.md(f"""
    <div style="margin-top: 15px; padding: 15px; border-radius: 8px; border: 1px solid #ddd; background-color: #ffffff; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
        <h4 style="margin-top:0; color: #2E86AB; text-transform: uppercase;">Вердикт Моделі:</h4>
        {predict_spam(user_input.value, model_selector.value)}
    </div>
    """)

    mo.output.append(_prediction_output)
    return


@app.cell(hide_code=True)
def final_conclusions(css_no_scroll, mo, results):
    # Розпаковуємо результати для 5-ти моделей
    bow_res = results[0]
    tfidf_res = results[1]
    glove_res = results[2]
    ft_res = results[3]
    tr_res = results[4]

    _conclusions_ui = mo.callout(
        title="4. Аналіз та Оптимізація (Tech Lead Insight)",
        kind="warn",
        value=mo.md(f"""
        **1. 🎯 Аналіз метрик.**
        У ході роботи було досліджено еволюцію 5-ти підходів до представлення тексту для задачі класифікації спаму:
        - **TF-IDF**: Accuracy = {tfidf_res['Accuracy']*100:.2f}%, AUC = {tfidf_res['AUC']:.4f}, F1 = {tfidf_res['F1-Score']:.4f}
        - **Bag-of-Words**: Accuracy = {bow_res['Accuracy']*100:.2f}%, AUC = {bow_res['AUC']:.4f}, F1 = {bow_res['F1-Score']:.4f}
        - **FastText (300d)**: Accuracy = {ft_res['Accuracy']*100:.2f}%, AUC = {ft_res['AUC']:.4f}, F1 = {ft_res['F1-Score']:.4f}
        - **Word2Vec (GloVe 50d)**: Accuracy = {glove_res['Accuracy']*100:.2f}%, AUC = {glove_res['AUC']:.4f}, F1 = {glove_res['F1-Score']:.4f}
        - **Transformers (MiniLM)**: Accuracy = {tr_res['Accuracy']*100:.2f}%, AUC = {tr_res['AUC']:.4f}, F1 = {tr_res['F1-Score']:.4f}

        **2. 🔬 Інженерне обґрунтування вибору моделей.**
        - **Bag-of-Words (BoW)**: Використано як абсолютний базовий бенчмарк, що фіксує сиру частоту слів без семантичної ваги.
        - **TF-IDF**: Золотий стандарт для спаму. Математично штрафує загальновживані слова і надає максимальної ваги рідкісним, але критичним спам-тригерам (напр., *"free"*, *"urgent"*).
        - **Word2Vec (`glove-twitter-50`)**: Було свідомо обрано Твіттер-версію, а не класичну академічну (`wiki-gigaword`). Спам-повідомлення рідко пишуться вивіреною англійською з Вікіпедії. Спамери навмисно використовують скорочення, розмовний стиль ("u", "r", "gr8") або спотворюють слова, щоб обійти жорсткі фільтри. Ембедінги соцмереж ідеально "розуміють" цей живий та брудний інтернет-стиль.
        - **FastText (`subwords-300`)**: Розв'язує проблему незнайомих слів (OOV) завдяки розбиттю на n-грами символів. Це критично для перехоплення навмисно спотворених слів (напр., *"v1agra"*, *"f.r.e.e"*, *"w0n"*), оскільки алгоритм вловлює їхній корінь.
        - **Transformers (`all-MiniLM-L6-v2`)**: Сучасний SOTA-стандарт. Завдяки механізму Attention він відмовляється від статичних векторів слів і оцінює вектор *цілого речення*, дозволяючи розпізнавати прихований контекст маніпуляції чи фішингу, навіть якщо прямі тригер-слова відсутні.

        **3. 🧠 Еволюція підходів та їхня специфіка.**
        Попри наявність потужних семантичних моделей (Transformers), класичний **TF-IDF** залишається надзвичайно сильним бенчмарком у цій задачі. Зважування TF-IDF ідеально підсилює вагу спам-маркерів. Використання ж **Word2Vec/FastText** з операцією усереднення векторів (Mean Pooling) навпаки — «розмиває» сигнал окремих тригерних слів у загальному семантичному змісті легітимного листа.

        **4. 🚀 Подальші кроки (Оптимізація).**
        Для повноцінного розкриття потенціалу *Word2Vec/FastText* замість примітивного Mean Pooling необхідно використовувати рекурентні нейромережі (**BiLSTM / GRU**), які обробляють вектори як послідовність, зберігаючи контекст. А для **Transformers** максимальний результат досягається через повноцінний *Fine-Tuning* усієї архітектури (разом із Attention-шарами) суто під задачу бінарної класифікації спаму.
        """)
    )

    mo.vstack([
        css_no_scroll,
        _conclusions_ui
    ])
    return


if __name__ == "__main__":
    app.run()
