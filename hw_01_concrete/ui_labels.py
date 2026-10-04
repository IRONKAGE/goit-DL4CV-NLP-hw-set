# 1. Єдине джерело істини (Базові короткі назви колонок)
UA_COLUMNS = {
    "cement": "Цемент (кг/м³)",
    "slag": "Доменний шлак (кг/м³)",
    "ash": "Зола-винесення (кг/м³)",
    "water": "Вода (кг/м³)",
    "superplastic": "Суперпластифікатор (кг/м³)",
    "coarseagg": "Великий заповнювач (кг/м³)",
    "fineagg": "Дрібний заповнювач (кг/м³)",
    "age": "Вік (дні)",
    "strength": "Міцність (МПа)"
}

# 2. Мапа відповідності (Альтернативні довгі назви -> Базові назви)
ALT_MAP = {
    "CementComponent": "cement",
    "BlastFurnaceSlag": "slag",
    "FlyAshComponent": "ash",
    "WaterComponent": "water",
    "SuperplasticizerComponent": "superplastic",
    "CoarseAggregateComponent": "coarseagg",
    "FineAggregateComponent": "fineagg",
    "AgeInDays": "age",
    "Strength": "strength"
}

# 3. Динамічне злиття (Без дублювання тексту!)
for alt_col, base_col in ALT_MAP.items():
    if base_col in UA_COLUMNS:
        UA_COLUMNS[alt_col] = UA_COLUMNS[base_col]
