"""Форматування тривалості для футерів ноутбука: 83.4 → '1 хв 23 с'."""


def fmt_elapsed(seconds: float) -> str:
    s = int(round(max(float(seconds), 0.0)))
    if s < 60:
        return f"{s} с"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m} хв {s:02d} с"
    h, m = divmod(m, 60)
    return f"{h} год {m:02d} хв"
