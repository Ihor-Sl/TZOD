"""
Інструменти ШІ-агента для роботи з енергетичними даними.

Кожна функція — це окремий "tool", який LLM викликає через function calling.
Усі інструменти повертають JSON-серіалізовані рядки (стандарт для OpenAI tools API).

Інструменти:
  - get_consumption_summary  — статистика за період
  - get_hourly_profile       — середній добовий профіль
  - detect_peak_hours        — виявлення пікових годин
  - forecast_next_days       — прогноз на N діб (наївний + ковзне середнє)
  - calculate_cost           — обчислення вартості за тарифом
  - compare_tariffs          — порівняння тарифних планів
  - find_anomalies           — пошук аномальних днів
  - generate_chart           — створення графіка (повертає шлях до PNG)
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHART_DIR = Path(__file__).resolve().parent.parent / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)


# ---------- Завантаження даних (кешується) ---------------------------------
_df_cache: Optional[pd.DataFrame] = None
_tariffs_cache: Optional[dict] = None


def _load_data() -> pd.DataFrame:
    global _df_cache
    if _df_cache is None:
        df = pd.read_csv(DATA_DIR / "consumption.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["timestamp"].dt.date
        df["month"] = df["timestamp"].dt.month
        _df_cache = df
    return _df_cache


def _load_tariffs() -> dict:
    global _tariffs_cache
    if _tariffs_cache is None:
        with open(DATA_DIR / "tariffs.json", encoding="utf-8") as f:
            _tariffs_cache = json.load(f)
    return _tariffs_cache


def _parse_period(start: Optional[str], end: Optional[str]) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Розпарсити рядки дат; за замовчуванням — весь датасет."""
    df = _load_data()
    if not start:
        start_ts = df["timestamp"].min()
    else:
        start_ts = pd.to_datetime(start)
    if not end:
        end_ts = df["timestamp"].max()
    else:
        end_ts = pd.to_datetime(end)
        # Якщо дано тільки дату — включаємо весь цей день
        if end_ts.hour == 0 and end_ts.minute == 0:
            end_ts = end_ts + pd.Timedelta(hours=23, minutes=59)
    return start_ts, end_ts


def _slice(start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    df = _load_data()
    s, e = _parse_period(start, end)
    return df[(df["timestamp"] >= s) & (df["timestamp"] <= e)].copy()


# ---------- TOOL 1. Зведена статистика -------------------------------------
def get_consumption_summary(start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> str:
    """Сумарна / середня / макс. / мін. статистика споживання за період."""
    sub = _slice(start_date, end_date)
    if sub.empty:
        return json.dumps({"error": "Немає даних за вказаний період."}, ensure_ascii=False)

    by_day = sub.groupby("date")["consumption_kwh"].sum()
    result = {
        "period": {"from": str(sub["timestamp"].min()), "to": str(sub["timestamp"].max())},
        "total_kwh": round(sub["consumption_kwh"].sum(), 2),
        "average_daily_kwh": round(by_day.mean(), 3),
        "max_daily_kwh": round(by_day.max(), 3),
        "max_daily_date": str(by_day.idxmax()),
        "min_daily_kwh": round(by_day.min(), 3),
        "min_daily_date": str(by_day.idxmin()),
        "days_count": int(by_day.size),
    }
    return json.dumps(result, ensure_ascii=False)


# ---------- TOOL 2. Добовий профіль ----------------------------------------
def get_hourly_profile(start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> str:
    """Середнє споживання за кожну годину доби."""
    sub = _slice(start_date, end_date)
    if sub.empty:
        return json.dumps({"error": "Немає даних."}, ensure_ascii=False)

    profile = sub.groupby("hour")["consumption_kwh"].mean().round(4).to_dict()
    return json.dumps({"hourly_average_kwh": profile}, ensure_ascii=False)


# ---------- TOOL 3. Пікові години ------------------------------------------
def detect_peak_hours(start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      top_n: int = 3) -> str:
    """N годин з найбільшим середнім споживанням."""
    sub = _slice(start_date, end_date)
    if sub.empty:
        return json.dumps({"error": "Немає даних."}, ensure_ascii=False)

    profile = sub.groupby("hour")["consumption_kwh"].mean().sort_values(ascending=False)
    peaks = [
        {"hour": int(h), "avg_kwh": round(float(v), 4)}
        for h, v in profile.head(top_n).items()
    ]
    return json.dumps({"peak_hours": peaks}, ensure_ascii=False)


# ---------- TOOL 4. Прогноз ------------------------------------------------
def forecast_next_days(days: int = 7,
                       method: str = "moving_average") -> str:
    """Прогноз сумарного добового споживання на N діб.

    method ∈ {"naive", "moving_average", "weekly_pattern"}
    """
    df = _load_data()
    last_date = df["timestamp"].max().normalize()
    by_day = df.groupby("date")["consumption_kwh"].sum()
    by_day.index = pd.to_datetime(by_day.index)

    history_window = by_day.tail(14)
    forecast = []
    if method == "naive":
        value = float(history_window.iloc[-1])
        for i in range(1, days + 1):
            forecast.append({"date": str((last_date + timedelta(days=i)).date()),
                             "forecast_kwh": round(value, 3)})
    elif method == "weekly_pattern":
        # повторюємо тижневий патерн з останніх 7 днів
        last7 = list(by_day.tail(7).values)
        for i in range(1, days + 1):
            value = float(last7[(i - 1) % 7])
            forecast.append({"date": str((last_date + timedelta(days=i)).date()),
                             "forecast_kwh": round(value, 3)})
    else:  # moving_average
        ma = float(history_window.mean())
        # додаємо невеликий шум, щоб не була пласка лінія
        for i in range(1, days + 1):
            value = ma * np.random.uniform(0.93, 1.07)
            forecast.append({"date": str((last_date + timedelta(days=i)).date()),
                             "forecast_kwh": round(value, 3)})

    return json.dumps({
        "method": method,
        "based_on_period": f"останні {len(history_window)} днів",
        "history_avg_kwh": round(float(history_window.mean()), 3),
        "forecast": forecast,
    }, ensure_ascii=False)


# ---------- TOOL 5. Розрахунок вартості ------------------------------------
def _hour_zone(hour: int, zones: dict) -> Optional[str]:
    for zone_name, ranges in zones.items():
        for lo, hi in ranges:
            if lo <= hour < hi:
                return zone_name
    return None


def calculate_cost(tariff_id: str = "single_zone",
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None) -> str:
    """Обчислити вартість спожитої е/е за вибраним тарифом."""
    tariffs = _load_tariffs()
    if tariff_id not in tariffs:
        return json.dumps({"error": f"Невідомий тариф '{tariff_id}'. Доступні: {list(tariffs)}"},
                          ensure_ascii=False)

    sub = _slice(start_date, end_date)
    if sub.empty:
        return json.dumps({"error": "Немає даних."}, ensure_ascii=False)

    tariff = tariffs[tariff_id]
    if tariff_id == "single_zone":
        total_kwh = sub["consumption_kwh"].sum()
        price = tariff["price_uah_per_kwh"]["all"]
        total_uah = total_kwh * price
        breakdown = {"all": {"kwh": round(float(total_kwh), 2), "uah": round(total_uah, 2)}}
    else:
        zones = tariff["zones"]
        sub = sub.copy()
        sub["zone"] = sub["hour"].apply(lambda h: _hour_zone(int(h), zones))
        by_zone = sub.groupby("zone")["consumption_kwh"].sum()
        breakdown = {}
        total_uah = 0.0
        for zone, kwh in by_zone.items():
            price = tariff["price_uah_per_kwh"][zone]
            uah = float(kwh) * price
            total_uah += uah
            breakdown[zone] = {"kwh": round(float(kwh), 2),
                               "price_uah_per_kwh": price,
                               "uah": round(uah, 2)}

    return json.dumps({
        "tariff": tariff["name"],
        "total_kwh": round(float(sub["consumption_kwh"].sum()), 2),
        "total_uah": round(total_uah, 2),
        "breakdown": breakdown,
    }, ensure_ascii=False)


# ---------- TOOL 6. Порівняння тарифів -------------------------------------
def compare_tariffs(start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> str:
    """Порахувати вартість за всіма тарифами і визначити найвигідніший."""
    tariffs = _load_tariffs()
    results = []
    for tariff_id in tariffs:
        raw = json.loads(calculate_cost(tariff_id, start_date, end_date))
        if "error" in raw:
            continue
        results.append({
            "tariff_id": tariff_id,
            "name": raw["tariff"],
            "total_uah": raw["total_uah"],
            "total_kwh": raw["total_kwh"],
        })
    if not results:
        return json.dumps({"error": "Не вдалося порівняти."}, ensure_ascii=False)

    best = min(results, key=lambda r: r["total_uah"])
    worst = max(results, key=lambda r: r["total_uah"])
    return json.dumps({
        "comparison": results,
        "best_tariff": best["tariff_id"],
        "best_uah": best["total_uah"],
        "savings_vs_worst_uah": round(worst["total_uah"] - best["total_uah"], 2),
    }, ensure_ascii=False)


# ---------- TOOL 7. Аномалії -----------------------------------------------
def find_anomalies(start_date: Optional[str] = None,
                   end_date: Optional[str] = None,
                   z_threshold: float = 2.0) -> str:
    """Знайти дні з аномально високим/низьким споживанням (z-score)."""
    sub = _slice(start_date, end_date)
    if sub.empty:
        return json.dumps({"error": "Немає даних."}, ensure_ascii=False)

    by_day = sub.groupby("date")["consumption_kwh"].sum()
    mean = by_day.mean()
    std = by_day.std()
    if std == 0:
        return json.dumps({"anomalies": []}, ensure_ascii=False)

    z = (by_day - mean) / std
    anomalies = [
        {"date": str(d), "kwh": round(float(by_day[d]), 3), "z_score": round(float(z[d]), 2)}
        for d in by_day.index
        if abs(z[d]) >= z_threshold
    ]
    return json.dumps({
        "threshold_z": z_threshold,
        "baseline_mean_kwh": round(float(mean), 3),
        "baseline_std_kwh": round(float(std), 3),
        "anomalies_count": len(anomalies),
        "anomalies": anomalies[:30],  # обмеження для контексту LLM
    }, ensure_ascii=False)


# ---------- TOOL 8. Графік -------------------------------------------------
def generate_chart(chart_type: str,
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None) -> str:
    """Згенерувати графік. chart_type ∈ {"daily","hourly_profile","monthly"}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    sub = _slice(start_date, end_date)
    if sub.empty:
        return json.dumps({"error": "Немає даних для побудови графіка."}, ensure_ascii=False)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    if chart_type == "daily":
        by_day = sub.groupby("date")["consumption_kwh"].sum()
        ax.plot(pd.to_datetime(by_day.index), by_day.values, color="#2E75B6", linewidth=1.4)
        ax.set_title("Добове споживання, кВт·год")
        ax.set_xlabel("Дата")
        ax.set_ylabel("кВт·год")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
        fig.autofmt_xdate()
    elif chart_type == "hourly_profile":
        prof = sub.groupby("hour")["consumption_kwh"].mean()
        ax.bar(prof.index, prof.values, color="#70AD47", alpha=0.85)
        ax.set_title("Середній добовий профіль, кВт·год")
        ax.set_xlabel("Година доби")
        ax.set_ylabel("кВт·год")
        ax.set_xticks(range(0, 24))
    elif chart_type == "monthly":
        by_month = sub.groupby(sub["timestamp"].dt.to_period("M"))["consumption_kwh"].sum()
        labels = [str(p) for p in by_month.index]
        ax.bar(labels, by_month.values, color="#ED7D31", alpha=0.9)
        ax.set_title("Місячне споживання, кВт·год")
        ax.set_xlabel("Місяць")
        ax.set_ylabel("кВт·год")
        plt.xticks(rotation=45)
    else:
        plt.close(fig)
        return json.dumps({"error": f"Невідомий chart_type '{chart_type}'."}, ensure_ascii=False)

    ax.grid(alpha=0.3)
    fig.tight_layout()

    fname = f"chart_{chart_type}_{datetime.now().strftime('%H%M%S%f')}.png"
    fpath = CHART_DIR / fname
    fig.savefig(fpath, dpi=110)
    plt.close(fig)

    return json.dumps({"chart_path": str(fpath),
                       "chart_type": chart_type,
                       "data_points": len(sub)},
                      ensure_ascii=False)


# ---------- Список інструментів у форматі OpenAI ---------------------------
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_consumption_summary",
            "description": "Повертає сумарне, середнє, максимальне і мінімальне споживання електроенергії за вказаний період. Використовуй, коли користувач питає 'скільки я спожив', 'яке загальне споживання' тощо.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string", "description": "Початок періоду у форматі YYYY-MM-DD. Якщо не вказано — увесь датасет."},
                    "end_date": {"type": "string", "description": "Кінець періоду у форматі YYYY-MM-DD."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_hourly_profile",
            "description": "Середнє споживання за кожну годину доби (24 значення). Корисно для розуміння режиму використання.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_peak_hours",
            "description": "Знаходить top_n годин доби з найвищим середнім споживанням.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                    "top_n": {"type": "integer", "default": 3},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forecast_next_days",
            "description": "Прогнозує сумарне добове споживання на наступні N діб. Метод: 'moving_average' (рекомендується), 'naive', 'weekly_pattern'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "default": 7},
                    "method": {"type": "string", "enum": ["naive", "moving_average", "weekly_pattern"], "default": "moving_average"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_cost",
            "description": "Обчислює вартість спожитої е/е за обраним тарифом. tariff_id: 'single_zone', 'two_zone', 'three_zone'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tariff_id": {"type": "string", "enum": ["single_zone", "two_zone", "three_zone"], "default": "single_zone"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_tariffs",
            "description": "Порівнює вартість за всіма трьома тарифними планами і визначає найвигідніший.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_anomalies",
            "description": "Знаходить дні з аномально високим або низьким споживанням (за z-score).",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                    "z_threshold": {"type": "number", "default": 2.0},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_chart",
            "description": "Створює PNG-графік. chart_type: 'daily' (добове за період), 'hourly_profile' (середній добовий профіль), 'monthly' (по місяцях).",
            "parameters": {
                "type": "object",
                "properties": {
                    "chart_type": {"type": "string", "enum": ["daily", "hourly_profile", "monthly"]},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                },
                "required": ["chart_type"],
            },
        },
    },
]


# Маппінг імен → функцій (для виконання після виклику LLM)
TOOL_FUNCTIONS = {
    "get_consumption_summary": get_consumption_summary,
    "get_hourly_profile": get_hourly_profile,
    "detect_peak_hours": detect_peak_hours,
    "forecast_next_days": forecast_next_days,
    "calculate_cost": calculate_cost,
    "compare_tariffs": compare_tariffs,
    "find_anomalies": find_anomalies,
    "generate_chart": generate_chart,
}


if __name__ == "__main__":
    # Маленький smoke-тест
    print("=== get_consumption_summary (січень 2024) ===")
    print(get_consumption_summary("2024-01-01", "2024-01-31"))
    print("\n=== detect_peak_hours ===")
    print(detect_peak_hours())
    print("\n=== compare_tariffs (січень) ===")
    print(compare_tariffs("2024-01-01", "2024-01-31"))
    print("\n=== forecast_next_days (3) ===")
    print(forecast_next_days(days=3))
