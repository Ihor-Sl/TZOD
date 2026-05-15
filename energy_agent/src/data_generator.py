"""
Генератор синтетичних даних енергоспоживання.

Створює реалістичний річний профіль для домогосподарства в Україні:
  - Базове споживання з добовою сезонністю (ранкові/вечірні піки)
  - Тижнева сезонність (вихідні vs будні)
  - Річна сезонність (зима/літо: опалення, кондиціонер)
  - Випадкові аномалії та шум
  - Тарифи: одно-, дво- та тризонний

Вихід: data/consumption.csv, data/tariffs.json
"""
from __future__ import annotations

import json
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def hourly_pattern(hour: int, is_weekend: bool) -> float:
    """Профіль навантаження протягом доби (множник до базового рівня)."""
    if is_weekend:
        # У вихідні люди прокидаються пізніше і пік триваліший увечері
        morning = 1.2 * math.exp(-((hour - 10) ** 2) / 8)
        evening = 1.8 * math.exp(-((hour - 20) ** 2) / 6)
        night = 0.35
    else:
        # Будні: різкі піки 7-9 ранку та 18-22 вечора
        morning = 1.5 * math.exp(-((hour - 8) ** 2) / 4)
        evening = 1.9 * math.exp(-((hour - 19) ** 2) / 6)
        night = 0.3
    return night + morning + evening


def seasonal_factor(date: datetime) -> float:
    """Сезонний множник: зима — опалення/освітлення, літо — кондиціонер."""
    day_of_year = date.timetuple().tm_yday
    # Зимовий пік ~січень, літній пік ~липень-серпень (менший)
    winter = 0.7 * math.cos(2 * math.pi * (day_of_year - 15) / 365)
    summer = 0.25 * math.cos(2 * math.pi * (day_of_year - 200) / 365)
    return 1.0 + winter + summer


def generate_consumption(start_date: str = "2024-01-01", days: int = 365) -> pd.DataFrame:
    """Згенерувати погодинний датасет споживання за `days` діб."""
    start = datetime.fromisoformat(start_date)
    rows = []
    base_load_kwh = 0.45  # середнє погодинне споживання, кВт·год

    for d in range(days):
        date = start + timedelta(days=d)
        is_weekend = date.weekday() >= 5
        season_mul = seasonal_factor(date)

        # Випадкова "аномалія" — наприклад, прихід гостей, ремонт, поломка
        anomaly_day = random.random() < 0.03
        anomaly_mul = random.uniform(1.4, 1.9) if anomaly_day else 1.0

        for h in range(24):
            timestamp = date + timedelta(hours=h)
            mul = hourly_pattern(h, is_weekend) * season_mul * anomaly_mul
            noise = np.random.normal(loc=1.0, scale=0.07)
            consumption = max(0.05, base_load_kwh * mul * noise)
            rows.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "consumption_kwh": round(consumption, 4),
                    "is_weekend": is_weekend,
                    "is_anomaly": anomaly_day,
                    "hour": h,
                    "day_of_week": date.strftime("%A"),
                }
            )

    df = pd.DataFrame(rows)
    return df


def generate_tariffs() -> dict:
    """Тарифні плани, чинні в Україні (приклад на 2024 рік)."""
    return {
        "single_zone": {
            "name": "Одна зона (стандартний)",
            "description": "Єдина ціна за весь час доби.",
            "price_uah_per_kwh": {"all": 4.32},
            "currency": "UAH",
        },
        "two_zone": {
            "name": "Дві зони (день/ніч)",
            "description": "Денний (07:00–23:00) і нічний (23:00–07:00) тарифи. Нічний дешевший на 50%.",
            "price_uah_per_kwh": {
                "day": 4.32,   # 07:00 – 23:00
                "night": 2.16, # 23:00 – 07:00
            },
            "zones": {
                "day": [(7, 23)],
                "night": [(0, 7), (23, 24)],
            },
            "currency": "UAH",
        },
        "three_zone": {
            "name": "Три зони (пік/напівпік/ніч)",
            "description": "Піковий (08:00–11:00 та 20:00–22:00), напівпіковий і нічний.",
            "price_uah_per_kwh": {
                "peak": 6.48,
                "half_peak": 4.32,
                "night": 1.73,
            },
            "zones": {
                "peak": [(8, 11), (20, 22)],
                "half_peak": [(7, 8), (11, 20), (22, 23)],
                "night": [(0, 7), (23, 24)],
            },
            "currency": "UAH",
        },
    }


def main() -> None:
    print("Генерація погодинних даних споживання за 2024 рік…")
    df = generate_consumption("2024-01-01", days=366)  # 2024 — високосний
    df.to_csv(DATA_DIR / "consumption.csv", index=False)
    print(f"  → {DATA_DIR / 'consumption.csv'}  ({len(df):,} рядків)")
    print(f"  → сумарне споживання за рік: {df['consumption_kwh'].sum():.1f} кВт·год")
    print(f"  → середнє погодинне: {df['consumption_kwh'].mean():.3f} кВт·год")

    tariffs = generate_tariffs()
    with open(DATA_DIR / "tariffs.json", "w", encoding="utf-8") as f:
        json.dump(tariffs, f, ensure_ascii=False, indent=2)
    print(f"  → {DATA_DIR / 'tariffs.json'}")


if __name__ == "__main__":
    main()
