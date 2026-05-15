# ЕнергоАгент — ШІ-агент чат-інтерфейсу до енергетичних даних

> Міні проект, варіант 19.
> Автор: Ігор, група ТВ-32, НТУУ «КПІ ім. Ігоря Сікорського».

## Про проєкт

**ЕнергоАгент** — це інтерактивний ШІ-агент, який дає природномовний (українською!)
чат-доступ до енергетичних даних домогосподарства. На відміну від звичайного
чат-бота, агент:

- **планує** виконання запиту (через function calling LLM);
- **обирає та викликає інструменти** (8 функцій доступу до даних);
- **має пам'ять** — короткострокову (історія) і довгострокову (профіль);
- **реалізує ReAct-цикл** — Reason → Act → Observe → Reason …;
- **самокоригується** — може зробити кілька викликів інструментів підряд.

## Швидкий старт

```bash
# 1. Клонування / розпакування проєкту
cd energy_agent

# 2. Встановити залежності
pip install -r requirements.txt

# 3. Згенерувати синтетичні дані (один раз)
python src/data_generator.py

# 4. Запустити веб-інтерфейс
cd src
streamlit run app.py
```

Після запуску відкриється браузер з адресою `http://localhost:8501`.

### Опційно: підключити справжній LLM

Без ключа агент працює в **DEMO-режимі** (rule-based intent detection — повноцінна
демонстрація UI, інструментів, пам'яті без інтернету).

Для повного LLM-режиму (OpenAI або сумісний):

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4o-mini"        # за замовчуванням

# Або безкоштовний Groq (потрібен ключ з groq.com)
export OPENAI_API_KEY="gsk_..."
export OPENAI_BASE_URL="https://api.groq.com/openai/v1"
export OPENAI_MODEL="llama-3.3-70b-versatile"

# Або локальний Ollama
export OPENAI_API_KEY="ollama"
export OPENAI_BASE_URL="http://localhost:11434/v1"
export OPENAI_MODEL="llama3.1"

streamlit run app.py
```

## Структура проєкту

```
energy_agent/
├── README.md
├── requirements.txt
├── data/
│   ├── consumption.csv       # 8 784 погодинних записи за 2024 рік
│   ├── tariffs.json          # 3 тарифні плани
│   └── user_profile.json     # довгострокова пам'ять (генерується)
├── src/
│   ├── data_generator.py     # синтез даних
│   ├── tools.py              # 8 інструментів агента
│   ├── agent.py              # ядро (ReAct-цикл + LLM)
│   ├── app.py                # Streamlit UI
│   ├── diagrams.py           # генератор діаграм для звіту
│   └── screenshots.py        # мокапи UI для звіту
├── images/                   # PNG-діаграми та скріншоти
├── charts/                   # графіки, згенеровані агентом (runtime)
└── docs/
    └── Звіт.docx             # академічний звіт українською
```

## Інструменти агента (8)

| Назва | Опис |
|---|---|
| `get_consumption_summary` | Сумарне/середнє/max/min споживання за період |
| `get_hourly_profile` | Середній добовий профіль (24 значення) |
| `detect_peak_hours` | Top-N годин з найвищим середнім споживанням |
| `forecast_next_days` | Прогноз на N діб (3 методи) |
| `calculate_cost` | Вартість за обраним тарифом |
| `compare_tariffs` | Порівняння 3 тарифів + рекомендація |
| `find_anomalies` | Аномальні дні за z-score |
| `generate_chart` | PNG-графік (daily / hourly_profile / monthly) |

## Приклади запитів

- *Скільки я спожив у січні?*
- *Який тариф мені вигідніший за весь рік?*
- *Покажи прогноз на наступний тиждень*
- *Які пікові години навантаження?*
- *Чи є аномалії в грудні?*
- *Покажи графік споживання за весь рік*
- *Скільки я заплачу за лютий за тарифом 'дві зони'?*

## Тестування інструментів окремо (CLI)

```bash
python src/tools.py     # smoke-тест інструментів
python src/agent.py     # smoke-тест агента
```

## Метрики якості (вибірка з звіту)

| Метрика | Значення |
|---|---|
| Latency (DEMO mode, один запит) | < 100 ms |
| Latency (LLM mode, gpt-4o-mini) | 1.5–3.5 s |
| Tool-call accuracy (10 типових запитів) | 100% |
| Coverage (типи запитів) | 8 інтентів |
| Розмір системного промпту | ~750 токенів |

## Технології

- Python 3.10+
- Streamlit (UI)
- OpenAI Python SDK (LLM API; сумісний з Groq, Ollama тощо)
- Pandas / NumPy / Matplotlib

## Ліцензія

Навчальний проект; код вільно поширюється за MIT.
