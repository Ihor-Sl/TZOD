"""Створює архітектурні діаграми для звіту (PNG)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

IMG_DIR = Path(__file__).resolve().parent.parent / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
#   Діаграма 1. Загальна архітектура агента
# ============================================================
def draw_architecture():
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 70)
    ax.axis("off")

    def box(x, y, w, h, text, color="#D5E8F0", edge="#2E75B6", fs=10, weight="normal"):
        bb = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5",
                            linewidth=1.8, facecolor=color, edgecolor=edge)
        ax.add_patch(bb)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, weight=weight, wrap=True)

    def arrow(x1, y1, x2, y2, color="#555", style="->"):
        a = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle=style, mutation_scale=15,
                            linewidth=1.5, color=color)
        ax.add_patch(a)

    # User
    box(2, 56, 16, 9, "Користувач", color="#FFF2CC", edge="#BF8F00", fs=11, weight="bold")

    # UI
    box(24, 56, 22, 9, "Streamlit UI\n(чат-інтерфейс)", color="#E1D5E7", edge="#9673A6", fs=10, weight="bold")

    # Agent core
    box(52, 56, 30, 9, "EnergyAgent\n(ReAct-цикл)", color="#D5E8F0", edge="#2E75B6", fs=11, weight="bold")

    # LLM
    box(86, 56, 12, 9, "LLM API\n(OpenAI/\nGroq)", color="#F8CECC", edge="#B85450", fs=9, weight="bold")

    # Memory
    box(52, 38, 13, 10, "Пам'ять\n(коротко-\nстрокова +\nдовгострокова)", color="#FFF2CC", edge="#BF8F00", fs=9)

    # Planner
    box(68, 38, 14, 10, "Планувальник\n(через function-\ncalling LLM)", color="#D5E8F0", edge="#2E75B6", fs=9)

    # Tools row
    tools = [
        ("get_consumption\n_summary", "#D5E8D4"),
        ("get_hourly\n_profile", "#D5E8D4"),
        ("detect_peak\n_hours", "#D5E8D4"),
        ("forecast_next\n_days", "#D5E8D4"),
        ("calculate_cost", "#D5E8D4"),
        ("compare_tariffs", "#D5E8D4"),
        ("find_anomalies", "#D5E8D4"),
        ("generate_chart", "#D5E8D4"),
    ]
    for i, (name, color) in enumerate(tools):
        x = 2 + i * 12
        box(x, 20, 11, 8, name, color=color, edge="#70AD47", fs=7.5)

    # Data
    box(15, 4, 30, 9, "consumption.csv\n(8 784 погодинних записи, 2024)",
        color="#FFE4B5", edge="#D79B00", fs=9, weight="bold")
    box(55, 4, 25, 9, "tariffs.json\n(3 тарифні плани)",
        color="#FFE4B5", edge="#D79B00", fs=9, weight="bold")

    # Section labels
    ax.text(50, 31, "Інструменти (Tools)", ha="center", fontsize=12, weight="bold", color="#555")
    ax.text(50, 16, "Джерела даних", ha="center", fontsize=12, weight="bold", color="#555")

    # Arrows
    arrow(18, 60.5, 24, 60.5)
    arrow(46, 60.5, 52, 60.5)
    arrow(82, 60.5, 86, 60.5)
    arrow(86, 58, 82, 58, style="<-")

    # Agent → memory / planner
    arrow(63, 56, 58, 48)
    arrow(72, 56, 75, 48)
    arrow(58, 48, 63, 56, style="<-")
    arrow(75, 48, 72, 56, style="<-")

    # Agent → tools (як умовний пучок)
    for i in range(8):
        x = 2 + i * 12 + 5.5
        arrow(67, 56, x, 28, color="#999")

    # Tools → data
    for i in range(8):
        x = 2 + i * 12 + 5.5
        arrow(x, 20, 30, 13, color="#bbb")

    plt.tight_layout()
    out = IMG_DIR / "fig1_architecture.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out}")


# ============================================================
#   Діаграма 2. ReAct-цикл
# ============================================================
def draw_react_loop():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")

    def box(x, y, w, h, text, color, edge, fs=10, weight="bold"):
        bb = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4",
                            linewidth=1.8, facecolor=color, edgecolor=edge)
        ax.add_patch(bb)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, weight=weight)

    def arrow(x1, y1, x2, y2, label="", color="#555", curve=0.0):
        a = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle="->", mutation_scale=18,
                            linewidth=1.5, color=color,
                            connectionstyle=f"arc3,rad={curve}")
        ax.add_patch(a)
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1.5, label,
                    ha="center", fontsize=8.5, style="italic", color="#444")

    box(5, 22, 16, 10, "1. Запит\nкористувача", "#FFF2CC", "#BF8F00")
    box(28, 35, 18, 10, "2. Thought\n(міркування)", "#D5E8F0", "#2E75B6")
    box(54, 35, 18, 10, "3. Action\n(виклик tool)", "#D5E8D4", "#70AD47")
    box(78, 22, 18, 10, "4. Observation\n(результат)", "#F8CECC", "#B85450")
    box(28, 6, 50, 10, "5. Final answer (синтез відповіді)", "#E1D5E7", "#9673A6")

    arrow(21, 27, 28, 40, "input")
    arrow(46, 40, 54, 40, "tool_call")
    arrow(72, 40, 78, 32, "execute")
    arrow(78, 25, 46, 36, "observation\n(feed back)", curve=-0.25)
    arrow(37, 35, 37, 16, "якщо інформації\nдостатньо", color="#9673A6")

    plt.tight_layout()
    out = IMG_DIR / "fig2_react_loop.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out}")


# ============================================================
#   Діаграма 3. Пам'ять
# ============================================================
def draw_memory():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")

    def box(x, y, w, h, title, body, color, edge):
        bb = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5",
                            linewidth=1.8, facecolor=color, edgecolor=edge)
        ax.add_patch(bb)
        ax.text(x + w / 2, y + h - 3, title, ha="center", va="top",
                fontsize=11, weight="bold")
        ax.text(x + w / 2, y + h / 2 - 2, body, ha="center", va="center",
                fontsize=9, wrap=True)

    box(5, 7, 38, 38,
        "Короткострокова пам'ять",
        "• Історія повідомлень\n  (user / assistant / tool)\n\n"
        "• Зберігається в session_state\n\n"
        "• Передається в кожен виклик LLM\n  як messages[]\n\n"
        "• Скидається кнопкою «Reset»",
        "#FFF2CC", "#BF8F00")

    box(55, 7, 40, 38,
        "Довгострокова пам'ять",
        "• data/user_profile.json\n\n"
        "• Зберігає:\n"
        "    — ім'я користувача\n"
        "    — бажаний тариф\n"
        "    — нотатки\n\n"
        "• Підвантажується в SYSTEM_PROMPT\n  на кожній сесії",
        "#E1D5E7", "#9673A6")

    plt.tight_layout()
    out = IMG_DIR / "fig3_memory.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out}")


# ============================================================
#   Діаграма 4. Приклад даних
# ============================================================
def draw_data_sample():
    import pandas as pd
    df = pd.read_csv(IMG_DIR.parent / "data" / "consumption.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5))

    # (a) добове споживання за рік
    by_day = df.groupby(df["timestamp"].dt.date)["consumption_kwh"].sum()
    axes[0].plot(pd.to_datetime(by_day.index), by_day.values,
                 color="#2E75B6", linewidth=1.0)
    axes[0].fill_between(pd.to_datetime(by_day.index), by_day.values,
                         alpha=0.2, color="#2E75B6")
    axes[0].set_title("(a) Добове споживання за 2024 рік, кВт·год", fontsize=11, weight="bold")
    axes[0].set_xlabel("Місяць")
    axes[0].set_ylabel("кВт·год / добу")
    axes[0].grid(alpha=0.3)

    # (b) середній добовий профіль
    profile = df.groupby(df["timestamp"].dt.hour)["consumption_kwh"].mean()
    axes[1].bar(profile.index, profile.values, color="#70AD47", alpha=0.85)
    axes[1].set_title("(b) Середній добовий профіль (24 год)", fontsize=11, weight="bold")
    axes[1].set_xlabel("Година доби")
    axes[1].set_ylabel("кВт·год")
    axes[1].set_xticks(range(0, 24))
    axes[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out = IMG_DIR / "fig4_data_sample.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out}")


# ============================================================
#   Діаграма 5. Порівняння тарифів (приклад візуалізації результату)
# ============================================================
def draw_tariff_comparison():
    fig, ax = plt.subplots(figsize=(8, 4.5))
    tariffs = ["Одна зона", "Дві зони\n(день/ніч)", "Три зони\n(пік/н-пік/ніч)"]
    costs = [15466.42, 14251.56, 16756.25]
    colors = ["#5B9BD5", "#70AD47", "#ED7D31"]

    bars = ax.bar(tariffs, costs, color=colors, alpha=0.9, edgecolor="black", linewidth=0.6)
    best_idx = costs.index(min(costs))
    bars[best_idx].set_edgecolor("#2e7d32")
    bars[best_idx].set_linewidth(2.5)

    for b, c in zip(bars, costs):
        ax.text(b.get_x() + b.get_width() / 2, c + 200, f"{c:.0f} грн",
                ha="center", fontsize=10, weight="bold")

    ax.set_ylabel("Витрати, грн")
    ax.grid(alpha=0.3, axis="y")
    ax.text(best_idx, min(costs) / 2, "★ найвигідніше",
            ha="center", fontsize=11, weight="bold", color="#2e7d32")

    plt.tight_layout()
    out = IMG_DIR / "fig5_tariff_comparison.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out}")


if __name__ == "__main__":
    print("Генерація діаграм…")
    draw_architecture()
    draw_react_loop()
    draw_memory()
    draw_data_sample()
    draw_tariff_comparison()
    print("Готово.")
