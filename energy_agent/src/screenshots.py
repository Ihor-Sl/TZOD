"""Мокап-скріншоти UI для звіту (без emoji)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

IMG_DIR = Path(__file__).resolve().parent.parent / "images"


def draw_ui_mockup(title, dialog, out_name, trace=None):
    fig = plt.figure(figsize=(13, 9), facecolor="white")
    gs = fig.add_gridspec(1, 4, wspace=0.05)

    ax_side = fig.add_subplot(gs[0, 0])
    ax_side.set_xlim(0, 10); ax_side.set_ylim(0, 100); ax_side.axis("off")
    ax_side.add_patch(Rectangle((0, 0), 10, 100, facecolor="#f0f2f6", edgecolor="#ddd"))
    ax_side.text(5, 95, "ЕнергоАгент", ha="center", fontsize=13, weight="bold")
    ax_side.add_patch(FancyBboxPatch((2.5, 88), 5, 3, boxstyle="round,pad=0.2",
                                     facecolor="#d4f0d4", edgecolor="#2e7d32"))
    ax_side.text(5, 89.5, "Режим: LLM", ha="center", fontsize=8.5, weight="bold", color="#2e7d32")
    ax_side.text(0.5, 82, "Датасет", fontsize=10, weight="bold")
    ax_side.text(0.5, 78, "Записів: 8 784", fontsize=8.5)
    ax_side.text(0.5, 75, "Спожито: 3 580 кВт·год", fontsize=8.5)
    ax_side.text(0.5, 72, "Період: 2024-01 → 2024-12", fontsize=8)
    ax_side.text(0.5, 64, "Довгострокова пам'ять", fontsize=10, weight="bold")
    ax_side.text(0.5, 60, "Ім'я:    Ihor", fontsize=8.5, family="monospace")
    ax_side.text(0.5, 57, "Тариф:   two_zone", fontsize=8.5, family="monospace")
    ax_side.add_patch(FancyBboxPatch((0.5, 50), 9, 4, boxstyle="round,pad=0.2",
                                     facecolor="#fff", edgecolor="#bbb"))
    ax_side.text(5, 52, "Зберегти пам'ять", ha="center", fontsize=8.5, weight="bold")
    ax_side.text(0.5, 43, "Приклади запитів", fontsize=10, weight="bold")
    examples = ["Скільки я спожив у січні?",
                "Який тариф вигідніший?",
                "Прогноз на 7 днів",
                "Пікові години?",
                "Аномалії в грудні?"]
    for i, ex in enumerate(examples):
        ax_side.add_patch(FancyBboxPatch((0.3, 36 - i*5.2), 9.4, 3.7, boxstyle="round,pad=0.15",
                                         facecolor="#fff", edgecolor="#ccc"))
        ax_side.text(5, 37.7 - i*5.2, ex, ha="center", fontsize=7.8)

    ax_chat = fig.add_subplot(gs[0, 1:])
    ax_chat.set_xlim(0, 100); ax_chat.set_ylim(0, 100); ax_chat.axis("off")
    ax_chat.text(3, 96, title, fontsize=12.5, weight="bold")
    ax_chat.text(3, 93, "ШІ-агент, що поєднує LLM з 8 інструментами доступу до даних, пам'яттю та ReAct-плануванням.",
                 fontsize=8, style="italic", color="#666")

    y = 88
    for role, content in dialog:
        if role == "user":
            color, edge, label = "#d4edff", "#2E75B6", "Користувач:"
        else:
            color, edge, label = "#e8f5e9", "#70AD47", "Агент:"
        lines = content.split("\n")
        h = max(5, 3 + 2.2 * len(lines))
        ax_chat.add_patch(FancyBboxPatch((3, y - h), 92, h, boxstyle="round,pad=0.4",
                                         facecolor=color, edgecolor=edge, linewidth=1.2))
        ax_chat.text(4, y - 1.5, label, fontsize=8.5, weight="bold", color=edge)
        for i, line in enumerate(lines):
            ff = "monospace" if line.startswith(" ") else None
            ax_chat.text(5, y - 3.7 - i*2.2, line, fontsize=8, family=ff)
        y -= h + 1.5

    if trace and y > 18:
        h = 3 + 2.0 * len(trace)
        ax_chat.add_patch(FancyBboxPatch((3, y - h - 1), 92, h, boxstyle="round,pad=0.4",
                                         facecolor="#f6f8fa", edgecolor="#bbb"))
        ax_chat.text(4, y - 2.5, "ReAct-траса (як агент думав):", fontsize=8.5, weight="bold")
        for i, step in enumerate(trace):
            ax_chat.text(5, y - 4.7 - i*2.0, step, fontsize=7.3, family="monospace")

    plt.tight_layout(pad=0.5)
    out = IMG_DIR / out_name
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out}")


if __name__ == "__main__":
    draw_ui_mockup(
        "ЕнергоАгент — приклад роботи: статистика за період",
        [("user", "Скільки я спожив у січні?"),
         ("assistant",
          "Підсумок споживання за січень 2024:\n"
          "    • Всього:        457.29 кВт·год за 31 днів\n"
          "    • Середньо/добу: 14.75 кВт·год\n"
          "    • Максимум:      24.32 кВт·год (17.01.2024)\n"
          "    • Мінімум:       13.12 кВт·год (22.01.2024)\n"
          "Січневе споживання на 38% вище від річного середнього —\n"
          "це типово через опалення та коротший світловий день.")],
        "fig6_ui_demo1.png",
        trace=["[Thought] Користувач питає про січень — викличу get_consumption_summary",
               "[Action]  get_consumption_summary(start='2024-01-01', end='2024-01-31')",
               "[Result]  {total_kwh: 457.29, average_daily: 14.75, max: 24.32, ...}",
               "[Final]   Сформовано відповідь з інтерпретацією"])

    draw_ui_mockup(
        "ЕнергоАгент — рекомендація тарифу",
        [("user", "Який тариф мені вигідніший за весь рік?"),
         ("assistant",
          "Порівняння тарифів за 2024 рік (3 580 кВт·год):\n"
          "    • Одна зона:                15 466.42 грн\n"
          "    • Дві зони (день/ніч):      14 251.56 грн  <- найвигідніше\n"
          "    • Три зони (пік/н-пік/ніч): 16 756.25 грн\n"
          "\n"
          "Рекомендую перейти на тариф 'Дві зони'.\n"
          "Економія: 1 214.86 грн порівняно з 'Одна зона',\n"
          "         2 504.69 грн порівняно з тризонним тарифом.")],
        "fig7_ui_demo2.png",
        trace=["[Thought] Запит на порівняння тарифів — викличу compare_tariffs",
               "[Action]  compare_tariffs(start='2024-01-01', end='2024-12-31')",
               "[Result]  {best_tariff: 'two_zone', savings_vs_worst: 2504.69, ...}",
               "[Final]   Сформовано відповідь з рекомендацією"])

    draw_ui_mockup(
        "ЕнергоАгент — багатокроковий ReAct (2 інструменти підряд)",
        [("user", "Знайди аномалії в грудні і покажи на графіку"),
         ("assistant",
          "Знайдено 1 аномалію в грудні 2024:\n"
          "    • 2024-12-22: 26.47 кВт·год (z-score = 5.27)\n"
          "\n"
          "Це понад 5 стандартних відхилень від середнього грудневого\n"
          "значення. Імовірна причина — прийом гостей на свята або\n"
          "одночасна робота кількох потужних споживачів.\n"
          "Графік нижче показує піковий день у контексті місяця.")],
        "fig8_ui_demo3.png",
        trace=["[Thought] Потрібні і аномалії, і графік — викличу 2 інструменти",
               "[Action 1] find_anomalies(start='2024-12-01', end='2024-12-31')",
               "[Result 1] {anomalies_count: 1, anomalies: [{date: '2024-12-22'...}]}",
               "[Thought] Тепер генерую графік",
               "[Action 2] generate_chart(chart_type='daily', start='2024-12-01'...)",
               "[Result 2] {chart_path: '/charts/chart_daily_120345.png'}",
               "[Final]   Сформовано відповідь з інтерпретацією + графіком"])
