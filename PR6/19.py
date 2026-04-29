"""
Практична робота 6
Вимірювання інтенсивності світла після відбиття від різних поверхонь
Поверхні: метал, скло, дерево, стіна

Дані зчитуються з CSV-файлів датчика освітленості телефону (Phyphox / Science Journal)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import savgol_filter
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# 1. ЗЧИТУВАННЯ ДАНИХ З CSV
# ─────────────────────────────────────────────────────────────

# Еталонні коефіцієнти відбиття та кольори для графіків
SURFACE_CONFIG = {
    "Метал":  {"true_reflectance": 0.72, "color": "#5B8DB8"},
    "Скло":   {"true_reflectance": 0.64, "color": "#7EC8C8"},
    "Дерево": {"true_reflectance": 0.28, "color": "#C89B6E"},
    "Стіна":  {"true_reflectance": 0.42, "color": "#B5A8D5"},
}

all_dfs = {}

for surface in SURFACE_CONFIG:
    fname = f"sensor_data/{surface.lower()}_light.csv"
    df = pd.read_csv(fname)
    all_dfs[surface] = df
    print(f"[✓] Зчитано: {fname}  ({len(df)} записів)")

# ─────────────────────────────────────────────────────────────
# 2. ОБРОБКА ДАНИХ
# ─────────────────────────────────────────────────────────────

stats_rows = []

for surface, df in all_dfs.items():
    # Фільтрація Savitzky–Golay (згладжування як при реальній обробці)
    df["lux_reflected_smooth"] = savgol_filter(df["lux_reflected"], window_length=11, polyorder=2)
    df["reflectance"] = df["lux_reflected_smooth"] / df["lux_incident"]

    r = df["reflectance"]
    inc = df["lux_incident"]
    ref = df["lux_reflected_smooth"]

    stats_rows.append({
        "Поверхня":              surface,
        "Серед. падаюче (lux)":  round(inc.mean(), 1),
        "Серед. відбите (lux)":  round(ref.mean(), 1),
        "Коеф. відбиття (R)":    round(r.mean(), 4),
        "Std R":                 round(r.std(), 4),
        "Min R":                 round(r.min(), 4),
        "Max R":                 round(r.max(), 4),
        "95% CI":                f"[{round(r.mean()-1.96*r.std()/np.sqrt(len(r)),4)}, "
                                 f"{round(r.mean()+1.96*r.std()/np.sqrt(len(r)),4)}]",
        "Еталон R":              SURFACE_CONFIG[surface]["true_reflectance"],
        "color":                 SURFACE_CONFIG[surface]["color"],
    })

stats_df = pd.DataFrame(stats_rows)
print("\n── Зведена статистика ──")
print(stats_df[["Поверхня","Серед. падаюче (lux)","Серед. відбите (lux)",
                "Коеф. відбиття (R)","Std R","95% CI"]].to_string(index=False))

# ─────────────────────────────────────────────────────────────
# 3. ВІЗУАЛІЗАЦІЯ
# ─────────────────────────────────────────────────────────────

COLORS  = [r["color"]    for r in stats_rows]
SURFACES = [r["Поверхня"] for r in stats_rows]
R_MEAN  = [r["Коеф. відбиття (R)"] for r in stats_rows]
R_STD   = [r["Std R"]              for r in stats_rows]
R_REF   = [r["Еталон R"]           for r in stats_rows]

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "figure.facecolor": "#0F1117",
    "axes.facecolor":   "#1A1D27",
    "axes.edgecolor":   "#3A3D4D",
    "text.color":       "#E0E0F0",
    "axes.labelcolor":  "#B0B0D0",
    "xtick.color":      "#A0A0C0",
    "ytick.color":      "#A0A0C0",
    "grid.color":       "#2A2D3D",
    "grid.alpha":       0.6,
})

fig = plt.figure(figsize=(20, 22))
fig.suptitle(
    "Практична робота 6 · Аналіз відбиття світла від різних поверхонь",
    fontsize=18, fontweight="bold", color="#E8E8FF", y=0.98
)

gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

# ── 3.1 Часові ряди відбитого світла (по поверхні) ────────────
for idx, (surface, df) in enumerate(all_dfs.items()):
    ax = fig.add_subplot(gs[idx // 2, idx % 2])
    color = SURFACE_CONFIG[surface]["color"]

    ax.plot(df["time_s"], df["lux_reflected"],
            color=color, alpha=0.3, linewidth=0.8, label="Сирі дані")
    ax.plot(df["time_s"], df["lux_reflected_smooth"],
            color=color, linewidth=2.0, label="Згладжено (S-G)")
    ax.axhline(df["lux_reflected_smooth"].mean(), color="#FFD700",
               linestyle="--", linewidth=1.2, label=f"Середнє: {df['lux_reflected_smooth'].mean():.1f} lux")

    ax.set_title(f"Поверхня: {surface}", fontsize=13, color=color, pad=8)
    ax.set_xlabel("Час (с)", fontsize=10)
    ax.set_ylabel("Відбита освітленість (lux)", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True)

# ── 3.2 Стовпчаста діаграма коефіцієнтів відбиття ─────────────
ax5 = fig.add_subplot(gs[2, 0])
x = np.arange(len(SURFACES))
bars = ax5.bar(x, R_MEAN, color=COLORS, width=0.5, zorder=3,
               yerr=R_STD, capsize=6, error_kw={"ecolor":"#FFD700","elinewidth":1.5})
ax5.scatter(x, R_REF, color="#FF6B6B", zorder=5, s=80,
            marker="D", label="Еталонне значення")

for bar, val in zip(bars, R_MEAN):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
             f"{val:.3f}", ha="center", va="bottom", fontsize=10,
             color="#E0E0F0", fontweight="bold")

ax5.set_xticks(x)
ax5.set_xticklabels(SURFACES, fontsize=11)
ax5.set_ylabel("Коефіцієнт відбиття R", fontsize=11)
ax5.set_title("Середній коефіцієнт відбиття (з похибкою ±σ)", fontsize=12, pad=8)
ax5.set_ylim(0, 1.0)
ax5.legend(fontsize=9)
ax5.grid(True, axis="y")

# ── 3.3 Розподіл R — скрипкові графіки ────────────────────────
ax6 = fig.add_subplot(gs[2, 1])
reflectance_data = [all_dfs[s]["reflectance"].values for s in SURFACES]

vp = ax6.violinplot(reflectance_data, positions=range(len(SURFACES)),
                    showmeans=True, showmedians=True)
for body, color in zip(vp["bodies"], COLORS):
    body.set_facecolor(color)
    body.set_alpha(0.65)
for part in ("cmeans","cmedians","cbars","cmins","cmaxes"):
    vp[part].set_color("#FFD700")
    vp[part].set_linewidth(1.5)

ax6.set_xticks(range(len(SURFACES)))
ax6.set_xticklabels(SURFACES, fontsize=11)
ax6.set_ylabel("Коефіцієнт відбиття R", fontsize=11)
ax6.set_title("Розподіл коефіцієнта відбиття (violin plot)", fontsize=12, pad=8)
ax6.grid(True, axis="y")

# ── 3.4 Порівняння: вимірянe vs еталон ────────────────────────
ax7 = fig.add_subplot(gs[3, 0])
x = np.arange(len(SURFACES))
w = 0.35
b1 = ax7.bar(x - w/2, R_MEAN, w, color=COLORS, label="Виміряно", zorder=3)
b2 = ax7.bar(x + w/2, R_REF,  w, color="#FF6B6B", alpha=0.75, label="Еталон", zorder=3)

ax7.set_xticks(x)
ax7.set_xticklabels(SURFACES, fontsize=11)
ax7.set_ylabel("R", fontsize=11)
ax7.set_title("Виміряний vs еталонний коефіцієнт відбиття", fontsize=12, pad=8)
ax7.legend(fontsize=9)
ax7.set_ylim(0, 1.0)
ax7.grid(True, axis="y")

# ── 3.5 Відносна похибка вимірювання ──────────────────────────
ax8 = fig.add_subplot(gs[3, 1])
rel_errors = [abs(m - e) / e * 100 for m, e in zip(R_MEAN, R_REF)]
bars2 = ax8.bar(SURFACES, rel_errors, color=COLORS, zorder=3)
for bar, val in zip(bars2, rel_errors):
    ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f"{val:.2f}%", ha="center", va="bottom", fontsize=10,
             color="#E0E0F0", fontweight="bold")

ax8.set_ylabel("Відносна похибка (%)", fontsize=11)
ax8.set_title("Відносна похибка від еталонного значення", fontsize=12, pad=8)
ax8.axhline(5, color="#FF6B6B", linestyle="--", linewidth=1.2, label="5% поріг")
ax8.legend(fontsize=9)
ax8.grid(True, axis="y")

plt.savefig("light_reflection_analysis.png", dpi=150,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()
print("\n[✓] Графік збережено: light_reflection_analysis.png")

# ─────────────────────────────────────────────────────────────
# 4. ПІДСУМКОВИЙ ЗВІТ У КОНСОЛЬ
# ─────────────────────────────────────────────────────────────

print("\n" + "═"*65)
print("  ПІДСУМКИ АНАЛІЗУ ВІДБИТТЯ СВІТЛА")
print("═"*65)
for row in stats_rows:
    s = row["Поверхня"]
    r = row["Коеф. відбиття (R)"]
    e = row["Еталон R"]
    err = abs(r - e) / e * 100
    print(f"\n  {s:8s}  |  R = {r:.4f}  |  Еталон = {e:.2f}  |  Похибка = {err:.2f}%")
    print(f"            Середнє відбите: {row['Серед. відбите (lux)']:.1f} lux")
    print(f"            95% CI: {row['95% CI']}")

print("\n" + "═"*65)
print("\n  Ранжування поверхонь за коефіцієнтом відбиття:")
ranked = sorted(stats_rows, key=lambda x: x["Коеф. відбиття (R)"], reverse=True)
for i, row in enumerate(ranked, 1):
    bar = "█" * int(row["Коеф. відбиття (R)"] * 30)
    print(f"  {i}. {row['Поверхня']:8s}  R={row['Коеф. відбиття (R)']:.3f}  {bar}")

print("\n  Висновок:")
best = ranked[0]["Поверхня"]
worst = ranked[-1]["Поверхня"]
print(f"  • Найбільше відбиває: {best} (дзеркально-подібна поверхня)")
print(f"  • Найменше відбиває:  {worst} (матова поверхня, дифузне розсіювання)")
print(f"  • Похибки вимірювань: від {min(abs(r['Коеф. відбиття (R)']-r['Еталон R'])/r['Еталон R']*100 for r in stats_rows):.2f}%"
      f" до {max(abs(r['Коеф. відбиття (R)']-r['Еталон R'])/r['Еталон R']*100 for r in stats_rows):.2f}%")
print("  • Фільтр Savitzky–Golay успішно зменшив шум датчика.")
print("═"*65)