"""
Практична робота №4
Моделювання попиту та пропозиції електроенергії

Два методи:
  1. Диференціальні рівняння (ДР)
  2. Трендова екстраполяція (поліном)
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# ──────────────────────────────────────────
# МЕТОД 1: Диференціальні рівняння
# ──────────────────────────────────────────
#
# Ідея проста:
#   - Попит D зростає з часом (люди купують більше приладів)
#   - Пропозиція S реагує на попит із запізненням
#   - Якщо D > S — є дефіцит, якщо S > D — надлишок
#
# Система рівнянь:
#   dD/dt = a * D          (попит росте пропорційно собі — як депозит)
#   dS/dt = b * (D - S)    (пропозиція "наздоганяє" попит)

a = 0.1   # швидкість зростання попиту (10% на рік)
b = 0.3   # швидкість реакції пропозиції

def model(state, t):
    D, S = state
    dD_dt = a * D            # попит зростає
    dS_dt = b * (D - S)      # пропозиція тягнеться за попитом
    return [dD_dt, dS_dt]

# Початкові умови
D0, S0 = 100, 80   # попит = 100 ГВт, пропозиція = 80 ГВт (є дефіцит)

t = np.linspace(0, 20, 200)   # 20 років

# Розв'язуємо систему ДР
solution = odeint(model, [D0, S0], t)
D_ode = solution[:, 0]
S_ode = solution[:, 1]


# ──────────────────────────────────────────
# МЕТОД 2: Трендова екстраполяція
# ──────────────────────────────────────────
#
# Беремо "реальні" дані за перші 10 років,
# підбираємо поліном і продовжуємо лінію вперед.

# "Реальні" дані (перші 10 років) = беремо з ДР + додаємо випадковий шум
np.random.seed(42)
t_data = np.linspace(0, 10, 20)                        # 20 точок спостережень
idx    = np.searchsorted(t, t_data)                    # індекси відповідних моментів
D_data = D_ode[idx] + np.random.normal(0, 3, 20)       # попит зі шумом
S_data = S_ode[idx] + np.random.normal(0, 3, 20)       # пропозиція зі шумом

# Апроксимуємо поліномом 2-го степеня (парабола)
poly_D = np.polyfit(t_data, D_data, deg=2)
poly_S = np.polyfit(t_data, S_data, deg=2)

# Екстраполяція на всі 20 років
D_trend = np.polyval(poly_D, t)
S_trend = np.polyval(poly_S, t)


# ──────────────────────────────────────────
# ПОБУДОВА ГРАФІКІВ
# ──────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Моделювання попиту та пропозиції електроенергії", fontsize=13, fontweight='bold')

# --- Графік 1: Метод ДР ---
ax = axes[0]
ax.plot(t, D_ode, 'b-',  lw=2, label='Попит D')
ax.plot(t, S_ode, 'g-',  lw=2, label='Пропозиція S')
ax.fill_between(t, D_ode, S_ode,
                where=(D_ode > S_ode), alpha=0.2, color='red',   label='Дефіцит')
ax.fill_between(t, D_ode, S_ode,
                where=(S_ode > D_ode), alpha=0.2, color='green', label='Надлишок')
ax.set_title('Метод 1: Диференціальні рівняння')
ax.set_xlabel('Час (роки)')
ax.set_ylabel('Потужність (ГВт)')
ax.legend()
ax.grid(alpha=0.3)

# --- Графік 2: Трендова екстраполяція ---
ax = axes[1]
ax.scatter(t_data, D_data, color='blue',  s=20, zorder=5, label='Дані D (факт)')
ax.scatter(t_data, S_data, color='green', s=20, zorder=5, label='Дані S (факт)')
ax.plot(t, D_trend, 'b--', lw=2, label='Тренд D')
ax.plot(t, S_trend, 'g--', lw=2, label='Тренд S')
ax.axvline(10, color='gray', ls=':', lw=1.5, label='Кінець даних → прогноз')
ax.set_title('Метод 2: Трендова екстраполяція')
ax.set_xlabel('Час (роки)')
ax.set_ylabel('Потужність (ГВт)')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# --- Графік 3: Порівняння методів ---
ax = axes[2]
ax.plot(t, D_ode,   'b-',  lw=2,   label='ДР — попит')
ax.plot(t, D_trend, 'b--', lw=1.5, label='Тренд — попит')
ax.plot(t, S_ode,   'g-',  lw=2,   label='ДР — пропозиція')
ax.plot(t, S_trend, 'g--', lw=1.5, label='Тренд — пропозиція')
ax.axvline(10, color='gray', ls=':', lw=1.5, label='Межа прогнозу')
ax.set_title('Порівняння двох методів')
ax.set_xlabel('Час (роки)')
ax.set_ylabel('Потужність (ГВт)')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# ──────────────────────────────────────────
# ВИСНОВКИ (роздрукуються в консоль)
# ──────────────────────────────────────────

print("\n========= РЕЗУЛЬТАТИ =========")
print(f"Початковий попит:       {D0} ГВт")
print(f"Початкова пропозиція:   {S0} ГВт  (дефіцит {D0-S0} ГВт)")
print(f"\nЧерез 20 років (ДР):")
print(f"  Попит:       {D_ode[-1]:.1f} ГВт")
print(f"  Пропозиція:  {S_ode[-1]:.1f} ГВт")
print(f"  Дисбаланс:   {D_ode[-1]-S_ode[-1]:.1f} ГВт")

# Коли пропозиція наздоганяє попит?
cross_idx = np.where(S_ode >= D_ode)[0]
if len(cross_idx) > 0:
    print(f"\n  Пропозиція наздоганяє попит через ~{t[cross_idx[0]]:.1f} років")

print("\n========= ВИСНОВОК ===========")
print("ДР: враховує зворотній зв'язок між попитом і пропозицією.")
print("Тренд: простий, але не реагує на зміну умов ринку.")
print("Після межі прогнозу (10 р.) методи дають різні результати.")