"""
Практична робота №6
Оптимізація коду для обробки великих наборів даних

Порівнюємо 3 підходи:
  1. Звичайний цикл (базовий)
  2. NumPy (векторизація)
  3. Multiprocessing (паралельні обчислення)
"""

import numpy as np
import time
import multiprocessing
import matplotlib.pyplot as plt


# ─────────────────────────────────────────
# Функції — оголошуємо ДО if __name__
# (multiprocessing вимагає це на Windows)
# ─────────────────────────────────────────

def process_loop(arr):
    """Звичайний цикл — найповільніший спосіб."""
    result = []
    for x in arr:
        result.append(np.sqrt(x**2 + 2*x + 1) / (x + 1))
    return result


def process_numpy(arr):
    """NumPy — операція одразу над усім масивом."""
    return np.sqrt(arr**2 + 2*arr + 1) / (arr + 1)


def process_chunk(chunk):
    """Обробка одного шматка — запускається в окремому процесі."""
    return np.sqrt(chunk**2 + 2*chunk + 1) / (chunk + 1)


def process_parallel(arr, n_workers=None):
    """Ділимо масив на шматки, кожен — на окремому ядрі."""
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    chunks = np.array_split(arr, n_workers)
    with multiprocessing.Pool(processes=n_workers) as pool:
        results = pool.map(process_chunk, chunks)
    return np.concatenate(results)


# ─────────────────────────────────────────
# ОБОВ'ЯЗКОВО на Windows: весь запуск
# тільки всередині if __name__ == '__main__'
# ─────────────────────────────────────────

if __name__ == '__main__':

    SIZE = 10_000_000
    data = np.random.uniform(1, 1000, SIZE)

    # --- Метод 1: цикл (на 500к, решта — оцінка) ---
    print("Метод 1: звичайний цикл...")
    t0 = time.time()
    res_loop = process_loop(data[:500_000])
    t_loop = time.time() - t0
    t_loop_full = t_loop * 20
    print(f"  500 000 елементів -> {t_loop:.2f} сек")
    print(f"  Оцінка для 10 млн -> ~{t_loop_full:.1f} сек\n")

    # --- Метод 2: NumPy ---
    print("Метод 2: NumPy...")
    t0 = time.time()
    res_numpy = process_numpy(data)
    t_numpy = time.time() - t0
    print(f"  10 000 000 елементів -> {t_numpy:.4f} сек\n")

    # --- Метод 3: Parallel ---
    print("Метод 3: Multiprocessing...")
    n_cpu = multiprocessing.cpu_count()
    print(f"  Доступно CPU ядер: {n_cpu}")
    t0 = time.time()
    res_parallel = process_parallel(data)
    t_parallel = time.time() - t0
    print(f"  10 000 000 елементів -> {t_parallel:.4f} сек\n")

    # --- Перевірка ---
    diff = np.max(np.abs(res_numpy - res_parallel))
    print(f"Різниця NumPy vs Parallel: {diff:.2e}")

    # --- Звіт ---
    print("\n" + "="*48)
    print("  ПОРІВНЯННЯ МЕТОДІВ (10 млн елементів)")
    print("="*48)
    print(f"  {'Метод':<25} {'Час':>8}  {'Прискорення':>12}")
    print("-"*48)
    print(f"  {'Цикл (оцінка)':<25} {t_loop_full:>7.1f}с  {'1.0x':>12}")
    print(f"  {'NumPy':<25} {t_numpy:>7.4f}с  {t_loop_full/t_numpy:>11.0f}x")
    print(f"  {f'Parallel ({n_cpu} ядра)':<25} {t_parallel:>7.4f}с  {t_loop_full/t_parallel:>11.0f}x")
    print("="*48)

    # --- Графік ---
    methods = ['Цикл\n(оцінка)', 'NumPy', f'Parallel\n({n_cpu} ядра)']
    times   = [t_loop_full, t_numpy, t_parallel]
    colors  = ['#e74c3c', '#2ecc71', '#3498db']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Оптимізація обробки великих даних (10 млн елементів)',
                 fontsize=13, fontweight='bold')

    ax = axes[0]
    bars = ax.bar(methods, times, color=colors, width=0.5)
    ax.set_ylabel('Час (секунди, лог шкала)')
    ax.set_title('Час виконання')
    for bar, t in zip(bars, times):
        label = f'{t:.0f}с' if t >= 1 else f'{t*1000:.0f}мс'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                label, ha='center', fontweight='bold')
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)

    ax = axes[1]
    speedups = [1, t_loop_full/t_numpy, t_loop_full/t_parallel]
    bars2 = ax.bar(methods, speedups, color=colors, width=0.5)
    ax.set_ylabel('Прискорення (разів)')
    ax.set_title('Прискорення відносно циклу')
    for bar, s in zip(bars2, speedups):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                f'{s:.0f}x', ha='center', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()