import numpy as np
import time

def golden_section_search(func, a, b, tol=0.0001):
    gr = (np.sqrt(5) + 1) / 2
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    while abs(b - a) > tol:
        if func(c) < func(d):
            b = d
        else:
            a = c

        c = b - (b - a) / gr
        d = a + (b - a) / gr

    return (b + a) / 2

def bfgs(func, grad, x0, tol=0.0001, max_iter=200):
    print("Метод BFGS:")
    start_time = time.time()
    x = x0
    n = len(x0)
    H = np.eye(n)  # матрица Гессе
    for iteration in range(max_iter):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            print(f"Число итераций {iteration}.")
            break
        # Вычисление направления поиска
        p = -np.dot(H, g)
        # Линейный поиск для определения шага alpha
        phi = lambda alpha: func(x + alpha * p)
        alpha = golden_section_search(phi, 0, 1)
        # Обновление точки
        x_new = x + alpha * p
        g_new = grad(x_new)
        # Разности для обновления H
        s = x_new - x
        y = g_new - g
        # Проверка деления на ноль
        if np.dot(y, s) == 0:
            print("Деление на ноль при вычислении rho.")
            break
        rho = 1.0 / np.dot(y, s)
        # Обновление матрицы H по формуле BFGS
        I = np.eye(n)
        H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
        x = x_new
    else:
        print("Достигнуто максимальное количество итераций.")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Время выполнения метода BFGS: {elapsed_time:.10f} секунд.")

    return x, func(x)

def custom_func(x):
    # return 4 * (x[0] - 5) ** 2 + (x[1] - 6) ** 2
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
    # return x[0] ** 2 + 100 * x[1] ** 2 + x[2] ** 2
    # return np.sqrt(x[0]**2 + x[1]**2)
    # noise = np.random.normal(0, 0.1)
    # return x[0]**2 + x[1]**2 + noise

def custom_grad(x):
    # return np.array([8 * (x[0] - 5), 2 * (x[1] - 6)])
    return np.array([ -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]), 200 * (x[1] - x[0] ** 2)
    ])
    # return np.array([
    #     2 * x[0],
    #     200 * x[1],
    #     2 * x[2]
    # ])
    # denom = np.sqrt(x[0] ** 2 + x[1] ** 2)
    # return np.array([x[0] / denom, x[1] / denom])
    # noise = np.random.normal(0, 0.05, size=x.shape)
    # return 2 * x + noise

if __name__ == "__main__":
    x0 = np.array([0., 0.])
    xmin, fmin = bfgs(custom_func, custom_grad, x0)
    print("Точка минимума:", xmin)
    print("Минимальное значение функции:", fmin)
