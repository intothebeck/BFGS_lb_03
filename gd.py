import numpy as np
import time

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

def line_search(x, grad):
    def phi(lambda_):
        return custom_func(x - lambda_ * grad)

    return golden_section_search(phi, 0, 1)

def gradient_descent(x0, tol=0.00001):
    x = x0
    iteration = 0
    start_time = time.time()
    while True:
        grad = custom_grad(x)
        lambda_ = line_search(x, grad)
        x_new = x - lambda_ * grad
        if np.linalg.norm(x_new - x) <= tol:
            break
        if abs(custom_func(x_new) - custom_func(x)) <= tol:
            break
        if np.linalg.norm(custom_grad(x_new)) <= tol:
            break
        x = x_new
        iteration += 1

    end_time = time.time()
    execution_time = end_time - start_time
    return x, iteration, custom_func(x), execution_time

if __name__ == "__main__":
    x0 = np.array([0., 0.])
    tol = 0.0001
    xmin, num_iterations, fmin, execution_time = gradient_descent(x0, tol)
    print("Метод Градиентного спуска с золотым сечением:")
    print("Точка минимума:", xmin)
    print("Число итераций:", num_iterations)
    print("Минимальное значение функции:", fmin)
    print(f"Время выполнения метода: {execution_time:.10f} секунд")
