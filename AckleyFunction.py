import random
import numpy as np
import matplotlib.pyplot as plt


# Функция Эккли.
def ackley(x, y):
    a = 20
    b = 0.2
    c = 2 * np.pi
    # Формула функции Эккли
    return -a * np.exp(-b * np.sqrt(0.5 * (x ** 2 + y ** 2))) - np.exp(
        0.5 * (np.cos(c * x) + np.cos(c * y))) + a + np.exp(1)


# Генерирует случайную частицу.
# Функция создает массив из случайных чисел в заданном диапазоне.
# Каждая частица представляет собой позицию в пространстве поиска.
def generate_random_particle(min_bound, max_bound, dim=2):
    return np.array([random.uniform(min_bound, max_bound) for _ in range(dim)])


# Вычисляет приспособленность частицы.
# В этой функции мы вычисляем значение целевой функции для текущей позиции частицы.
def calculate_fitness(position):
    return ackley(position[0], position[1])


# Реализация алгоритма роя частиц
def pso(population_size=50, dimensions=2, iterations=100, w=0.7, c1=1.5, c2=1.5, min_bound=-5, max_bound=5):
    particles = []

    # Создаем начальную популяцию частиц
    for _ in range(population_size):
        position = generate_random_particle(min_bound, max_bound, dimensions)
        velocity = generate_random_particle(-1, 1, dimensions)
        pbest_position = np.copy(position)
        pbest_fitness = calculate_fitness(position)
        particles.append({"position": position, "velocity": velocity, "pbest_position": pbest_position,
                          "pbest_fitness": pbest_fitness})

    gbest_position = particles[0]["pbest_position"]
    gbest_fitness = particles[0]["pbest_fitness"]


    for i in range(1, len(particles)):
        if particles[i]["pbest_fitness"] < gbest_fitness:
            gbest_fitness = particles[i]["pbest_fitness"]
            gbest_position = particles[i][
                "pbest_position"]

    # Основной цикл алгоритма PSO
    for iteration in range(iterations):
        for particle in particles:
            current_position = particle["position"]
            current_velocity = particle["velocity"]
            pbest_position = particle["pbest_position"]

            new_velocity = w * current_velocity + \
                           c1 * random.random() * (pbest_position - current_position) + \
                           c2 * random.random() * (gbest_position - current_position)

            new_position = current_position + new_velocity

            particle["velocity"] = new_velocity
            particle["position"] = new_position

            current_fitness = calculate_fitness(new_position)  # Вычисляем приспособленность
            if current_fitness < particle["pbest_fitness"]:
                particle["pbest_fitness"] = current_fitness
                particle["pbest_position"] = np.copy(new_position)  # Обновляем лучшую позицию частицы

            # Обновляем глобальную лучшую позицию и приспособленность
            if current_fitness < gbest_fitness:
                gbest_fitness = current_fitness
                gbest_position = np.copy(new_position)

        print(f"Итерация {iteration + 1}: gbest_fitness = {gbest_fitness}, gbest_position = {gbest_position}")

    return gbest_position, gbest_fitness


# Функция для отрисовки графика функции Эккли
def plot_ackley(x_range, y_range):
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = ackley(X, Y)
    plt.figure()
    plt.contourf(X, Y, Z, levels=50)
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Ackley function")
    plt.show()


# Пример использования
plot_ackley([-5, 5], [-5, 5])
best_position, best_fitness = pso()
print(
    f"\nНайденный минимум: Позиция = {best_position},  Значение функции = {best_fitness}")
