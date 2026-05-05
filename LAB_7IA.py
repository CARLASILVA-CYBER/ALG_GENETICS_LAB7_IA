import numpy as np
import random
import math
from Orange.data import Table, Domain, ContinuousVariable, StringVariable

# ============================================================
# 1. LER OS DADOS DO ORANGE
# ============================================================

data = in_data

if data is None:
    raise ValueError("No data received. Connect the File widget to the Python Script widget.")

# Extract coordinates
x_values = np.array(data[:, "X"]).astype(float).flatten()
y_values = np.array(data[:, "Y"]).astype(float).flatten()

# Extract local names
local_names = [str(row["Local"]) for row in data]

points = list(zip(x_values, y_values))
n_points = len(points)

# ============================================================
# 2. FUNÇÃO PARA CALCULAR DISTÂNCIA ENTRE DOIS PONTOS
# ============================================================

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# ============================================================
# 3. FUNÇÃO DE FITNESS
# ============================================================

def route_distance(route):
    total = 0

    for i in range(len(route) - 1):
        total += distance(points[route[i]], points[route[i + 1]])

    # Return to the starting point
    total += distance(points[route[-1]], points[route[0]])

    return total

def fitness(route):
    return 1 / route_distance(route)

# ============================================================
# 4. CRIAR POPULAÇÃO INICIAL
# ============================================================

def create_individual():
    individual = list(range(n_points))
    random.shuffle(individual)
    return individual

def create_population(size):
    return [create_individual() for _ in range(size)]

# ============================================================
# 5. SELEÇÃO
# ============================================================

def selection(population):
    tournament = random.sample(population, 3)
    tournament.sort(key=route_distance)
    return tournament[0]

# ============================================================
# 6. CRUZAMENTO
# ============================================================

def crossover(parent1, parent2):
    start, end = sorted(random.sample(range(n_points), 2))

    child = [-1] * n_points

    # Copy part of parent1
    child[start:end] = parent1[start:end]

    # Fill with genes from parent2
    position = end

    for gene in parent2:
        if gene not in child:
            if position >= n_points:
                position = 0
            child[position] = gene
            position += 1

    return child

# ============================================================
# 7. MUTAÇÃO
# ============================================================

def mutation(individual, mutation_rate):
    if random.random() < mutation_rate:
        i, j = random.sample(range(n_points), 2)
        individual[i], individual[j] = individual[j], individual[i]

    return individual

# ============================================================
# 8. ALGORITMO GENÉTICO
# ============================================================

population_size = 100
generations = 200
mutation_rate = 0.10

population = create_population(population_size)

best_route = None
best_distance = float("inf")

for generation in range(generations):
    new_population = []

    population.sort(key=route_distance)

    # Elitism: keep the best solution
    new_population.append(population[0])

    while len(new_population) < population_size:
        parent1 = selection(population)
        parent2 = selection(population)

        child = crossover(parent1, parent2)
        child = mutation(child, mutation_rate)

        new_population.append(child)

    population = new_population

    current_best = population[0]
    current_distance = route_distance(current_best)

    if current_distance < best_distance:
        best_distance = current_distance
        best_route = current_best

# ============================================================
# 9. PREPARAR RESULTADOS PARA O ORANGE
# ============================================================

route_names = [local_names[i] for i in best_route]
route_names.append(local_names[best_route[0]])

output_rows = []

for order, index in enumerate(best_route):
    output_rows.append([
        order + 1,
        local_names[index],
        points[index][0],
        points[index][1],
        best_distance
    ])

# Add return to start
first_index = best_route[0]
output_rows.append([
    n_points + 1,
    local_names[first_index],
    points[first_index][0],
    points[first_index][1],
    best_distance
])

domain = Domain(
    [
        ContinuousVariable("Order"),
        ContinuousVariable("X"),
        ContinuousVariable("Y"),
        ContinuousVariable("Best_Total_Distance")
    ],
    metas=[
        StringVariable("Local")
    ]
)

X = np.array([[row[0], row[2], row[3], row[4]] for row in output_rows])
metas = np.array([[row[1]] for row in output_rows], dtype=object)

out_data = Table.from_numpy(domain, X, metas=metas)

print("Best route found:")
print(" -> ".join(route_names))
print("Best total distance:", round(best_distance, 2))
