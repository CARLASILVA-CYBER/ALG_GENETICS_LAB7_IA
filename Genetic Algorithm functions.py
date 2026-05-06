import random
import math
import numpy as np
import pandas as pd

from Orange.data import Table, Domain, ContinuousVariable, StringVariable


# ============================================================
# 1. Convert Orange Table to pandas DataFrame
# ============================================================

# Numeric attributes
df = pd.DataFrame(
    in_data.X,
    columns=[var.name for var in in_data.domain.attributes]
)

# Meta/string columns, for example Local
for meta in in_data.domain.metas:
    df[meta.name] = [str(v) for v in in_data.get_column(meta)]

# If Local does not exist, create it automatically
if "Local" not in df.columns:
    df["Local"] = [f"Local_{i+1}" for i in range(len(df))]


# ============================================================
# 2. Check required columns
# ============================================================

required_columns = ["X", "Y"]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"The dataset must contain the column: {col}")

# ============================================================
# 3. Add realistic variables
# ============================================================

# Sem seed fixa: cada execução pode gerar resultados diferentes
# random.seed(42)
# np.random.seed(42)

if "TrafficLevel" not in df.columns:
    df["TrafficLevel"] = np.random.randint(1, 6, size=len(df))

# BlockedRoad:
# 0 = road available
# 1 = road blocked
if "BlockedRoad" not in df.columns:
    df["BlockedRoad"] = np.random.choice(
        [0, 1],
        size=len(df),
        p=[0.85, 0.15]
    )

# Priority:
# 1 = low priority
# 5 = high priority
if "Priority" not in df.columns:
    df["Priority"] = np.random.randint(1, 6, size=len(df))

# TeamAvailable:
# 0 = no team available
# 1 = team available
if "TeamAvailable" not in df.columns:
    df["TeamAvailable"] = np.random.choice(
        [0, 1],
        size=len(df),
        p=[0.20, 0.80]
    )

# FuelCost:
# estimated fuel cost factor
if "FuelCost" not in df.columns:
    df["FuelCost"] = np.random.uniform(1.0, 3.0, size=len(df))

# RiskLevel:
# 1 = low risk
# 5 = high risk
if "RiskLevel" not in df.columns:
    df["RiskLevel"] = np.random.randint(1, 6, size=len(df))


# ============================================================
# 4. Distance function
# ============================================================

def euclidean_distance(place_a, place_b):
    return math.sqrt(
        (place_a["X"] - place_b["X"]) ** 2 +
        (place_a["Y"] - place_b["Y"]) ** 2
    )


# ============================================================
# 5. Cost function
# ============================================================

def route_cost(route, data):
    total_cost = 0

    for i in range(len(route) - 1):
        current_place = data.iloc[route[i]]
        next_place = data.iloc[route[i + 1]]

        distance = euclidean_distance(current_place, next_place)

        traffic = next_place["TrafficLevel"]
        blocked = next_place["BlockedRoad"]
        priority = next_place["Priority"]
        team = next_place["TeamAvailable"]
        fuel = next_place["FuelCost"]
        risk = next_place["RiskLevel"]

        # Penalties and bonuses
        blocked_penalty = 100 if blocked == 1 else 0
        team_penalty = 50 if team == 0 else 0
        priority_bonus = priority * 2

        # Global segment cost
        segment_cost = (
            distance
            + traffic * 2
            + fuel * distance
            + risk * 3
            + blocked_penalty
            + team_penalty
            - priority_bonus
        )

        total_cost += segment_cost

    return total_cost


def route_distance(route, data):
    total_distance = 0

    for i in range(len(route) - 1):
        current_place = data.iloc[route[i]]
        next_place = data.iloc[route[i + 1]]
        total_distance += euclidean_distance(current_place, next_place)

    return total_distance


# ============================================================
# 6. Genetic Algorithm functions
# ============================================================

def create_individual(number_of_places):
    route = list(range(number_of_places))
    random.shuffle(route)

    # Return to the starting point
    route.append(route[0])

    return route


def create_population(population_size, number_of_places):
    return [
        create_individual(number_of_places)
        for _ in range(population_size)
    ]


def repair_route(route):
    """
    Ensures that the route visits each place only once
    and returns to the starting point.
    """
    route_without_return = route[:-1]

    seen = []
    for place in route_without_return:
        if place not in seen:
            seen.append(place)

    missing = [
        place for place in range(len(route_without_return))
        if place not in seen
    ]

    repaired = seen + missing
    repaired.append(repaired[0])

    return repaired


def selection(population, data):
    sorted_population = sorted(
        population,
        key=lambda route: route_cost(route, data)
    )

    return sorted_population[:len(sorted_population) // 2]


def crossover(parent1, parent2):
    parent1_core = parent1[:-1]
    parent2_core = parent2[:-1]

    size = len(parent1_core)

    start = random.randint(0, size - 2)
    end = random.randint(start + 1, size - 1)

    child_core = [-1] * size

    # Copy part of parent1
    child_core[start:end] = parent1_core[start:end]

    # Complete with parent2
    pointer = 0

    for gene in parent2_core:
        if gene not in child_core:
            while child_core[pointer] != -1:
                pointer += 1
            child_core[pointer] = gene

    child_core.append(child_core[0])

    return child_core


def mutate(route, mutation_rate):
    route_core = route[:-1]

    if random.random() < mutation_rate:
        i, j = random.sample(range(len(route_core)), 2)
        route_core[i], route_core[j] = route_core[j], route_core[i]

    route_core.append(route_core[0])

    return route_core


def genetic_algorithm(
    data,
    population_size=100,
    generations=200,
    mutation_rate=0.10,
    elitism_size=5
):
    number_of_places = len(data)

    population = create_population(
        population_size,
        number_of_places
    )

    best_route = None
    best_cost = float("inf")

    for generation in range(generations):

        population = sorted(
            population,
            key=lambda route: route_cost(route, data)
        )

        current_best = population[0]
        current_cost = route_cost(current_best, data)

        if current_cost < best_cost:
            best_cost = current_cost
            best_route = current_best.copy()

        # Elitism: keep the best routes
        new_population = population[:elitism_size]

        selected = selection(population, data)

        while len(new_population) < population_size:
            parent1, parent2 = random.sample(selected, 2)

            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            child = repair_route(child)

            new_population.append(child)

        population = new_population

    return best_route, best_cost


# ============================================================
# 7. Parameters
# ============================================================

population_size = 100
generations = 200
mutation_rate = 0.10
elitism_size = 5


# ============================================================
# 8. Run Genetic Algorithm
# ============================================================

best_route, best_cost = genetic_algorithm(
    data=df,
    population_size=population_size,
    generations=generations,
    mutation_rate=mutation_rate,
    elitism_size=elitism_size
)

best_distance = route_distance(best_route, df)


# ============================================================
# 9. Create output table
# ============================================================

output_rows = []

for order, index in enumerate(best_route, start=1):
    place = df.iloc[index]

    output_rows.append({
        "Order": order,
        "Local": place["Local"],
        "X": place["X"],
        "Y": place["Y"],
        "TrafficLevel": place["TrafficLevel"],
        "BlockedRoad": place["BlockedRoad"],
        "Priority": place["Priority"],
        "TeamAvailable": place["TeamAvailable"],
        "FuelCost": round(place["FuelCost"], 2),
        "RiskLevel": place["RiskLevel"],
        "BestRouteCost": round(best_cost, 2),
        "BestRouteDistance": round(best_distance, 2)
    })

result_df = pd.DataFrame(output_rows)


# ============================================================
# 10. Send result back to Orange
# ============================================================

continuous_vars = [
    ContinuousVariable("Order"),
    ContinuousVariable("X"),
    ContinuousVariable("Y"),
    ContinuousVariable("TrafficLevel"),
    ContinuousVariable("BlockedRoad"),
    ContinuousVariable("Priority"),
    ContinuousVariable("TeamAvailable"),
    ContinuousVariable("FuelCost"),
    ContinuousVariable("RiskLevel"),
    ContinuousVariable("BestRouteCost"),
    ContinuousVariable("BestRouteDistance")
]

meta_vars = [
    StringVariable("Local")
]

domain = Domain(
    continuous_vars,
    metas=meta_vars
)

X_output = result_df[
    [
        "Order",
        "X",
        "Y",
        "TrafficLevel",
        "BlockedRoad",
        "Priority",
        "TeamAvailable",
        "FuelCost",
        "RiskLevel",
        "BestRouteCost",
        "BestRouteDistance"
    ]
].values

metas_output = result_df[["Local"]].astype(str).values

out_data = Table.from_numpy(
    domain,
    X_output,
    metas=metas_output
)


# ============================================================
# 11. Print result in Orange console
# ============================================================

route_names = [df.iloc[index]["Local"] for index in best_route]

print("Best route found:")
print(" -> ".join(route_names))
print("Best route cost:", round(best_cost, 2))
print("Best route distance:", round(best_distance, 2))
