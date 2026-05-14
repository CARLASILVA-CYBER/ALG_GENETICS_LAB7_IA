import random
import pandas as pd

from Orange.data import Table, Domain, StringVariable, ContinuousVariable

# ==============================
# 1. Ler dados recebidos do Orange
# ==============================

df_num = pd.DataFrame(
    in_data.X,
    columns=[a.name for a in in_data.domain.attributes]
)

df = df_num.copy()

if len(in_data.domain.metas) > 0:
    df_meta = pd.DataFrame(
        in_data.metas,
        columns=[m.name for m in in_data.domain.metas]
    )
    df = pd.concat([df_meta, df_num], axis=1)

# Se o Campo não aparecer, criar automaticamente
if "Campo" not in df.columns:
    df.insert(0, "Campo", ["C" + str(i + 1) for i in range(len(df))])

# Garantir tipos
df["Campo"] = df["Campo"].astype(str)
df["Custo"] = df["Custo"].astype(float)
df["Risco"] = df["Risco"].astype(float)
df["Prioridade"] = df["Prioridade"].astype(float)
df["X"] = df["X"].astype(float)
df["Y"] = df["Y"].astype(float)

campos = df.to_dict("records")

# ==============================
# 2. Parâmetros
# ==============================

MAX_RISK = 20
POP_SIZE = 100
GENERATIONS = 200
MUTATION_RATE = 0.25

# ==============================
# 3. Funções
# ==============================

def total_cost(route):
    return sum(c["Custo"] for c in route)

def total_risk(route):
    return sum(c["Risco"] for c in route)

def priority_bonus(route):
    bonus = 0

    for c in route:
        if c["Prioridade"] == 1:
            bonus += 30
        elif c["Prioridade"] == 2:
            bonus += 15
        else:
            bonus += 3

    return bonus

def priority_penalty(route):
    penalty = 0

    for i in range(len(route)):
        for j in range(i + 1, len(route)):
            if route[i]["Prioridade"] > route[j]["Prioridade"]:
                penalty += 20

    return penalty

def fitness(route):
    cost = total_cost(route)
    risk = total_risk(route)
    penalty = priority_penalty(route)
    bonus = priority_bonus(route)

    if risk > MAX_RISK:
        penalty += 1000 + ((risk - MAX_RISK) * 100)

    return cost + penalty - bonus

def create_individual():
    route = campos.copy()
    random.shuffle(route)

    selected = []
    risk = 0

    for c in route:
        if risk + c["Risco"] <= MAX_RISK:
            selected.append(c)
            risk += c["Risco"]

    return selected

def crossover(parent1, parent2):
    child = []

    for c in parent1 + parent2:
        exists = False

        for existing in child:
            if existing["Campo"] == c["Campo"]:
                exists = True

        if not exists:
            child.append(c)

    valid_child = []
    risk = 0

    for c in child:
        if risk + c["Risco"] <= MAX_RISK:
            valid_child.append(c)
            risk += c["Risco"]

    return valid_child

def mutate(route):
    new_route = route.copy()

    if len(new_route) > 1 and random.random() < MUTATION_RATE:
        i, j = random.sample(range(len(new_route)), 2)
        new_route[i], new_route[j] = new_route[j], new_route[i]

    return new_route

# ==============================
# 4. Algoritmo genético
# ==============================

population = [create_individual() for _ in range(POP_SIZE)]

for generation in range(GENERATIONS):

    population = sorted(population, key=fitness)

    best_half = population[:POP_SIZE // 2]
    new_population = best_half.copy()

    while len(new_population) < POP_SIZE:
        parent1, parent2 = random.sample(best_half, 2)

        child = crossover(parent1, parent2)
        child = mutate(child)

        new_population.append(child)

    population = new_population

best_route = sorted(population, key=fitness)[0]

# Ordenar para visualização
best_route = sorted(best_route, key=lambda x: (x["Prioridade"], x["Custo"]))

# ==============================
# 5. Criar resultado final
# ==============================

custo_total = total_cost(best_route)
risco_total = total_risk(best_route)
fitness_final = fitness(best_route)

result = []

for ordem, c in enumerate(best_route, start=1):
    result.append([
        ordem,
        c["Campo"],
        c["Custo"],
        c["Risco"],
        c["Prioridade"],
        c["X"],
        c["Y"],
        custo_total,
        risco_total,
        fitness_final
    ])

result_df = pd.DataFrame(result, columns=[
    "Ordem",
    "Campo",
    "Custo",
    "Risco",
    "Prioridade",
    "X",
    "Y",
    "Custo_Total_Rota",
    "Risco_Total_Rota",
    "Fitness"
])

print("ROTA OTIMIZADA")
print(result_df)

# ==============================
# 6. Enviar resultado para o Orange
# ==============================

domain = Domain(
    [
        ContinuousVariable("Ordem"),
        ContinuousVariable("Custo"),
        ContinuousVariable("Risco"),
        ContinuousVariable("Prioridade"),
        ContinuousVariable("X"),
        ContinuousVariable("Y"),
        ContinuousVariable("Custo_Total_Rota"),
        ContinuousVariable("Risco_Total_Rota"),
        ContinuousVariable("Fitness")
    ],
    metas=[
        StringVariable("Campo")
    ]
)

X_result = result_df[[
    "Ordem",
    "Custo",
    "Risco",
    "Prioridade",
    "X",
    "Y",
    "Custo_Total_Rota",
    "Risco_Total_Rota",
    "Fitness"
]].values

metas_result = result_df[["Campo"]].astype(str).values

out_data = Table.from_numpy(domain, X_result, metas=metas_result)
