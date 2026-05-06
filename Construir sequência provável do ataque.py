import pandas as pd
import numpy as np
import random

from Orange.data import Table, Domain, ContinuousVariable, StringVariable


# ============================================================
# 1. Converter dados do Orange para DataFrame
# ============================================================

rows = []

for row in in_data:
    record = {}

    # colunas numéricas / atributos
    for var in in_data.domain.attributes:
        record[var.name] = row[var]

    # colunas meta / texto
    for var in in_data.domain.metas:
        record[var.name] = str(row[var])

    # coluna alvo, se existir
    for var in in_data.domain.class_vars:
        record[var.name] = row[var]

    rows.append(record)

df = pd.DataFrame(rows)

print("Columns detected in Orange:")
print(df.columns.tolist())
print("Number of rows:", len(df))


# ============================================================
# 2. Garantir que as colunas existem
# ============================================================

required_columns = [
    "EventID",
    "TimeOrder",
    "Severity",
    "RiskLevel",
    "EvidenceConfidence",
    "Impact",
    "RelatedProbability",
    "TimeGapMinutes",
    "DataVolumeMB"
]

for col in required_columns:
    if col not in df.columns:
        raise Exception("Missing required column: " + col)

text_columns = [
    "EventType",
    "System",
    "SourceIP",
    "DestinationIP",
    "MITREPhase",
    "UserAccount"
]

for col in text_columns:
    if col not in df.columns:
        df[col] = "Unknown"


# ============================================================
# 3. Converter colunas numéricas
# ============================================================

for col in required_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=required_columns).reset_index(drop=True)


# ============================================================
# 4. Função de pontuação forense
# ============================================================

def forensic_score(row):
    score = (
        row["Severity"] * 2
        + row["RiskLevel"] * 2
        + row["EvidenceConfidence"] * 3
        + row["Impact"] * 2
        + row["RelatedProbability"] * 10
        + min(row["DataVolumeMB"] / 100, 5)
        - row["TimeGapMinutes"] * 0.05
    )
    return score


df["ForensicEventScore"] = df.apply(forensic_score, axis=1)


# ============================================================
# 5. Construir sequência provável do ataque
# ============================================================

# Primeiro: ordenar por pontuação forense alta
# Depois: respeitar a ordem temporal
df_sorted = df.sort_values(
    by=["ForensicEventScore", "TimeOrder"],
    ascending=[False, True]
)

# Selecionar os 10 eventos mais relevantes
best_events = df_sorted.head(10).copy()

# Ordenar esses eventos pela sequência temporal
best_events = best_events.sort_values(by="TimeOrder").reset_index(drop=True)

best_events["Order"] = range(1, len(best_events) + 1)

best_sequence_score = best_events["ForensicEventScore"].sum()
best_events["BestSequenceScore"] = round(best_sequence_score, 2)


# ============================================================
# 6. Preparar tabela de saída para o Orange
# ============================================================

numeric_columns = [
    "Order",
    "EventID",
    "TimeOrder",
    "Severity",
    "RiskLevel",
    "EvidenceConfidence",
    "Impact",
    "RelatedProbability",
    "TimeGapMinutes",
    "DataVolumeMB",
    "ForensicEventScore",
    "BestSequenceScore"
]

string_columns = [
    "EventType",
    "System",
    "SourceIP",
    "DestinationIP",
    "MITREPhase",
    "UserAccount"
]

# Garantir que todas existem
for col in string_columns:
    if col not in best_events.columns:
        best_events[col] = "Unknown"

for col in numeric_columns:
    best_events[col] = pd.to_numeric(best_events[col], errors="coerce").fillna(0)


domain = Domain(
    [ContinuousVariable(col) for col in numeric_columns],
    metas=[StringVariable(col) for col in string_columns]
)

X = best_events[numeric_columns].values
metas = best_events[string_columns].astype(str).values

out_data = Table.from_numpy(domain, X, metas=metas)


# ============================================================
# 7. Imprimir resultado na consola
# ============================================================

print("")
print("Most probable forensic attack sequence:")
print("--------------------------------------")

for _, row in best_events.iterrows():
    print(
        str(int(row["Order"])) + ". " +
        str(row["EventType"]) +
        " | System: " + str(row["System"]) +
        " | MITRE Phase: " + str(row["MITREPhase"]) +
        " | Score: " + str(round(row["ForensicEventScore"], 2))
    )

print("--------------------------------------")
print("Best sequence score:", round(best_sequence_score, 2))
