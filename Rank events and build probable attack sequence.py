import pandas as pd
import numpy as np

from Orange.data import Table, Domain, ContinuousVariable, StringVariable


# ============================================================
# 0. SETTINGS — parâmetros que os alunos podem alterar
# ============================================================

TOP_N_EVENTS = 10

WEIGHT_SEVERITY = 2
WEIGHT_RISK = 2
WEIGHT_CONFIDENCE = 3
WEIGHT_IMPACT = 2
WEIGHT_PROBABILITY = 10
WEIGHT_DATA_VOLUME = 1
WEIGHT_TIME_GAP = 0.05

USE_RANDOM_UNCERTAINTY = True
UNCERTAINTY_LEVEL = 3


# ============================================================
# 1. Convert Orange input data to pandas DataFrame
# ============================================================

rows = []

for row in in_data:
    record = {}

    for var in in_data.domain.attributes:
        record[var.name] = row[var]

    for var in in_data.domain.metas:
        record[var.name] = str(row[var])

    for var in in_data.domain.class_vars:
        record[var.name] = row[var]

    rows.append(record)

df = pd.DataFrame(rows)

print("======================================")
print("FORENSIC AI TOOL - DATA CHECK")
print("======================================")
print("Columns detected in Orange:")
print(df.columns.tolist())
print("Number of events:", len(df))


# ============================================================
# 2. Required columns
# ============================================================

required_numeric_columns = [
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

required_text_columns = [
    "EventType",
    "System",
    "SourceIP",
    "DestinationIP",
    "MITREPhase",
    "UserAccount"
]

for col in required_numeric_columns:
    if col not in df.columns:
        raise Exception("Missing required numeric column: " + col)

for col in required_text_columns:
    if col not in df.columns:
        df[col] = "Unknown"


# ============================================================
# 3. Clean and convert data
# ============================================================

for col in required_numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=required_numeric_columns).reset_index(drop=True)

for col in required_text_columns:
    df[col] = df[col].astype(str)


# ============================================================
# 4. Forensic scoring function
# ============================================================

def forensic_score(row):
    severity = row["Severity"]
    risk = row["RiskLevel"]
    confidence = row["EvidenceConfidence"]
    impact = row["Impact"]
    probability = row["RelatedProbability"]
    data_volume = row["DataVolumeMB"]
    time_gap = row["TimeGapMinutes"]

    score = (
        severity * WEIGHT_SEVERITY
        + risk * WEIGHT_RISK
        + confidence * WEIGHT_CONFIDENCE
        + impact * WEIGHT_IMPACT
        + probability * WEIGHT_PROBABILITY
        + min(data_volume / 100, 5) * WEIGHT_DATA_VOLUME
        - time_gap * WEIGHT_TIME_GAP
    )

    if USE_RANDOM_UNCERTAINTY:
        score = score + np.random.uniform(
            -UNCERTAINTY_LEVEL,
            UNCERTAINTY_LEVEL
        )

    return score


df["ForensicEventScore"] = df.apply(forensic_score, axis=1)


# ============================================================
# 5. Criticality classification
# ============================================================

def classify_criticality(score):
    if score >= 55:
        return "Critical"
    elif score >= 45:
        return "High"
    elif score >= 30:
        return "Medium"
    else:
        return "Low"


df["Criticality"] = df["ForensicEventScore"].apply(classify_criticality)


# ============================================================
# 6. Attack stage interpretation
# ============================================================

def attack_stage(mitre_phase):
    mitre_phase = str(mitre_phase)

    if "Initial" in mitre_phase:
        return "Initial Access"
    elif "Execution" in mitre_phase:
        return "Execution"
    elif "Privilege" in mitre_phase:
        return "Privilege Escalation"
    elif "Defense" in mitre_phase:
        return "Defense Evasion"
    elif "Credential" in mitre_phase:
        return "Credential Access"
    elif "Lateral" in mitre_phase:
        return "Lateral Movement"
    elif "Collection" in mitre_phase:
        return "Collection"
    elif "Command" in mitre_phase:
        return "Command and Control"
    elif "Exfiltration" in mitre_phase:
        return "Exfiltration"
    else:
        return "Other"


df["AttackStage"] = df["MITREPhase"].apply(attack_stage)


# ============================================================
# 7. Forensic interpretation
# ============================================================

def interpret_event(event_type, mitre_phase, criticality):
    event_type = str(event_type)
    mitre_phase = str(mitre_phase)

    if "Exfiltration" in mitre_phase or "Exfiltration" in event_type:
        return "Possible data exfiltration event"
    elif "Credential" in mitre_phase or "Credential" in event_type:
        return "Possible credential compromise"
    elif "Defense_Evasion" in mitre_phase or "Log_Cleared" in event_type:
        return "Possible attempt to hide traces"
    elif "Privilege" in mitre_phase or "Admin" in event_type:
        return "Possible privilege escalation"
    elif "Lateral" in mitre_phase or "Remote_Service" in event_type:
        return "Possible lateral movement"
    elif "Collection" in mitre_phase or "Query" in event_type or "File" in event_type:
        return "Possible data collection activity"
    elif "Command_Control" in mitre_phase or "External" in event_type:
        return "Possible command and control communication"
    else:
        return "Relevant security event requiring validation"


df["ForensicInterpretation"] = df.apply(
    lambda row: interpret_event(
        row["EventType"],
        row["MITREPhase"],
        row["Criticality"]
    ),
    axis=1
)


# ============================================================
# 8. Rank events and build probable attack sequence
# ============================================================

df_ranked = df.sort_values(
    by=["ForensicEventScore", "TimeOrder"],
    ascending=[False, True]
).reset_index(drop=True)

best_events = df_ranked.head(TOP_N_EVENTS).copy()

best_events = best_events.sort_values(by="TimeOrder").reset_index(drop=True)

best_events["Order"] = range(1, len(best_events) + 1)

best_sequence_score = best_events["ForensicEventScore"].sum()
best_events["BestSequenceScore"] = round(best_sequence_score, 2)


# ============================================================
# 9. Investigation priority
# ============================================================

def investigation_priority(row):
    if row["Criticality"] == "Critical":
        return "Investigate immediately"
    elif row["Criticality"] == "High":
        return "Investigate with high priority"
    elif row["Criticality"] == "Medium":
        return "Review if related to other evidence"
    else:
        return "Low priority review"


best_events["InvestigationPriority"] = best_events.apply(
    investigation_priority,
    axis=1
)


# ============================================================
# 10. Create visual attack path columns
# ============================================================

best_events["SequenceStep"] = best_events.apply(
    lambda row:
        str(int(row["Order"])) + " → " +
        str(row["EventType"]) + " → " +
        str(row["AttackStage"]) + " → " +
        str(row["Criticality"]),
    axis=1
)

best_events["SequenceStepDetailed"] = best_events.apply(
    lambda row:
        "Step " + str(int(row["Order"])) +
        " | Event: " + str(row["EventType"]) +
        " | Stage: " + str(row["AttackStage"]) +
        " | System: " + str(row["System"]) +
        " | Score: " + str(round(row["ForensicEventScore"], 2)) +
        " | Priority: " + str(row["InvestigationPriority"]),
    axis=1
)

attack_path = " → ".join(best_events["EventType"].astype(str))

attack_stage_path = " → ".join(best_events["AttackStage"].astype(str))

best_events["AttackPath"] = attack_path
best_events["AttackStagePath"] = attack_stage_path


# ============================================================
# 11. Prepare output columns
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
    "SequenceStep",
    "SequenceStepDetailed",
    "AttackPath",
    "AttackStagePath",
    "EventType",
    "System",
    "SourceIP",
    "DestinationIP",
    "MITREPhase",
    "AttackStage",
    "UserAccount",
    "Criticality",
    "ForensicInterpretation",
    "InvestigationPriority"
]

for col in numeric_columns:
    if col not in best_events.columns:
        best_events[col] = 0
    best_events[col] = pd.to_numeric(best_events[col], errors="coerce").fillna(0)

for col in string_columns:
    if col not in best_events.columns:
        best_events[col] = "Unknown"
    best_events[col] = best_events[col].astype(str)


# ============================================================
# 12. Send output table back to Orange
# ============================================================

domain = Domain(
    [ContinuousVariable(col) for col in numeric_columns],
    metas=[StringVariable(col) for col in string_columns]
)

X = best_events[numeric_columns].values
metas = best_events[string_columns].values

out_data = Table.from_numpy(domain, X, metas=metas)


# ============================================================
# 13. Print forensic report in Orange console
# ============================================================

print("")
print("======================================")
print("FORENSIC AI TOOL - SETTINGS")
print("======================================")
print("TOP_N_EVENTS:", TOP_N_EVENTS)
print("USE_RANDOM_UNCERTAINTY:", USE_RANDOM_UNCERTAINTY)
print("UNCERTAINTY_LEVEL:", UNCERTAINTY_LEVEL)

print("")
print("======================================")
print("MOST PROBABLE FORENSIC ATTACK SEQUENCE")
print("======================================")

for _, row in best_events.iterrows():
    print(str(row["SequenceStepDetailed"]))

print("--------------------------------------")
print("Best sequence score:", round(best_sequence_score, 2))
print("--------------------------------------")

print("")
print("ATTACK PATH")
print("--------------------------------------")
print(attack_path)

print("")
print("ATTACK STAGE PATH")
print("--------------------------------------")
print(attack_stage_path)

print("")
print("IMPORTANT NOTE")
print("--------------------------------------")
print(
    "This result is not absolute proof of the attack. "
    "It is a forensic hypothesis generated to support investigation and evidence prioritisation."
)
