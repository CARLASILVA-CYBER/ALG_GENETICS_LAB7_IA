import pandas as pd
import numpy as np

from Orange.data import Table, Domain, ContinuousVariable, StringVariable


# ============================================================
# 1. Convert Orange input data to pandas DataFrame
# ============================================================

rows = []

for row in in_data:
    record = {}

    # Numeric / attribute columns
    for var in in_data.domain.attributes:
        record[var.name] = row[var]

    # Text / meta columns
    for var in in_data.domain.metas:
        record[var.name] = str(row[var])

    # Class columns, if they exist
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


# Check numeric columns
for col in required_numeric_columns:
    if col not in df.columns:
        raise Exception("Missing required numeric column: " + col)

# If text columns do not exist, create them
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
    """
    Calculates the forensic relevance of each security event.

    Higher score means the event is more relevant to the suspected attack.
    """

    severity = row["Severity"]
    risk = row["RiskLevel"]
    confidence = row["EvidenceConfidence"]
    impact = row["Impact"]
    probability = row["RelatedProbability"]
    data_volume = row["DataVolumeMB"]
    time_gap = row["TimeGapMinutes"]

    score = (
        severity * 2
        + risk * 2
        + confidence * 3
        + impact * 2
        + probability * 10
        + min(data_volume / 100, 5)
        - time_gap * 0.05
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
# 6. Identify most relevant events
# ============================================================

# First: rank events by forensic score
df_ranked = df.sort_values(
    by=["ForensicEventScore", "TimeOrder"],
    ascending=[False, True]
).reset_index(drop=True)

# Select top 10 most relevant events
top_n = 10
best_events = df_ranked.head(top_n).copy()

# Then: order selected events by TimeOrder to create an attack timeline
best_events = best_events.sort_values(by="TimeOrder").reset_index(drop=True)

best_events["Order"] = range(1, len(best_events) + 1)

# Total score for the selected attack sequence
best_sequence_score = best_events["ForensicEventScore"].sum()
best_events["BestSequenceScore"] = round(best_sequence_score, 2)


# ============================================================
# 7. Add interpretation column
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


best_events["ForensicInterpretation"] = best_events.apply(
    lambda row: interpret_event(
        row["EventType"],
        row["MITREPhase"],
        row["Criticality"]
    ),
    axis=1
)


# ============================================================
# 8. Create narrative stage
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


best_events["AttackStage"] = best_events["MITREPhase"].apply(attack_stage)


# ============================================================
# 9. Prepare output columns
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
    "AttackStage",
    "UserAccount",
    "Criticality",
    "ForensicInterpretation"
]

# Ensure columns exist
for col in numeric_columns:
    if col not in best_events.columns:
        best_events[col] = 0
    best_events[col] = pd.to_numeric(best_events[col], errors="coerce").fillna(0)

for col in string_columns:
    if col not in best_events.columns:
        best_events[col] = "Unknown"
    best_events[col] = best_events[col].astype(str)


# ============================================================
# 10. Send output table back to Orange
# ============================================================

domain = Domain(
    [ContinuousVariable(col) for col in numeric_columns],
    metas=[StringVariable(col) for col in string_columns]
)

X = best_events[numeric_columns].values
metas = best_events[string_columns].values

out_data = Table.from_numpy(domain, X, metas=metas)


# ============================================================
# 11. Print forensic report in Orange console
# ============================================================

print("")
print("======================================")
print("MOST PROBABLE FORENSIC ATTACK SEQUENCE")
print("======================================")

for _, row in best_events.iterrows():
    print(
        str(int(row["Order"])) + ". " +
        str(row["EventType"]) +
        " | Stage: " + str(row["AttackStage"]) +
        " | System: " + str(row["System"]) +
        " | Criticality: " + str(row["Criticality"]) +
        " | Score: " + str(round(row["ForensicEventScore"], 2))
    )

print("--------------------------------------")
print("Best sequence score:", round(best_sequence_score, 2))
print("--------------------------------------")

print("")
print("FORENSIC NARRATIVE")
print("--------------------------------------")

event_names = list(best_events["EventType"].astype(str))
stages = list(best_events["AttackStage"].astype(str))

print(
    "The tool identified a probable attack sequence composed of "
    + str(len(best_events))
    + " relevant security events."
)

print(
    "The sequence suggests activity across the following stages: "
    + " -> ".join(stages)
)

print(
    "The most critical events should be validated using original logs, "
    "firewall records, SIEM alerts, EDR/XDR evidence and network traffic captures."
)

print("")
print("IMPORTANT NOTE")
print("--------------------------------------")
print(
    "This result is not absolute proof of the attack. "
    "It is a forensic hypothesis generated to support investigation and evidence prioritisation."
)
