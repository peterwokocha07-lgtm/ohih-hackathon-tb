
# app.py — OHIH TB Platform (Objectives 1–5) + A/B/C/D + Demo Data Generator
# Requirements (inside your .venv):
#   pip install streamlit pandas numpy pillow
# Optional heatmap:
#   pip install folium streamlit-folium

import os
import csv
import math
import random
import statistics
import datetime as dt
import subprocess
import tempfile
from dataclasses import dataclass

import pandas as pd
import streamlit as st

# A) Microscopy AI demo
import numpy as np
from PIL import Image

# Optional mapping (Obj 4 heatmap)
try:
    import folium
    from folium.plugins import HeatMap
    from streamlit_folium import st_folium
    HAS_MAP = True
except Exception:
    HAS_MAP = False


# =========================================================
# Storage (data folder inside your project)
# =========================================================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")

EVENTS_CSV = os.path.join(DATA_DIR, "events.csv")
DOCKING_QUEUE_CSV = os.path.join(DATA_DIR, "docking_queue.csv")
ADHERENCE_CSV = os.path.join(DATA_DIR, "adherence.csv")
DOTS_CSV = os.path.join(DATA_DIR, "dots_daily.csv")

EVENT_FIELDS = ["timestamp", "state", "lga_or_area", "lat", "lon", "tb_probability", "category", "gene_xpert"]
DOCKING_FIELDS = ["timestamp", "target", "drug_name", "drug_id", "smiles", "notes"]

ADH_FIELDS = [
    "timestamp",
    "patient_code",
    "treatment_start_date",
    "planned_duration_months",
    "weeks_on_treatment",
    "missed_days_last_7",
    "missed_days_last_28",
    "longest_missed_streak",
    "completed_regimen",
    "adherence_28d_percent",
    "flag_high_risk_25pct_missed",
    "cumulative_expected_days",
    "cumulative_missed_days",
    "cumulative_adherence_percent",
    "predicted_risk_score",
    "predicted_risk_category",
    "note"
]

DOTS_FIELDS = ["date", "patient_code", "dose_taken", "note"]


def ensure_storage():
    os.makedirs(DATA_DIR, exist_ok=True)

    def init_csv(path, fields):
        if not os.path.exists(path):
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fields).writeheader()

    init_csv(EVENTS_CSV, EVENT_FIELDS)
    init_csv(DOCKING_QUEUE_CSV, DOCKING_FIELDS)
    init_csv(ADHERENCE_CSV, ADH_FIELDS)
    init_csv(DOTS_CSV, DOTS_FIELDS)


def append_row(path: str, fields: list[str], row: dict):
    ensure_storage()
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writerow(row)


def load_csv(path: str) -> pd.DataFrame:
    ensure_storage()
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def save_event(row: dict):
    append_row(EVENTS_CSV, EVENT_FIELDS, row)


def load_events() -> pd.DataFrame:
    df = load_csv(EVENTS_CSV)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["lat"] = pd.to_numeric(df.get("lat"), errors="coerce")
    df["lon"] = pd.to_numeric(df.get("lon"), errors="coerce")
    df["tb_probability"] = pd.to_numeric(df.get("tb_probability"), errors="coerce")
    return df.dropna(subset=["timestamp"])


def save_to_docking_queue(rows: list[dict]):
    ensure_storage()
    with open(DOCKING_QUEUE_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=DOCKING_FIELDS)
        for r in rows:
            w.writerow(r)


def load_docking_queue() -> pd.DataFrame:
    df = load_csv(DOCKING_QUEUE_CSV)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df.dropna(subset=["timestamp"])


def save_adherence(row: dict):
    append_row(ADHERENCE_CSV, ADH_FIELDS, row)


def load_adherence() -> pd.DataFrame:
    df = load_csv(ADHERENCE_CSV)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["treatment_start_date"] = pd.to_datetime(df["treatment_start_date"], errors="coerce")

    for c in [
        "planned_duration_months", "weeks_on_treatment", "missed_days_last_7", "missed_days_last_28",
        "adherence_28d_percent", "cumulative_expected_days", "cumulative_missed_days",
        "cumulative_adherence_percent", "predicted_risk_score"
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.dropna(subset=["timestamp"])


def upsert_dots_record(date_iso: str, patient_code: str, dose_taken: bool, note: str):
    ensure_storage()
    df = load_csv(DOTS_CSV)
    if df.empty:
        df = pd.DataFrame(columns=DOTS_FIELDS)

    new_row = {"date": date_iso, "patient_code": patient_code, "dose_taken": bool(dose_taken), "note": note}

    if df.empty:
        df = pd.DataFrame([new_row])
    else:
        mask = (df["date"].astype(str) == str(date_iso)) & (df["patient_code"].astype(str) == str(patient_code))
        if mask.any():
            df.loc[mask, ["dose_taken", "note"]] = [bool(dose_taken), note]
        else:
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_csv(DOTS_CSV, index=False)


def load_dots() -> pd.DataFrame:
    df = load_csv(DOTS_CSV)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["dose_taken"] = df["dose_taken"].astype(str).str.lower().isin(["true", "1", "yes"])
    return df.dropna(subset=["date"])


# =========================================================
# Objective 2: Digital Diagnosis (simple explainable model)
# =========================================================
DEFAULT_PRETEST = {"Community screening": 0.08, "Hospital triage": 0.25}

TESTS = {
    "GeneXpert": {"sens": 0.85, "spec": 0.98},
    "Microscopy": {"sens": 0.60, "spec": 0.98},
    "CXR": {"sens": 0.85, "spec": 0.70},
}

SYMPTOMS = [
    ("Cough > 2 weeks", 0.06),
    ("Fever", 0.03),
    ("Night sweats", 0.05),
    ("Weight loss", 0.06),
    ("Hemoptysis", 0.08),
    ("Close TB contact", 0.10),
]
RISK_FACTORS = [
    ("HIV positive", 0.12),
    ("Diabetes", 0.04),
    ("Malnutrition", 0.05),
    ("Smoking", 0.03),
    ("Previous TB", 0.06),
]


def prob_to_odds(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return p / (1 - p)


def odds_to_prob(o: float) -> float:
    return o / (1 + o)


def lr_positive(sens: float, spec: float) -> float:
    return sens / (1 - spec)


def lr_negative(sens: float, spec: float) -> float:
    return (1 - sens) / spec


def apply_lrs(pretest_prob: float, lrs: list[float]) -> float:
    odds = prob_to_odds(pretest_prob)
    for lr in lrs:
        odds *= lr
    return odds_to_prob(odds)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def symptom_risk_boost(symptoms, risks, cap=0.20):
    boost = 0.0
    for name, w in SYMPTOMS:
        if name in symptoms:
            boost += w
    for name, w in RISK_FACTORS:
        if name in risks:
            boost += w
    return min(boost, cap)


def decision_support(prob, gx_result, danger):
    if danger:
        return "URGENT", "Red flags present. Urgent clinical review required. Stabilize and expedite TB evaluation."
    if gx_result == "Detected":
        return "CONFIRMED TB", "GeneXpert detected MTB. Start TB pathway per guideline and assess drug resistance."
    if prob >= 0.60:
        return "HIGH probability", "High likelihood of TB. Arrange urgent GeneXpert/CXR; consider isolation per protocol."
    if prob >= 0.30:
        return "MODERATE probability", "Moderate likelihood. Do GeneXpert/CXR (if not done) and reassess."
    if prob >= 0.10:
        return "LOW–MODERATE probability", "Some features present. Targeted testing advised if symptoms persist."
    return "LOW probability", "TB less likely. Treat alternatives and safety-net follow-up."


# =========================================================
# Objective 1: Repurposing (demo shortlist)
# =========================================================
REPURPOSE_DB = [
    {"target": "InhA", "drug_name": "Isoniazid (control)", "drug_id": "DB00951", "smiles": "NNC(=O)c1ccncc1", "notes": "First-line TB drug (control)"},
    {"target": "DNA gyrase (GyrA/B)", "drug_name": "Levofloxacin", "drug_id": "DB01137", "smiles": "N/A", "notes": "Used in TB regimens"},
    {"target": "DNA gyrase (GyrA/B)", "drug_name": "Moxifloxacin", "drug_id": "DB00218", "smiles": "N/A", "notes": "Used in TB regimens"},
    {"target": "ATP synthase", "drug_name": "Bedaquiline", "drug_id": "DB08904", "smiles": "N/A", "notes": "Approved for MDR-TB"},
    {"target": "DprE1", "drug_name": "BTZ043 (investigational)", "drug_id": "N/A", "smiles": "N/A", "notes": "Demo placeholder"},
]


def repurpose_candidates_for_target(target: str) -> pd.DataFrame:
    df = pd.DataFrame(REPURPOSE_DB)
    return df[df["target"] == target].reset_index(drop=True)


# =========================================================
# Objective 4: Outbreak detection (simple alerts)
# =========================================================
def compute_alerts(df: pd.DataFrame, days_window: int = 7, min_cases: int = 3, ratio_threshold: float = 2.0) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["area", "recent", "previous", "ratio", "alert"])

    now = df["timestamp"].max()
    recent_start = now - pd.Timedelta(days=days_window)
    prev_start = now - pd.Timedelta(days=2 * days_window)
    prev_end = recent_start

    d = df.copy()
    d["area"] = (d["state"].fillna("") + " | " + d["lga_or_area"].fillna("")).str.strip()

    recent = d[(d["timestamp"] >= recent_start) & (d["timestamp"] <= now)].groupby("area").size()
    previous = d[(d["timestamp"] >= prev_start) & (d["timestamp"] < prev_end)].groupby("area").size()

    rows = []
    for a in sorted(set(recent.index).union(set(previous.index))):
        r = int(recent.get(a, 0))
        p = int(previous.get(a, 0))
        ratio = (r / max(p, 1)) if p > 0 else (float("inf") if r > 0 else 0.0)
        alert = (r >= min_cases) and (ratio >= ratio_threshold)
        rows.append({"area": a, "recent": r, "previous": p, "ratio": ratio, "alert": alert})

    return pd.DataFrame(rows).sort_values(["alert", "recent", "ratio"], ascending=[False, False, False])


# =========================================================
# Objective 3: Docking helpers (Vina v1.2.3 compatible) + Interpretation
# =========================================================
def compute_receptor_center_from_pdbqt_text(pdbqt_text: str) -> tuple[float, float, float]:
    xs, ys, zs = [], [], []
    for line in pdbqt_text.splitlines():
        if line.startswith(("ATOM", "HETATM")) and len(line) >= 54:
            try:
                xs.append(float(line[30:38]))
                ys.append(float(line[38:46]))
                zs.append(float(line[46:54]))
            except Exception:
                pass
    if not xs:
        return 0.0, 0.0, 0.0
    return float(statistics.mean(xs)), float(statistics.mean(ys)), float(statistics.mean(zs))


@dataclass
class VinaPose:
    mode: int
    affinity: float
    rmsd_lb: float | None
    rmsd_ub: float | None


def parse_vina_table(vina_output: str) -> list[VinaPose]:
    poses: list[VinaPose] = []
    for line in vina_output.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0].isdigit():
            try:
                mode = int(parts[0])
                affinity = float(parts[1])
                rmsd_lb = float(parts[2]) if len(parts) >= 3 else None
                rmsd_ub = float(parts[3]) if len(parts) >= 4 else None
                poses.append(VinaPose(mode, affinity, rmsd_lb, rmsd_ub))
            except Exception:
                pass
    return poses


def interpret_affinity(aff: float) -> tuple[str, str]:
    if aff <= -9.0:
        return "Strong binding (very promising)", "Often indicates good fit; confirm with controls, rescoring, and wet-lab validation."
    if aff <= -7.5:
        return "Moderate–strong binding", "Worth prioritizing for further docking/MD simulations and shortlist."
    if aff <= -6.0:
        return "Moderate binding", "May still be useful depending on target, ligand size, and controls."
    return "Weak binding", "Usually lower priority unless supported by other evidence."


# =========================================================
# A) Microscopy AI (demo computer vision)
# =========================================================
def microscopy_demo_score(img: Image.Image) -> dict:
    g = img.convert("L")
    arr = np.array(g).astype(np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)

    contrast = float(arr.std())

    thr = 0.25
    small = g.resize((256, 256))
    a2 = np.array(small).astype(np.float32)
    a2 = (a2 - a2.min()) / (a2.max() - a2.min() + 1e-6)
    dark2 = (a2 < thr).astype(np.uint8)
    dark_density2 = float(dark2.mean())

    raw = 0.15 + 1.5 * dark_density2 + 0.9 * contrast
    prob = max(0.0, min(1.0, raw))

    label = "Likely AFB negative / low bacilli burden"
    if prob >= 0.75:
        label = "Likely AFB positive (high suspicion)"
    elif prob >= 0.50:
        label = "Indeterminate / moderate suspicion"

    explanation = (
        f"Contrast={contrast:.3f}, dark-density={dark_density2:.3f}. "
        "Higher dark-density + higher contrast increases the AI suspicion score."
    )

    return {"prob": prob, "label": label, "contrast": contrast, "dark_density": dark_density2, "explain": explanation}


# =========================================================
# B) Adherence + DOTS daily + risk prediction
# =========================================================
STREAK_OPTIONS = ["0 days", "1–2 days", "3–6 days", "1 week", "2 weeks", "3 weeks", "1 month or more"]


def streak_weight(streak: str) -> float:
    return {
        "0 days": 0.0,
        "1–2 days": 0.5,
        "3–6 days": 1.0,
        "1 week": 1.6,
        "2 weeks": 2.2,
        "3 weeks": 2.8,
        "1 month or more": 3.5,
    }.get(streak, 0.0)


def compute_adherence_28d_percent(missed_days_last_28: int) -> float:
    missed_days_last_28 = int(max(0, min(28, missed_days_last_28)))
    taken = 28 - missed_days_last_28
    return (taken / 28.0) * 100.0


def missed_over_25pct(missed_days_last_28: int) -> bool:
    return int(missed_days_last_28) > 7  # >25% of 28 days


def weeks_since(start_date: dt.date) -> int:
    delta = dt.date.today() - start_date
    return max(0, int(delta.days // 7))


def compute_recent_missed_from_dots(patient_code: str, days: int) -> int:
    df = load_dots()
    if df.empty:
        return days
    end = dt.date.today()
    start = end - dt.timedelta(days=days - 1)
    dfp = df[df["patient_code"].astype(str) == str(patient_code)].copy()
    dfp = dfp[(dfp["date"] >= start) & (dfp["date"] <= end)]
    taken = int(dfp["dose_taken"].sum()) if not dfp.empty else 0
    return max(0, days - taken)


def compute_cumulative_from_dots(patient_code: str, start_date: dt.date) -> tuple[int, int, float]:
    if start_date > dt.date.today():
        return 0, 0, 0.0
    expected_days = (dt.date.today() - start_date).days + 1

    df = load_dots()
    if df.empty:
        missed = expected_days
        pct = 0.0 if expected_days else 0.0
        return expected_days, missed, pct

    dfp = df[df["patient_code"].astype(str) == str(patient_code)].copy()
    dfp = dfp[(dfp["date"] >= start_date) & (dfp["date"] <= dt.date.today())]
    taken_days = int(dfp["dose_taken"].sum()) if not dfp.empty else 0
    missed_days = max(0, expected_days - taken_days)
    adherence_pct = (taken_days / expected_days) * 100.0 if expected_days else 0.0
    return expected_days, missed_days, adherence_pct


def predict_adherence_risk(missed_28: int, streak: str, cumulative_adherence_pct: float, completed: bool) -> tuple[float, str, str]:
    if completed:
        return 0.05, "Completed", "Regimen marked completed. Still ensure follow-up as per TB program."

    m28 = max(0, min(28, int(missed_28)))
    streak_w = streak_weight(streak)
    cum = max(0.0, min(100.0, float(cumulative_adherence_pct)))

    x_missed = m28 / 28.0
    x_streak = min(1.0, streak_w / 3.5)
    x_cum = 1.0 - (cum / 100.0)

    z = 2.2 * x_missed + 1.4 * x_streak + 1.6 * x_cum - 0.8
    risk = 1.0 / (1.0 + math.exp(-z))

    if risk >= 0.80:
        cat = "Very high risk"
        msg = "High risk of treatment failure / resistance. Intensify adherence support and urgent program review."
    elif risk >= 0.60:
        cat = "High risk"
        msg = "Significant interruption risk. Strengthen DOTS support; reassess clinically and with bacteriology."
    elif risk >= 0.35:
        cat = "Moderate risk"
        msg = "Moderate risk. Reinforce counseling, check barriers, follow-up closely."
    else:
        cat = "Low risk"
        msg = "Low predicted risk currently. Continue routine adherence support."

    return float(risk), cat, msg


# =========================================================
# D) Reports & Pitch
# =========================================================
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def make_clinical_report_text(
    patient_code: str,
    start_date: dt.date,
    planned_months: int,
    missed7: int,
    missed28: int,
    streak: str,
    adh28: float,
    cum_expected: int,
    cum_missed: int,
    cum_pct: float,
    risk: float,
    risk_cat: str,
    completed: bool,
) -> str:
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append("OHIH TB Platform — Patient Adherence Report")
    lines.append(f"Generated: {stamp}")
    lines.append("")
    lines.append(f"Patient code: {patient_code}")
    lines.append(f"Treatment start: {start_date.isoformat()}")
    lines.append(f"Planned duration: {planned_months} months")
    lines.append(f"Regimen completed: {'YES' if completed else 'NO'}")
    lines.append("")
    lines.append("Recent adherence")
    lines.append(f"- Missed doses (last 7 days): {missed7}")
    lines.append(f"- Missed doses (last 28 days): {missed28}")
    lines.append(f"- 28-day adherence: {adh28:.1f}%")
    lines.append(f"- Longest missed streak: {streak}")
    lines.append("")
    lines.append("Cumulative adherence")
    lines.append(f"- Expected days since start: {cum_expected}")
    lines.append(f"- Cumulative missed days: {cum_missed}")
    lines.append(f"- Cumulative adherence: {cum_pct:.1f}%")
    lines.append("")
    lines.append("Risk prediction (demo)")
    lines.append(f"- Predicted risk score: {risk:.2f}")
    lines.append(f"- Category: {risk_cat}")
    if missed_over_25pct(missed28):
        lines.append("- Flag: missed >25% of doses in last 28 days (programmatic high-risk rule).")
    lines.append("")
    lines.append("Recommended actions (general)")
    if completed:
        lines.append("- Ensure follow-up as per TB program (clinical and bacteriologic if indicated).")
    else:
        if risk >= 0.60:
            lines.append("- Intensify DOTS/adherence support immediately; address barriers (cost, transport, stigma, ADRs).")
            lines.append("- Consider urgent clinical review + bacteriology per TB program guidance.")
        else:
            lines.append("- Continue adherence support; safety-net instructions; follow-up scheduled.")
    lines.append("")
    lines.append("Note: Educational demo report. Clinical decisions should follow national TB guidelines.")
    return "\n".join(lines)


def make_pitch_script() -> str:
    return """HACKATHON PITCH (10–12 minutes)

1) Problem (1 min)
Tuberculosis remains a major cause of death. Key gaps:
- delayed diagnosis,
- treatment non-adherence leading to resistance,
- weak outbreak detection,
- slow drug discovery.

2) Solution (1 min)
OHIH TB Platform: an end-to-end AI-assisted TB control system:
Diagnosis AI → Adherence/DOTS tracking → Outbreak analytics → Drug repurposing + docking.

3) Demo Flow (6–7 min)
A) Digital Diagnosis:
- Enter symptoms + test results (GeneXpert/microscopy/CXR).
- System outputs probability and clinical decision support.

B) Microscopy AI (demo):
- Upload a smartphone-captured microscopy image.
- System generates an AI suspicion score and “likely AFB+/AFB−”.

C) Treatment Adherence + DOTS:
- Tick today’s dose.
- System computes missed doses (7d/28d) + cumulative adherence.
- Predicts risk of failure/resistance and flags >25% missed.

D) Outbreak analytics:
- Saved anonymized cases produce trend charts, alerts, and heatmap.

E) Drug repurposing + docking:
- Pick target, shortlist drugs, send to docking.
- Run Vina docking (real) and interpret affinity automatically.

4) Impact (1 min)
- Earlier detection, better triage.
- Reduced default rate and resistance risk through DOTS + risk prediction.
- Faster candidate prioritization via repurposing + docking.
- Public-health visibility via outbreak alerts.

5) Why this is scalable (1 min)
- Offline-friendly data capture.
- Modular: can expand to HIV/malaria.
- Uses open tools and explainable AI methods.

6) Ask (30 sec)
We’re seeking support for data partnerships, validation with TB programs, and pilot deployment.
"""


# =========================================================
# Demo Data Generator (NEW)
# =========================================================
def _rand_point(center_lat, center_lon, spread=0.15):
    return (
        center_lat + random.uniform(-spread, spread),
        center_lon + random.uniform(-spread, spread),
    )


def generate_demo_data(
    n_events=80,
    days=30,
    spike_area="Rivers | Port Harcourt",
    spike_multiplier=3,
    include_dots=True,
    include_adherence=True,
    include_docking=True,
):
    ensure_storage()

    areas = [
        {"state": "Rivers", "lga": "Port Harcourt", "lat": 4.8156, "lon": 7.0498},
        {"state": "Rivers", "lga": "Obio/Akpor", "lat": 4.8400, "lon": 6.9700},
        {"state": "Rivers", "lga": "Eleme", "lat": 4.7900, "lon": 7.1200},
        {"state": "Lagos", "lga": "Ikeja", "lat": 6.6018, "lon": 3.3515},
        {"state": "Abuja (FCT)", "lga": "Garki", "lat": 9.0333, "lon": 7.5333},
        {"state": "Kano", "lga": "Municipal", "lat": 12.0000, "lon": 8.5167},
    ]

    spike_state, spike_lga = [x.strip() for x in spike_area.split("|")]
    spike_candidates = [a for a in areas if a["state"] == spike_state and a["lga"] == spike_lga]
    if not spike_candidates:
        spike_candidates = [areas[0]]

    now = dt.datetime.now()

    # Events
    for _ in range(int(n_events)):
        day_offset = int(random.triangular(0, days - 1, 3))  # bias toward recent
        ts = now - dt.timedelta(days=day_offset, hours=random.randint(0, 23), minutes=random.randint(0, 59))

        choose_spike = (day_offset <= 7) and (random.random() < min(0.85, 0.25 * spike_multiplier))
        area = random.choice(spike_candidates if choose_spike else areas)
        lat, lon = _rand_point(area["lat"], area["lon"], spread=0.12)

        base_prob = random.betavariate(2, 5)  # mostly low/mod
        if choose_spike:
            base_prob = min(1.0, base_prob + random.uniform(0.25, 0.55))

        gx = random.choices(["Not done", "Detected", "Not detected"], weights=[0.5, 0.25, 0.25])[0]
        category = "LOW probability"
        if gx == "Detected":
            category = "CONFIRMED TB"
            base_prob = max(base_prob, random.uniform(0.70, 0.95))
        elif base_prob >= 0.60:
            category = "HIGH probability"
        elif base_prob >= 0.30:
            category = "MODERATE probability"
        elif base_prob >= 0.10:
            category = "LOW–MODERATE probability"

        save_event({
            "timestamp": ts.isoformat(timespec="seconds"),
            "state": area["state"],
            "lga_or_area": area["lga"],
            "lat": float(lat),
            "lon": float(lon),
            "tb_probability": float(base_prob),
            "category": category,
            "gene_xpert": gx,
        })

    # DOTS
    if include_dots:
        patients = ["PT-001", "PT-002", "PT-003", "PT-004", "PT-005"]
        starts = {
            "PT-001": dt.date.today() - dt.timedelta(days=45),
            "PT-002": dt.date.today() - dt.timedelta(days=20),
            "PT-003": dt.date.today() - dt.timedelta(days=75),
            "PT-004": dt.date.today() - dt.timedelta(days=10),
            "PT-005": dt.date.today() - dt.timedelta(days=120),
        }

        for p in patients:
            start = starts[p]
            for d in range(0, 60):
                date = dt.date.today() - dt.timedelta(days=d)
                if date < start:
                    continue
                taken_prob = 0.65 if p in ["PT-003", "PT-005"] else 0.88
                dose_taken = (random.random() < taken_prob)
                upsert_dots_record(date.isoformat(), p, dose_taken, note="demo tick")

    # Adherence snapshots
    if include_adherence:
        for p in ["PT-001", "PT-002", "PT-003", "PT-004", "PT-005"]:
            dots = load_dots()
            if not dots.empty:
                dp = dots[dots["patient_code"].astype(str) == p]
                start_date = min(dp["date"]) if not dp.empty else (dt.date.today() - dt.timedelta(days=30))
            else:
                start_date = dt.date.today() - dt.timedelta(days=30)

            missed_7 = compute_recent_missed_from_dots(p, 7)
            missed_28 = compute_recent_missed_from_dots(p, 28)
            adh28 = compute_adherence_28d_percent(missed_28)
            high_rule = missed_over_25pct(missed_28)
            wks = weeks_since(start_date)
            cum_expected, cum_missed, cum_pct = compute_cumulative_from_dots(p, start_date)

            if missed_28 >= 10:
                streak = random.choice(["1 week", "2 weeks", "3 weeks", "1 month or more"])
            elif missed_28 >= 5:
                streak = random.choice(["3–6 days", "1 week"])
            else:
                streak = random.choice(["0 days", "1–2 days", "3–6 days"])

            completed = False
            risk, risk_cat, _ = predict_adherence_risk(missed_28, streak, cum_pct, completed)

            save_adherence({
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                "patient_code": p,
                "treatment_start_date": start_date.isoformat(),
                "planned_duration_months": 6,
                "weeks_on_treatment": int(wks),
                "missed_days_last_7": int(missed_7),
                "missed_days_last_28": int(missed_28),
                "longest_missed_streak": streak,
                "completed_regimen": str(bool(completed)),
                "adherence_28d_percent": float(adh28),
                "flag_high_risk_25pct_missed": str(bool(high_rule)),
                "cumulative_expected_days": int(cum_expected),
                "cumulative_missed_days": int(cum_missed),
                "cumulative_adherence_percent": float(cum_pct),
                "predicted_risk_score": float(risk),
                "predicted_risk_category": risk_cat,
                "note": "demo snapshot",
            })

    # Docking queue
    if include_docking:
        ts = dt.datetime.now().isoformat(timespec="seconds")
        demo_rows = [
            {"timestamp": ts, "target": "InhA", "drug_name": "Isoniazid (control)", "drug_id": "DB00951", "smiles": "N/A", "notes": "demo queue"},
            {"timestamp": ts, "target": "DNA gyrase (GyrA/B)", "drug_name": "Moxifloxacin", "drug_id": "DB00218", "smiles": "N/A", "notes": "demo queue"},
            {"timestamp": ts, "target": "ATP synthase", "drug_name": "Bedaquiline", "drug_id": "DB08904", "smiles": "N/A", "notes": "demo queue"},
        ]
        save_to_docking_queue(demo_rows)


def clear_all_demo_data():
    ensure_storage()
    for p in [EVENTS_CSV, DOCKING_QUEUE_CSV, ADHERENCE_CSV, DOTS_CSV]:
        try:
            os.remove(p)
        except Exception:
            pass
    ensure_storage()


# =========================================================
# Streamlit App
# =========================================================
ensure_storage()
st.set_page_config(page_title="OHIH TB Platform (All-in-One)", layout="wide")

st.title("OHIH TB Platform (Hackathon Build)")
st.caption("Obj1 Repurposing • Obj2 Diagnosis • A Microscopy AI • B Adherence+DOTS+Risk • Obj3 Docking+Interpretation • Obj4 Outbreak • D Reports+Pitch • Demo Data Generator")

module = st.sidebar.radio(
    "Modules",
    [
        "Home",
        "Drug Repurposing (Obj 1)",
        "Digital Diagnosis (Obj 2)",
        "Microscopy AI (A)",
        "Treatment Adherence + DOTS + Risk (B)",
        "Docking + Interpretation (Obj 3 + C)",
        "Outbreak Analytics (Obj 4)",
        "Reports & Pitch (D)",
    ]
)

# ---------------- Home ----------------
if module == "Home":
    st.subheader("Flagship demo path")
    st.markdown("""
1) **Digital Diagnosis (Obj2)** → Save event  
2) **Microscopy AI (A)** → upload image demo  
3) **Adherence + DOTS + Risk (B)** → tick dose → see risk (+ >25% missed flag)  
4) **Outbreak Analytics (Obj4)** → trend/alerts/heatmap  
5) **Repurposing (Obj1)** → send to docking queue  
6) **Docking + Interpretation (Obj3+C)** → run docking → interpret
""")
    st.info("Judges love the continuity-of-care story: diagnosis → adherence → outbreak → drug discovery.")

    st.divider()
    st.subheader("Hackathon Demo Data Generator (NEW)")

    colA, colB = st.columns([1, 1])
    with colA:
        n_events = st.slider("Number of demo TB events", 20, 250, 80, 10)
        days = st.slider("Spread across (days)", 7, 120, 30, 1)
        spike_area = st.selectbox(
            "Spike area (cluster hotspot)",
            ["Rivers | Port Harcourt", "Rivers | Obio/Akpor", "Lagos | Ikeja", "Kano | Municipal"]
        )
        spike_mult = st.slider("Spike strength", 1, 6, 3, 1)

    with colB:
        include_dots = st.checkbox("Generate DOTS daily ticks", True)
        include_adherence = st.checkbox("Generate adherence snapshots", True)
        include_docking = st.checkbox("Generate docking queue items", True)

        if st.button("✅ Generate demo data"):
            generate_demo_data(
                n_events=n_events,
                days=days,
                spike_area=spike_area,
                spike_multiplier=spike_mult,
                include_dots=include_dots,
                include_adherence=include_adherence,
                include_docking=include_docking,
            )
            st.success("Demo data generated ✅ Go to Outbreak Analytics / Adherence / Reports to see it.")

        if st.button("🧹 Clear ALL demo data (reset)"):
            clear_all_demo_data()
            st.success("All demo data cleared ✅")

# ---------------- Obj 1 ----------------
elif module == "Drug Repurposing (Obj 1)":
    st.header("Objective 1 — Drug Repurposing (MVP)")
    target = st.selectbox("Select TB target", ["InhA", "DprE1", "DNA gyrase (GyrA/B)", "ATP synthase"])
    dfc = repurpose_candidates_for_target(target)
    st.dataframe(dfc[["drug_name", "drug_id", "smiles", "notes"]], use_container_width=True)

    selected = st.multiselect("Select drugs to send to docking queue", dfc["drug_name"].tolist())
    note = st.text_input("Notes (optional)", value="From repurposing shortlist (MVP)")
    if st.button("Send selected to docking queue"):
        if not selected:
            st.error("Select at least one drug.")
        else:
            ts = dt.datetime.now().isoformat(timespec="seconds")
            rows = []
            for name in selected:
                row = dfc[dfc["drug_name"] == name].iloc[0].to_dict()
                rows.append({
                    "timestamp": ts,
                    "target": target,
                    "drug_name": row["drug_name"],
                    "drug_id": row["drug_id"],
                    "smiles": row["smiles"],
                    "notes": note
                })
            save_to_docking_queue(rows)
            st.success(f"Sent {len(rows)} drug(s) ✅ Now go to **Docking + Interpretation**.")

# ---------------- Obj 2 ----------------
elif module == "Digital Diagnosis (Obj 2)":
    st.header("Objective 2 — Digital TB Diagnosis + Decision Support")

    left, right = st.columns([1.2, 0.8], gap="large")
    with left:
        setting = st.selectbox("Clinical setting", list(DEFAULT_PRETEST.keys()))
        danger = st.checkbox("Red flags present (severe distress/shock/altered sensorium/massive hemoptysis)")

        symptoms = st.multiselect("Symptoms", [s[0] for s in SYMPTOMS])
        risks = st.multiselect("Risk factors", [r[0] for r in RISK_FACTORS])

        st.divider()
        gx = st.selectbox("GeneXpert MTB/RIF", ["Not done", "Detected", "Not detected"])
        mic = st.selectbox("Sputum microscopy", ["Not done", "Positive", "Negative"])
        cxr = st.selectbox("Chest X-ray", ["Not done", "Suggestive", "Not suggestive"])

        st.divider()
        st.subheader("Anonymized location for analytics")
        state = st.text_input("State", value="Rivers")
        lga_or_area = st.text_input("LGA/Area", value="Port Harcourt")
        c1, c2 = st.columns(2)
        with c1:
            lat = st.number_input("Latitude (optional)", value=4.8156, format="%.6f")
        with c2:
            lon = st.number_input("Longitude (optional)", value=7.0498, format="%.6f")

    with right:
        st.subheader("Model settings")
        pretest = st.slider("Pre-test probability", 0.01, 0.60, DEFAULT_PRETEST[setting], 0.01)
        boost_cap = st.slider("Max symptom/risk boost", 0.05, 0.40, 0.20, 0.01)

        gx_sens = st.slider("GeneXpert sensitivity", 0.50, 0.98, TESTS["GeneXpert"]["sens"], 0.01)
        gx_spec = st.slider("GeneXpert specificity", 0.70, 0.995, TESTS["GeneXpert"]["spec"], 0.005)

        mic_sens = st.slider("Microscopy sensitivity", 0.20, 0.85, TESTS["Microscopy"]["sens"], 0.01)
        mic_spec = st.slider("Microscopy specificity", 0.80, 0.995, TESTS["Microscopy"]["spec"], 0.005)

        cxr_sens = st.slider("CXR sensitivity", 0.50, 0.95, TESTS["CXR"]["sens"], 0.01)
        cxr_spec = st.slider("CXR specificity", 0.30, 0.90, TESTS["CXR"]["spec"], 0.01)

    boost = symptom_risk_boost(symptoms, risks, boost_cap)
    pretest_boosted = clamp01(pretest + boost)
    lrs = []

    if gx == "Detected":
        lrs.append(lr_positive(gx_sens, gx_spec))
    elif gx == "Not detected":
        lrs.append(lr_negative(gx_sens, gx_spec))

    if mic == "Positive":
        lrs.append(lr_positive(mic_sens, mic_spec))
    elif mic == "Negative":
        lrs.append(lr_negative(mic_sens, mic_spec))

    if cxr == "Suggestive":
        lrs.append(lr_positive(cxr_sens, cxr_spec))
    elif cxr == "Not suggestive":
        lrs.append(lr_negative(cxr_sens, cxr_spec))

    post_prob = apply_lrs(pretest_boosted, lrs) if lrs else pretest_boosted
    category, advice = decision_support(post_prob, gx, danger)

    st.divider()
    a, b = st.columns([0.9, 1.1], gap="large")
    with a:
        st.metric("TB Probability", f"{post_prob*100:.1f}%")
        st.write(f"**Category:** {category}")
        st.write(f"Pre-test: {pretest*100:.1f}% | Boost: +{boost*100:.1f}% | Tests applied: {len(lrs)}")
    with b:
        st.success(advice)
        st.caption("Educational demo — does not replace clinical judgment or TB program guidelines.")

    st.divider()
    if st.button("Save anonymized event"):
        save_event({
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "state": state.strip(),
            "lga_or_area": lga_or_area.strip(),
            "lat": float(lat),
            "lon": float(lon),
            "tb_probability": float(post_prob),
            "category": category,
            "gene_xpert": gx
        })
        st.success("Saved ✅ Go to **Outbreak Analytics**.")

# ---------------- A) Microscopy AI ----------------
elif module == "Microscopy AI (A)":
    st.header("A) Microscopy AI (Demo)")
    st.caption("Upload a microscopy image (smartphone photo). This demo AI produces a suspicion score.")

    img_file = st.file_uploader("Upload microscopy image (jpg/png)", type=["jpg", "jpeg", "png"])
    if img_file is None:
        st.info("Upload a microscopy image to run the demo AI.")
    else:
        img = Image.open(img_file)
        st.image(img, caption="Uploaded image", use_container_width=True)

        result = microscopy_demo_score(img)
        st.metric("AI suspicion score", f"{result['prob']*100:.1f}%")

        if result["prob"] >= 0.75:
            st.error(result["label"])
        elif result["prob"] >= 0.50:
            st.warning(result["label"])
        else:
            st.success(result["label"])

        st.code(result["explain"])
        st.caption("Demo only. Real deployment would use validated CNN models trained on labeled AFB microscopy datasets.")

# ---------------- B) Adherence + DOTS + Risk ----------------
elif module == "Treatment Adherence + DOTS + Risk (B)":
    st.header("B) Treatment Adherence + DOTS + Risk Prediction")
    st.caption("Daily DOTS tick + auto missed-dose stats + cumulative adherence + predicted risk.")

    patient_code = st.text_input("Patient code (anonymized)", value="PT-001")
    start_date = st.date_input("Treatment start date", value=dt.date.today() - dt.timedelta(days=14))
    planned_duration_months = st.selectbox("Planned regimen duration (months)", [2, 4, 6, 9, 12], index=2)

    st.divider()
    st.subheader("DOTS Daily Tick (Today)")
    today = dt.date.today()
    dose_taken = st.checkbox("Dose taken today", value=False)
    dots_note = st.text_input("DOTS note (optional)", value="")
    if st.button("Save today's DOTS tick"):
        upsert_dots_record(today.isoformat(), patient_code.strip(), bool(dose_taken), dots_note.strip())
        st.success("Saved today's DOTS tick ✅")

    missed_7 = compute_recent_missed_from_dots(patient_code.strip(), 7)
    missed_28 = compute_recent_missed_from_dots(patient_code.strip(), 28)
    adh28 = compute_adherence_28d_percent(missed_28)
    high_rule = missed_over_25pct(missed_28)
    wks = weeks_since(start_date)
    cum_expected, cum_missed, cum_pct = compute_cumulative_from_dots(patient_code.strip(), start_date)

    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Missed (7d)", missed_7)
        st.metric("Missed (28d)", missed_28)
    with c2:
        st.metric("28-day adherence", f"{adh28:.1f}%")
        st.metric("Weeks on treatment", f"{wks}")
    with c3:
        st.metric("Cumulative missed", f"{cum_missed} / {cum_expected}")
        st.metric("Cumulative adherence", f"{cum_pct:.1f}%")

    st.divider()
    st.subheader("Risk prediction inputs")
    streak = st.selectbox("Longest missed streak (category)", STREAK_OPTIONS, index=0)
    completed = st.checkbox("Regimen completed", value=False)

    risk, risk_cat, risk_msg = predict_adherence_risk(missed_28, streak, cum_pct, completed)

    st.subheader("Risk prediction output")
    st.metric("Predicted risk score", f"{risk:.2f}")
    if risk >= 0.80:
        st.error(risk_cat + " — " + risk_msg)
    elif risk >= 0.60:
        st.warning(risk_cat + " — " + risk_msg)
    elif risk >= 0.35:
        st.info(risk_cat + " — " + risk_msg)
    else:
        st.success(risk_cat + " — " + risk_msg)

    if high_rule and not completed:
        st.error("Program rule flag: missed >25% in last 28 days → higher risk of not fully cured / resistance.")

    note = st.text_input("Clinical note (optional)", value="")
    if st.button("Save adherence snapshot"):
        save_adherence({
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "patient_code": patient_code.strip(),
            "treatment_start_date": start_date.isoformat(),
            "planned_duration_months": int(planned_duration_months),
            "weeks_on_treatment": int(wks),
            "missed_days_last_7": int(missed_7),
            "missed_days_last_28": int(missed_28),
            "longest_missed_streak": streak,
            "completed_regimen": str(bool(completed)),
            "adherence_28d_percent": float(adh28),
            "flag_high_risk_25pct_missed": str(bool(high_rule)),
            "cumulative_expected_days": int(cum_expected),
            "cumulative_missed_days": int(cum_missed),
            "cumulative_adherence_percent": float(cum_pct),
            "predicted_risk_score": float(risk),
            "predicted_risk_category": risk_cat,
            "note": note.strip(),
        })
        st.success("Saved adherence snapshot ✅")

    st.divider()
    st.subheader("Recent DOTS records")
    ddf = load_dots()
    if ddf.empty:
        st.info("No DOTS records yet.")
    else:
        view = ddf[ddf["patient_code"].astype(str) == str(patient_code.strip())].copy()
        view = view.sort_values("date", ascending=False).head(60)
        st.dataframe(view, use_container_width=True)

    st.subheader("Adherence snapshots")
    adf = load_adherence()
    if adf.empty:
        st.info("No adherence snapshots saved yet.")
    else:
        v2 = adf[adf["patient_code"].astype(str) == str(patient_code.strip())].copy()
        v2 = v2.sort_values("timestamp", ascending=False)
        st.dataframe(v2, use_container_width=True)
        if not v2.empty and "cumulative_adherence_percent" in v2.columns:
            st.subheader("Trend: cumulative adherence (%)")
            st.line_chart(v2.sort_values("timestamp").set_index("timestamp")["cumulative_adherence_percent"])

# ---------------- Obj 3 + C) Docking + Interpretation ----------------
elif module == "Docking + Interpretation (Obj 3 + C)":
    st.header("Obj 3 + C — Docking + Interpretation")
    st.caption("Run Vina (real) and get interpretation + pose table.")

    q = load_docking_queue()
    if q.empty:
        st.info("Docking queue is empty (optional). Use Repurposing or Demo Generator to add drugs.")
    else:
        st.subheader("Docking queue")
        st.dataframe(q.sort_values("timestamp", ascending=False), use_container_width=True)

    st.divider()
    mode = st.radio("Mode", ["Demo (instant)", "Real (AutoDock Vina v1.2.3)"], horizontal=True)

    if mode == "Demo (instant)":
        ligand_name = st.text_input("Ligand name (demo)", value="Ligand")
        score = -6.0 - ((sum(ord(c) for c in ligand_name) % 60) / 10.0)
        label, expl = interpret_affinity(score)
        st.metric("Demo affinity", f"{score:.2f} kcal/mol")
        st.info(label)
        st.write(expl)

    else:
        vina_cmd = st.text_input("Path to vina.exe", value=r"C:\tools\vina\vina.exe")
        receptor_file = st.file_uploader("Upload receptor PDBQT (protein)", type=["pdbqt"])
        ligand_file = st.file_uploader("Upload ligand PDBQT (drug)", type=["pdbqt"])

        st.subheader("Docking box")
        auto_center = st.checkbox("Auto-center from receptor", value=True)

        colA, colB, colC = st.columns(3)
        with colA:
            center_x = st.number_input("center_x", value=0.0)
        with colB:
            center_y = st.number_input("center_y", value=0.0)
        with colC:
            center_z = st.number_input("center_z", value=0.0)

        colD, colE, colF = st.columns(3)
        with colD:
            size_x = st.number_input("size_x", value=50.0, min_value=5.0)
        with colE:
            size_y = st.number_input("size_y", value=50.0, min_value=5.0)
        with colF:
            size_z = st.number_input("size_z", value=50.0, min_value=5.0)

        colG, colH = st.columns(2)
        with colG:
            exhaustiveness = st.number_input("exhaustiveness", value=8, min_value=1, max_value=64, step=1)
        with colH:
            num_modes = st.number_input("num_modes", value=9, min_value=1, max_value=30, step=1)

        if st.button("Run REAL docking"):
            if receptor_file is None or ligand_file is None:
                st.error("Upload BOTH receptor and ligand PDBQT files.")
                st.stop()

            receptor_bytes = receptor_file.getbuffer().tobytes()
            ligand_bytes = ligand_file.getbuffer().tobytes()

            if auto_center:
                rx, ry, rz = compute_receptor_center_from_pdbqt_text(receptor_bytes.decode("utf-8", errors="ignore"))
                center_x, center_y, center_z = rx, ry, rz
                st.info(f"Auto center: x={center_x:.3f}, y={center_y:.3f}, z={center_z:.3f}")

            with tempfile.TemporaryDirectory() as tmp:
                receptor_path = os.path.join(tmp, "receptor.pdbqt")
                ligand_path = os.path.join(tmp, "ligand.pdbqt")
                out_path = os.path.join(tmp, "out.pdbqt")

                with open(receptor_path, "wb") as f:
                    f.write(receptor_bytes)
                with open(ligand_path, "wb") as f:
                    f.write(ligand_bytes)

                cmd = [
                    vina_cmd,
                    "--receptor", receptor_path,
                    "--ligand", ligand_path,
                    "--center_x", str(center_x),
                    "--center_y", str(center_y),
                    "--center_z", str(center_z),
                    "--size_x", str(size_x),
                    "--size_y", str(size_y),
                    "--size_z", str(size_z),
                    "--exhaustiveness", str(int(exhaustiveness)),
                    "--num_modes", str(int(num_modes)),
                    "--out", out_path
                ]

                try:
                    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
                except FileNotFoundError:
                    st.error("vina.exe not found. Check the path.")
                    st.stop()

                vina_output = (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else "")
                if completed.returncode != 0:
                    st.error("Vina failed:")
                    st.code(vina_output)
                    st.stop()

                st.session_state["last_vina_output"] = vina_output

                poses = parse_vina_table(vina_output)
                if poses:
                    best = poses[0].affinity
                    label, expl = interpret_affinity(best)
                    st.success("Docking complete ✅")
                    st.metric("Best affinity (mode 1)", f"{best:.2f} kcal/mol")
                    st.info(label)
                    st.write(expl)

                    st.subheader("Pose table")
                    st.dataframe(
                        pd.DataFrame([{
                            "mode": p.mode, "affinity": p.affinity, "rmsd_lb": p.rmsd_lb, "rmsd_ub": p.rmsd_ub
                        } for p in poses]),
                        use_container_width=True
                    )
                else:
                    st.warning("Docking finished but pose table was not parsed. See output below.")

                st.subheader("Vina output (raw)")
                st.code(vina_output)

                out_bytes = open(out_path, "rb").read()
                stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button("Download out.pdbqt", data=out_bytes, file_name=f"out_{stamp}.pdbqt")
                st.download_button("Download vina_output.txt", data=vina_output.encode("utf-8"), file_name=f"vina_{stamp}.txt")

# ---------------- Obj 4 ----------------
elif module == "Outbreak Analytics (Obj 4)":
    st.header("Objective 4 — Early Outbreak Detection & Analytics")

    df = load_events()
    if df.empty:
        st.warning("No events saved yet. Use Demo Generator (Home) or save an event from Digital Diagnosis.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        days = st.selectbox("Time window (days)", [7, 14, 30, 60], index=1)
    with col2:
        prob_min = st.slider("Minimum TB probability to include", 0.0, 1.0, 0.30, 0.05)
    with col3:
        states = sorted([s for s in df["state"].dropna().unique().tolist() if str(s).strip()])
        state_filter = st.multiselect("States", states, default=states[:1] if states else [])

    cutoff = df["timestamp"].max() - pd.Timedelta(days=days)
    dff = df[df["timestamp"] >= cutoff].copy()
    dff = dff[dff["tb_probability"] >= prob_min].copy()
    if state_filter:
        dff = dff[dff["state"].isin(state_filter)].copy()

    st.write(f"**Events in view:** {len(dff)} (of {len(df)} total saved)")

    st.subheader("Trend (daily suspected TB events)")
    daily = dff.assign(day=dff["timestamp"].dt.date).groupby("day").size().reset_index(name="count").sort_values("day")
    st.line_chart(daily.set_index("day")["count"])

    st.subheader("Cluster alerts (simple spike detection)")
    alerts = compute_alerts(dff, days_window=7, min_cases=3, ratio_threshold=2.0)
    st.dataframe(alerts, use_container_width=True)

    st.subheader("Heatmap")
    if not HAS_MAP:
        st.warning("Heatmap requires: pip install folium streamlit-folium")
    else:
        map_df = dff[(dff["lat"].abs() > 0.000001) & (dff["lon"].abs() > 0.000001)].copy()
        if map_df.empty:
            st.warning("No valid lat/lon points found. Add coordinates when saving events.")
        else:
            center_lat = float(map_df["lat"].mean())
            center_lon = float(map_df["lon"].mean())
            m = folium.Map(location=[center_lat, center_lon], zoom_start=7)
            heat_points = map_df[["lat", "lon", "tb_probability"]].values.tolist()
            HeatMap(heat_points, radius=18, blur=12, max_zoom=10).add_to(m)
            st_folium(m, width=1100, height=600)

    st.subheader("Raw data")
    st.dataframe(dff.sort_values("timestamp", ascending=False), use_container_width=True)

# ---------------- D) Reports & Pitch ----------------
elif module == "Reports & Pitch (D)":
    st.header("D) Reports & Presentation Polish")

    st.subheader("Export datasets")
    ev = load_events()
    dq = load_docking_queue()
    ad = load_adherence()
    dots = load_dots()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.download_button("Download events.csv", data=df_to_csv_bytes(ev), file_name="events.csv")
    with c2:
        st.download_button("Download docking_queue.csv", data=df_to_csv_bytes(dq), file_name="docking_queue.csv")
    with c3:
        st.download_button("Download adherence.csv", data=df_to_csv_bytes(ad), file_name="adherence.csv")
    with c4:
        st.download_button("Download dots_daily.csv", data=df_to_csv_bytes(dots), file_name="dots_daily.csv")

    st.divider()
    st.subheader("Generate patient clinical report (text)")
    patient_code = st.text_input("Patient code for report", value="PT-001")
    adf = load_adherence()
    if adf.empty:
        st.info("No adherence snapshots found yet. Save one in the Adherence module or use Demo Generator.")
    else:
        latest = adf[adf["patient_code"].astype(str) == str(patient_code)].sort_values("timestamp", ascending=False)
        if latest.empty:
            st.warning("No adherence snapshot found for this patient code.")
        else:
            row = latest.iloc[0]
            start_date = row["treatment_start_date"].date() if pd.notnull(row["treatment_start_date"]) else dt.date.today()
            planned = int(row.get("planned_duration_months", 6) if pd.notnull(row.get("planned_duration_months", 6)) else 6)
            missed7 = int(row.get("missed_days_last_7", 0) if pd.notnull(row.get("missed_days_last_7", 0)) else 0)
            missed28 = int(row.get("missed_days_last_28", 0) if pd.notnull(row.get("missed_days_last_28", 0)) else 0)
            streak = str(row.get("longest_missed_streak", "0 days"))
            adh28 = float(row.get("adherence_28d_percent", 0.0) if pd.notnull(row.get("adherence_28d_percent", 0.0)) else 0.0)
            cum_expected = int(row.get("cumulative_expected_days", 0) if pd.notnull(row.get("cumulative_expected_days", 0)) else 0)
            cum_missed = int(row.get("cumulative_missed_days", 0) if pd.notnull(row.get("cumulative_missed_days", 0)) else 0)
            cum_pct = float(row.get("cumulative_adherence_percent", 0.0) if pd.notnull(row.get("cumulative_adherence_percent", 0.0)) else 0.0)
            risk = float(row.get("predicted_risk_score", 0.0) if pd.notnull(row.get("predicted_risk_score", 0.0)) else 0.0)
            risk_cat = str(row.get("predicted_risk_category", "Unknown"))
            completed = str(row.get("completed_regimen", "False")).lower() == "true"

            report = make_clinical_report_text(
                patient_code=patient_code,
                start_date=start_date,
                planned_months=planned,
                missed7=missed7,
                missed28=missed28,
                streak=streak,
                adh28=adh28,
                cum_expected=cum_expected,
                cum_missed=cum_missed,
                cum_pct=cum_pct,
                risk=risk,
                risk_cat=risk_cat,
                completed=completed
            )
            st.text_area("Report preview", report, height=280)
            st.download_button("Download report.txt", data=report.encode("utf-8"), file_name=f"{patient_code}_report.txt")

    st.divider()
    st.subheader("Hackathon pitch script")
    script = make_pitch_script()
    st.text_area("Pitch script", script, height=320)
    st.download_button("Download pitch_script.txt", data=script.encode("utf-8"), file_name="pitch_script.txt")

    st.divider()
    st.subheader("Docking output interpretation (paste or use last run)")
    pasted = st.text_area("Paste Vina output here (optional)", value="", height=180)
    vina_output = pasted.strip() if pasted.strip() else st.session_state.get("last_vina_output", "")
    if vina_output:
        poses = parse_vina_table(vina_output)
        if poses:
            best = poses[0].affinity
            label, expl = interpret_affinity(best)
            st.metric("Best affinity (mode 1)", f"{best:.2f} kcal/mol")
            st.info(label)
            st.write(expl)
            st.dataframe(pd.DataFrame([p.__dict__ for p in poses]), use_container_width=True)
        else:
            st.warning("Could not parse a pose table from the provided output.")
    else:
        st.info("Run a docking first (Docking module), or paste Vina output above.")
