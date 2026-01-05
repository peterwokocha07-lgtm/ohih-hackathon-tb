import os
import csv
import datetime as dt
import subprocess
import tempfile
import statistics

import pandas as pd
import streamlit as st

# Optional mapping
try:
    import folium
    from folium.plugins import HeatMap
    from streamlit_folium import st_folium
    HAS_MAP = True
except Exception:
    HAS_MAP = False


# =========================================================
# Storage
# =========================================================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
EVENTS_CSV = os.path.join(DATA_DIR, "events.csv")
DOCKING_QUEUE_CSV = os.path.join(DATA_DIR, "docking_queue.csv")

EVENT_FIELDS = ["timestamp", "state", "lga_or_area", "lat", "lon", "tb_probability", "category", "gene_xpert"]
DOCKING_FIELDS = ["timestamp", "target", "drug_name", "drugbank_or_pubchem_id", "smiles", "notes"]

def ensure_storage():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(EVENTS_CSV):
        with open(EVENTS_CSV, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=EVENT_FIELDS).writeheader()
    if not os.path.exists(DOCKING_QUEUE_CSV):
        with open(DOCKING_QUEUE_CSV, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=DOCKING_FIELDS).writeheader()

def save_event(row: dict):
    ensure_storage()
    with open(EVENTS_CSV, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=EVENT_FIELDS).writerow(row)

def load_events() -> pd.DataFrame:
    ensure_storage()
    df = pd.read_csv(EVENTS_CSV)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["tb_probability"] = pd.to_numeric(df["tb_probability"], errors="coerce")
    return df.dropna(subset=["timestamp"])

def save_to_docking_queue(rows: list[dict]):
    ensure_storage()
    with open(DOCKING_QUEUE_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=DOCKING_FIELDS)
        for r in rows:
            w.writerow(r)

def load_docking_queue() -> pd.DataFrame:
    ensure_storage()
    df = pd.read_csv(DOCKING_QUEUE_CSV)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


# =========================================================
# Objective 2: Simple Bayesian / LR demo model
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
# Objective 1: Repurposing demo dataset
# =========================================================
REPURPOSE_DB = [
    {"target": "InhA", "drug_name": "Isoniazid (control)", "id": "DB00951", "smiles": "NNC(=O)c1ccncc1", "notes": "First-line TB drug (control)"},
    {"target": "DNA gyrase (GyrA/B)", "drug_name": "Levofloxacin", "id": "DB01137", "smiles": "N/A", "notes": "Used in TB regimens"},
    {"target": "DNA gyrase (GyrA/B)", "drug_name": "Moxifloxacin", "id": "DB00218", "smiles": "N/A", "notes": "Used in TB regimens"},
    {"target": "ATP synthase", "drug_name": "Bedaquiline", "id": "DB08904", "smiles": "N/A", "notes": "Approved for MDR-TB"},
    {"target": "DprE1", "drug_name": "BTZ043 (investigational)", "id": "N/A", "smiles": "N/A", "notes": "Demo placeholder"},
]

def repurpose_candidates_for_target(target: str) -> pd.DataFrame:
    df = pd.DataFrame(REPURPOSE_DB)
    return df[df["target"] == target].reset_index(drop=True)

# =========================================================
# Obj 4: Alerting
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
# Objective 3: Docking helpers
# =========================================================
def compute_receptor_center_from_pdbqt(pdbqt_text: str) -> tuple[float, float, float]:
    xs, ys, zs = [], [], []
    for line in pdbqt_text.splitlines():
        if line.startswith(("ATOM", "HETATM")) and len(line) >= 54:
            try:
                xs.append(float(line[30:38]))
                ys.append(float(line[38:46]))
                zs.append(float(line[46:54]))
            except:
                pass
    if not xs:
        # fallback
        return 0.0, 0.0, 0.0
    return float(statistics.mean(xs)), float(statistics.mean(ys)), float(statistics.mean(zs))

def parse_vina_best_affinity(log_text: str):
    # Finds the first affinity line in the "mode | affinity | rmsd..." table
    for line in log_text.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0].isdigit():
            try:
                return float(parts[1])
            except:
                pass
    return None

def demo_score(name: str) -> float:
    base = (sum(ord(c) for c in name) % 60) / 10.0
    return -6.0 - base

# =========================================================
# Streamlit UI
# =========================================================
st.set_page_config(page_title="OHIH TB Platform (MVP)", layout="wide")
st.title("OHIH TB Platform (MVP)")
st.caption("Integrated: Diagnosis (Obj 2) • Outbreak (Obj 4) • Repurposing (Obj 1) • Docking (Obj 3) • Dashboard (Obj 5)")

module = st.sidebar.radio(
    "Select module",
    ["Home", "Digital Diagnosis (Obj 2)", "Outbreak Analytics (Obj 4)", "Drug Repurposing (Obj 1)", "Docking (Obj 3)"]
)

if module == "Home":
    st.subheader("MVP Status")
    st.markdown("""
    ✅ **Obj 2** – TB probability + decision support  
    ✅ **Obj 4** – Event saving + heatmap + alerts  
    ✅ **Obj 1** – Repurposing shortlist + send to docking queue  
    ✅ **Obj 3** – Demo docking + Real AutoDock Vina docking  
    ✅ **Obj 5** – Unified dashboard (this app)
    """)

elif module == "Digital Diagnosis (Obj 2)":
    st.header("Digital TB Diagnosis (Explainable AI)")

    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        setting = st.selectbox("Clinical setting", list(DEFAULT_PRETEST.keys()))
        danger = st.checkbox("Red flags present")

        symptoms = st.multiselect("Symptoms", [s[0] for s in SYMPTOMS])
        risks = st.multiselect("Risk factors", [r[0] for r in RISK_FACTORS])

        st.divider()
        gx = st.selectbox("GeneXpert MTB/RIF", ["Not done", "Detected", "Not detected"])
        mic = st.selectbox("Sputum microscopy", ["Not done", "Positive", "Negative"])
        cxr = st.selectbox("Chest X-ray", ["Not done", "Suggestive", "Not suggestive"])

        st.divider()
        st.subheader("Anonymized location for outbreak analytics")
        state = st.text_input("State", value="Rivers")
        lga_or_area = st.text_input("LGA/Area", value="Port Harcourt")
        col_lat, col_lon = st.columns(2)
        with col_lat:
            lat = st.number_input("Latitude (optional)", value=4.8156, format="%.6f")
        with col_lon:
            lon = st.number_input("Longitude (optional)", value=7.0498, format="%.6f")

    with right:
        st.subheader("Model configuration (editable)")
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
        st.write(f"Pre-test: {pretest*100:.1f}% | Boost: +{boost*100:.1f}% | Tests: {len(lrs)}")
    with b:
        st.success(advice)
        st.caption("Demo decision support only — not a substitute for clinical judgment.")

    st.divider()
    st.subheader("Save event to outbreak analytics (anonymized)")
    if st.button("Save anonymized event"):
        row = {
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "state": state.strip(),
            "lga_or_area": lga_or_area.strip(),
            "lat": float(lat),
            "lon": float(lon),
            "tb_probability": float(post_prob),
            "category": category,
            "gene_xpert": gx
        }
        save_event(row)
        st.success("Saved ✅ Go to **Outbreak Analytics (Obj 4)** to view heatmap and alerts.")

elif module == "Outbreak Analytics (Obj 4)":
    st.header("Early Outbreak Detection (Obj 4)")
    df = load_events()
    if df.empty:
        st.warning("No events saved yet. Go to **Digital Diagnosis** and click **Save anonymized event**.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        days = st.selectbox("Time window", [7, 14, 30, 60], index=1)
    with col2:
        prob_min = st.slider("Minimum TB probability to include", 0.0, 1.0, 0.30, 0.05)
    with col3:
        states = sorted([s for s in df["state"].dropna().unique().tolist() if str(s).strip() != ""])
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

    st.subheader("Heat map")
    if not HAS_MAP:
        st.warning("Mapping libraries missing. Run: pip install folium streamlit-folium")
        st.stop()

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

elif module == "Drug Repurposing (Obj 1)":
    st.header("AI-Driven Drug Repurposing Engine (Obj 1)")
    st.caption("MVP: Target → shortlist (demo dataset) → send selected to docking queue.")

    target = st.selectbox("Select TB target", ["InhA", "DprE1", "DNA gyrase (GyrA/B)", "ATP synthase"])
    dfc = repurpose_candidates_for_target(target)
    st.subheader("Candidate shortlist")
    st.dataframe(dfc[["drug_name", "id", "smiles", "notes"]], use_container_width=True)

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
                    "drugbank_or_pubchem_id": row["id"],
                    "smiles": row["smiles"],
                    "notes": note
                })
            save_to_docking_queue(rows)
            st.success(f"Sent {len(rows)} drug(s) to docking queue ✅ Go to **Docking (Obj 3)**.")

elif module == "Docking (Obj 3)":
    st.header("Molecular Docking (Obj 3) — Demo + Real Vina")
    st.caption("Demo docking always works. Real docking runs AutoDock Vina on uploaded PDBQT files.")

    q = load_docking_queue()
    if q.empty:
        st.info("Docking queue is empty. (Optional) Use **Drug Repurposing** to populate it.")
    else:
        st.subheader("Docking queue")
        st.dataframe(q.sort_values("timestamp", ascending=False), use_container_width=True)

    st.divider()
    mode = st.radio("Docking mode", ["Demo (instant)", "Real (AutoDock Vina)"], horizontal=True)

    if mode == "Demo (instant)":
        st.subheader("Demo docking (instant scoring)")
        ligand_name = st.text_input("Ligand name (for demo score)", value="Ligand")
        if st.button("Run DEMO docking"):
            score = demo_score(ligand_name)
            st.success("Demo docking complete ✅")
            st.write(f"**Best (demo) affinity:** {score:.2f} kcal/mol")
            st.info("Next: Switch to Real docking to run AutoDock Vina.")

    else:
        st.subheader("Real docking (AutoDock Vina)")
        vina_cmd = st.text_input("Vina executable path", value=r"C:\tools\vina\vina.exe")

        receptor_file = st.file_uploader("Upload receptor PDBQT (protein)", type=["pdbqt"])
        ligand_file = st.file_uploader("Upload ligand PDBQT (small molecule)", type=["pdbqt"])

        st.subheader("Docking box")
        auto_center = st.checkbox("Auto-compute center from receptor (recommended)", value=True)

        # Default big box
        size_x = st.number_input("size_x", value=50.0, min_value=5.0)
        size_y = st.number_input("size_y", value=50.0, min_value=5.0)
        size_z = st.number_input("size_z", value=50.0, min_value=5.0)

        exhaustiveness = st.number_input("exhaustiveness", value=8, min_value=1, max_value=64, step=1)
        num_modes = st.number_input("num_modes", value=9, min_value=1, max_value=30, step=1)

        # Manual center (only used if auto_center is off)
        colA, colB, colC = st.columns(3)
        with colA:
            center_x = st.number_input("center_x", value=0.0)
        with colB:
            center_y = st.number_input("center_y", value=0.0)
        with colC:
            center_z = st.number_input("center_z", value=0.0)

        if st.button("Run REAL docking now"):
            if receptor_file is None or ligand_file is None:
                st.error("Upload BOTH receptor.pdbqt and ligand.pdbqt first.")
                st.stop()

            receptor_bytes = receptor_file.getbuffer().tobytes()
            ligand_bytes = ligand_file.getbuffer().tobytes()

            # Auto center from receptor file contents
            if auto_center:
                rx, ry, rz = compute_receptor_center_from_pdbqt(receptor_bytes.decode("utf-8", errors="ignore"))
                center_x, center_y, center_z = rx, ry, rz
                st.info(f"Auto center: x={center_x:.3f}, y={center_y:.3f}, z={center_z:.3f}")

            with tempfile.TemporaryDirectory() as tmp:
                receptor_path = os.path.join(tmp, "receptor.pdbqt")
                ligand_path = os.path.join(tmp, "ligand.pdbqt")
                out_path = os.path.join(tmp, "out.pdbqt")
                log_path = os.path.join(tmp, "log.txt")

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
                    "--out", out_path,
                    "--log", log_path
                ]

                try:
                    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
                except FileNotFoundError:
                    st.error("Vina executable not found. Check the path in 'Vina executable path'.")
                    st.stop()

                if completed.returncode != 0:
                    st.error("Vina failed. Output below:")
                    st.code((completed.stdout or "") + "\n" + (completed.stderr or ""))
                    st.stop()

                log_txt = open(log_path, "r", encoding="utf-8", errors="ignore").read()
                best = parse_vina_best_affinity(log_txt)

                st.success("Real docking complete ✅")
                if best is not None:
                    st.write(f"**Best affinity (mode 1):** {best:.2f} kcal/mol")
                st.subheader("Vina log")
                st.code(log_txt)

                out_bytes = open(out_path, "rb").read()
                st.download_button("Download docked poses (out.pdbqt)", data=out_bytes, file_name="out.pdbqt", mime="chemical/x-pdbqt")
                st.download_button("Download vina log (log.txt)", data=log_txt.encode("utf-8"), file_name="log.txt", mime="text/plain")
