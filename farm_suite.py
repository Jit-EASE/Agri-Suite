# farm_suite_app.py
# Smart Farm Suite (PastureBase-style) + Camera & Edge/Offline Inference
# Designed by Jit

import os, io, math, base64
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image

# ========= Optional deps (graceful fallbacks) =========
HAS_CV2 = True
try:
    import cv2
except Exception:
    HAS_CV2 = False

HAS_SKIMAGE = True
try:
    from skimage.morphology import skeletonize
    from skimage.filters import sato
except Exception:
    HAS_SKIMAGE = False

HAS_WEBRTC = True
try:
    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration
except Exception:
    HAS_WEBRTC = False

# ========= Optional OpenAI (Contextual AI) =========
USE_OPENAI = False
_openai_mode = None
def _noop_ai(_: str) -> str: return "(AI offline) Using rule-based advisory."
try:
    from openai import OpenAI
    if os.getenv("OPENAI_API_KEY"):
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        def _call_openai(prompt: str) -> str:
            rsp = _client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role":"system","content":"You are a precise Irish agronomy & dairy advisor. Give short, actionable bullet points with rates, timings, and risk flags. Stay conservative and compliant with EU/Nitrates rules."},
                    {"role":"user","content":prompt}],
                temperature=0.2)
            return rsp.choices[0].message.content.strip()
        USE_OPENAI = True
        _openai_mode = "client"
except Exception:
    try:
        import openai
        if os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.getenv("OPENAI_API_KEY")
            def _call_openai(prompt: str) -> str:
                rsp = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role":"system","content":"You are a precise Irish agronomy & dairy advisor. Give short, actionable bullet points with rates, timings, and risk flags. Stay conservative and compliant with EU/Nitrates rules."},
                        {"role":"user","content":prompt}],
                    temperature=0.2)
                return rsp["choices"][0]["message"]["content"].strip()
            USE_OPENAI = True
            _openai_mode = "legacy"
    except Exception:
        USE_OPENAI = False
        _openai_mode = None

# ========= Page Config & Theme =========
st.set_page_config(page_title="Smart Farm Suite (PastureBase-style)", layout="wide", page_icon="🌿")
st.markdown("""
<style>
/* ---- Responsive, non-clipping toolbar ---- */
.toolbar {
  display: flex;
  align-items: center;
  gap: .6rem;
  flex-wrap: wrap;
  width: 100%;
  box-sizing: border-box;
  padding: .65rem .9rem;
  min-height: 70px;
  background: linear-gradient(90deg, #2aa24f, #5fd37a);
  border-radius: 12px;
  color: #fff;
  margin: .25rem 0 1rem 0;
  box-shadow: 0 3px 14px rgba(0,0,0,.07);
  overflow: visible;
}
.toolbar .title{
  font-weight: 800;
  letter-spacing: .3px;
  font-size: clamp(0.95rem, 1vw + 0.55rem, 1.15rem);
  line-height: 1.25;
  flex: 1 1 auto;
}
.toolbar .btn{
  background: #fff;
  color: #1b7c3b;
  border: none;
  border-radius: 10px;
  padding: .45rem .85rem;
  font-weight: 700;
  cursor: pointer;
  box-shadow: 0 2px 8px rgba(0,0,0,.08);
  font-size: clamp(.85rem, .25vw + .8rem, .95rem);
}
.toolbar .btn:active{ transform: translateY(1px); }
@media (max-width: 992px){ .toolbar { padding: .55rem .75rem; border-radius: 10px; } }
@media (max-width: 640px){ .toolbar { padding: .5rem .6rem; border-radius: 8px; } }
.block-container { padding-top: 1rem; padding-bottom: 1.25rem; }
</style>
""", unsafe_allow_html=True)


# ========= Constants & Helpers =========
MONTHS = ["Jan/Feb","March","April","May","June","July","August","September","October"]
FERTILISERS = {
    "Protected Urea (46-0-0)": (0.46, 0.00),
    "38-0-0 + 7S": (0.38, 0.07),
    "26-0-0 + 4S": (0.26, 0.04),
    "CAN (27-0-0)": (0.27, 0.00),
    "10-10-20": (0.10, 0.00),
    "Urea (46-0-0)": (0.46, 0.00),
    "Custom Blend": (0.00, 0.00),
}
ACRE_TO_HA = 0.4046856422

def ytd(series): return pd.Series(series).fillna(0).cumsum().round(1)

# ---- SAFE Excel writer (fix invalid sheet names) ----
def excel_download_button(dfs: dict, filename="FarmSuite_Export.xlsx", label="⬇️ Excel Export"):
    """
    Writes multiple DataFrames to an in-memory Excel file with SAFE worksheet names:
      - disallow []:*?/\
      - trim to 31 chars
      - de-duplicate by adding _1, _2, ...
    """
    import re, io, pandas as pd

    def sanitize_sheet_name(name: str) -> str:
        if not isinstance(name, str):
            name = str(name)
        name = re.sub(r'[\[\]\:\*\?\/\\]', ' ', name)   # remove invalid chars
        name = re.sub(r'\s+', ' ', name).strip()
        if not name: name = "Sheet"
        return name[:31]

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        used = set()
        for raw_name, df in dfs.items():
            base = sanitize_sheet_name(raw_name)
            name = base
            i = 1
            while name in used:
                suffix = f"_{i}"
                name = base[: max(0, 31 - len(suffix))] + suffix
                i += 1
            used.add(name)
            df.to_excel(writer, sheet_name=name, index=True)

    st.download_button(
        label=label,
        data=buffer.getvalue(),
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

def contextual_ai_fallback(plan_dict: Dict) -> str:
    limit = plan_dict.get("n_limit", 170)
    org, chem = plan_dict.get("organic_by_mo", []), plan_dict.get("chemical_by_mo", [])
    tot = [round((org[i] if i<len(org) else 0)+(chem[i] if i<len(chem) else 0),1) for i in range(len(MONTHS))]
    y = sum(tot)
    tips = [
        f"Keep total N ≤ {limit} kg N/ha unless derogation applies.",
        "Front-load slurry in spring (trailing shoe). Pause chemical N if growth outpaces demand.",
        "Use Protected Urea in warm/dry windows; switch to CAN when soil moisture is limiting.",
        "If covers >1,100 kg DM/ha, consider pausing N 2–3 weeks to avoid luxury uptake.",
        "Keep soil pH ≥6.3; target slurry to low P/K paddocks.",
        "Avoid spreading before heavy rain or on saturated soils."
    ]
    return "(Rule-based advisor)\n• " + "\n• ".join(tips) + f"\n• YTD plan ≈ {y:.1f} kg N/ha."

def contextual_ai(prompt: str, plan_dict: Dict) -> str:
    if USE_OPENAI:
        try: return _call_openai(prompt)
        except Exception as e: return f"(OpenAI error: {e})\n" + contextual_ai_fallback(plan_dict)
    return contextual_ai_fallback(plan_dict)

def fit_ols_numpy(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    X1 = np.c_[np.ones((X.shape[0],1)), X]
    beta = np.linalg.pinv(X1.T @ X1) @ (X1.T @ y)
    yhat = X1 @ beta
    ss_res = np.sum((y - yhat)**2); ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot>0 else 0.0
    return beta, float(r2)

def predict_ols_numpy(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    X1 = np.c_[np.ones((X.shape[0],1)), X]
    return X1 @ beta

def calc_thi(temp_c: float, rh: float) -> float:
    return (1.8*temp_c + 32) - (0.55 - 0.0055*rh)*(1.8*temp_c - 26)

# ---- Rebuild plan live from session_state (used by Finance/Reports if cache missing) ----
def get_current_plan_from_state() -> Dict:
    months = MONTHS
    kgN1000 = float(st.session_state.get("kgN_per_1000gal", 10.6))
    area_ha = float(st.session_state.get("area_ha_header", 72.59))
    n_limit = int(st.session_state.get("n_limit_header", 170))

    slurry_gal = [float(st.session_state.slurry_gal_ac.get(m, 0.0)) for m in months]
    org_kgN = [(g/1000.0) * kgN1000 for g in slurry_gal]
    chem_kgN = [float(st.session_state.chem_target_kgN_ha.get(m, 0.0)) for m in months]

    total_monthly = [org_kgN[i] + chem_kgN[i] for i in range(len(months))]
    ytd_kgN_per_ha = float(np.sum(total_monthly))  # total across year (kg N/ha)

    return {
        "n_limit": n_limit,
        "organic_by_mo": [round(x, 1) for x in org_kgN],
        "chemical_by_mo": [round(x, 1) for x in chem_kgN],
        "ytd": ytd_kgN_per_ha,
        "area_ha": area_ha,
    }

# ========= Edge/Offline Inference Utilities =========
def _np_img_from_upload(file) -> np.ndarray:
    """Convert uploaded/camera image to RGB numpy array."""
    img = Image.open(file).convert("RGB")
    return np.array(img)

def _ensure_cv2():
    if not HAS_CV2:
        st.error("OpenCV not installed. Install with: `pip install opencv-python`")
        return False
    return True

def analyze_leaf_rgb(rgb: np.ndarray, vari_thresh: float = 0.02, lesion_canny: Tuple[int,int]=(60,140)) -> Dict:
    """
    Offline leaf stress analysis:
    - VARI = (G - R) / (G + R - B)
    - ExG = 2G - R - B
    - Edge/lesion density via Canny on V channel
    Returns metrics + annotated image.
    """
    if rgb is None: return {}
    arr = rgb.astype(np.float32) / 255.0
    R, G, B = arr[...,0], arr[...,1], arr[...,2]
    vari = (G - R) / (G + R - B + 1e-6)
    exg  = 2*G - R - B

    mean_vari = float(np.clip(np.nanmean(vari), -1, 1))
    if HAS_CV2:
        hsv = cv2.cvtColor((arr*255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        V = hsv[...,2]
        edges = cv2.Canny(V, lesion_canny[0], lesion_canny[1])
        lesion_density = float(edges.mean()/255.0)
    else:
        edges = (np.abs(np.gradient(arr.mean(axis=2))[0])>0.2).astype(np.uint8)*255
        lesion_density = float(edges.mean()/255.0)

    stress_score = float(np.clip((0.15 - mean_vari) * 2.5 + lesion_density*0.8, 0, 1))
    status = "Healthy" if stress_score < 0.33 else ("Moderate" if stress_score < 0.66 else "High Stress")

    heat = ((vari - vari.min())/(vari.max()-vari.min()+1e-6))
    overlay = (np.stack([1-heat, heat, 0.2*np.ones_like(heat)], axis=2)*255).astype(np.uint8)
    out = (0.6*rgb + 0.4*overlay).astype(np.uint8)

    return {
        "mean_VARI": round(mean_vari, 3),
        "lesion_density": round(lesion_density, 3),
        "stress_score": round(stress_score, 3),
        "status": status,
        "edges": edges,
        "overlay": out
    }

def analyze_crack_rgb(rgb: np.ndarray, canny: Tuple[int,int]=(100,200), use_sato: bool=True) -> Dict:
    """Offline crack/surface stress analysis."""
    if not _ensure_cv2(): return {}
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (3,3), 0)
    edges = cv2.Canny(gray_blur, canny[0], canny[1])

    vessel = None
    if HAS_SKIMAGE and use_sato:
        vessel = sato(gray_blur/255.0)
        vessel = (255 * (vessel - vessel.min())/(vessel.max()-vessel.min()+1e-6)).astype(np.uint8)

    density = float(edges.mean()/255.0)
    skel_len = None
    if HAS_SKIMAGE:
        skel = skeletonize((edges>0).astype(np.uint8))
        skel_len = int(skel.sum())

    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    color[edges>0] = [255, 50, 50]
    if vessel is not None:
        v_col = cv2.applyColorMap(vessel, cv2.COLORMAP_TURBO)
        color = cv2.addWeighted(color, 0.6, v_col, 0.4, 0)

    risk = "Low" if density < 0.02 else ("Moderate" if density < 0.06 else "High")
    return {"edge_density": round(density, 4), "skeleton_length": skel_len, "risk": risk, "overlay": color}

# ========= Sidebar =========
with st.sidebar:
    st.markdown("## 🌿 Smart Farm Suite")
    st.markdown("### Navigation")
    section = st.radio(
        "",
        ["Dashboard","Nitrogen Plan","Grass & Crops","Dairy","Livestock","Weather & Soil",
         "Sensors (Camera & Edge AI)","Finance","AI Advisor","Reports","Settings"],
        index=1
    )

# ========= Top Toolbar =========
st.markdown(
    f"""
<div class="toolbar">
  <div class="title">Smart Farm Suite • {datetime.now().strftime('%d %b %Y')}</div>
  <div class="sp"></div>
  <button class="btn" onclick="window.location.reload()">🔄 Refresh</button>
</div>
""", unsafe_allow_html=True)

# ========= Session defaults for planner =========
if "fert_choice" not in st.session_state:
    st.session_state.fert_choice = {m: "Protected Urea (46-0-0)" for m in MONTHS}
if "chem_target_kgN_ha" not in st.session_state:
    st.session_state.chem_target_kgN_ha = {m: 0.0 for m in MONTHS}
if "slurry_gal_ac" not in st.session_state:
    st.session_state.slurry_gal_ac = {m: 0.0 for m in MONTHS}

# ========= Sections =========
if section == "Dashboard":
    st.subheader("Farm KPIs")
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Area (ha)", "72.6")
    with c2: st.metric("Herd (LU)", "120")
    with c3: st.metric("Milk (L/day)", "2,450")
    with c4: st.metric("Carbon (kg CO₂e/L)", "0.98")
    st.info("Use **Nitrogen Plan** to build monthly plans. **Sensors** enables on-device analysis for leaves or surfaces.")

elif section == "Nitrogen Plan":
    st.markdown("### Nitrogen Plan (Organic + Chemical)")
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1: paddock_usage = st.selectbox("Paddock Usage", ["Grazing","Grazing + 1 Silage Cut","Silage Only"], index=0)
    with c2: n_paddocks = st.number_input("No. of Paddocks", min_value=1, max_value=300, value=26, step=1)
    with c3: area_ha = st.number_input("Area of Paddocks (ha)", min_value=0.0, value=72.59, step=0.01, format="%.2f")
    with c4: pct_farm = st.number_input("Percentage of Farm Area (%)", min_value=0.0, max_value=100.0, value=80.0, step=0.5)
    with c5: avail_slurry = st.number_input("Available Slurry (gallons)", min_value=0.0, value=95732.0, step=100.0)

    # Persist key header values for other tabs
    st.session_state["area_ha_header"] = float(area_ha)

    a1,a2,a3,a4 = st.columns(4)
    with a1:
        kgN_per_1000gal = st.number_input("kg N/ha per 1,000 gal/acre (spring)", 0.0, 100.0, 10.6, step=0.1)
        st.session_state["kgN_per_1000gal"] = float(kgN_per_1000gal)  # persist
    with a2:
        bag_weight = st.number_input("Fertiliser Bag Weight (kg)", 25, 100, 50, step=5)
    with a3:
        derogation = st.selectbox("Nitrates Limit", ["170 kg N/ha","220 kg N/ha (derogation)","250 kg N/ha (derogation)"], index=0)
        n_limit = 170 if "170" in derogation else (220 if "220" in derogation else 250)
        st.session_state["n_limit_header"] = int(n_limit)  # persist
    with a4:
        units_factor = st.number_input("Units N/acre per kg N/ha", 0.50, 1.20, 0.81, step=0.01)

    # Organic N
    st.markdown("#### Organic N (Slurry)")
    cols = st.columns(len(MONTHS))
    for i,m in enumerate(MONTHS):
        with cols[i]:
            st.session_state.slurry_gal_ac[m] = st.number_input(f"{m}\nGallons/acre", min_value=0.0,
                                                                value=float(st.session_state.slurry_gal_ac[m]),
                                                                step=100.0, key=f"sl_{m}")
    org_kgN_ha = {m: (st.session_state.slurry_gal_ac[m]/1000.0)*kgN_per_1000gal for m in MONTHS}
    org_units_ac = {m: org_kgN_ha[m]*units_factor for m in MONTHS}
    organic_df = pd.DataFrame({
        "Gallons/acre":[st.session_state.slurry_gal_ac[m] for m in MONTHS],
        "kg N/ha":[org_kgN_ha[m] for m in MONTHS],
        "Units N/acre":[org_units_ac[m] for m in MONTHS],
        "Total Organic kg N/ha YTD": list(ytd([org_kgN_ha[m] for m in MONTHS]))
    }, index=MONTHS).round(1)
    st.dataframe(organic_df, use_container_width=True, height=240)

    # Chemical N
    st.markdown("#### Chemical Fertiliser")
    row1 = st.columns(len(MONTHS))
    for i,m in enumerate(MONTHS):
        with row1[i]:
            st.session_state.fert_choice[m] = st.selectbox(f"{m} Product", list(FERTILISERS.keys()),
                                                           index=list(FERTILISERS).index(st.session_state.fert_choice[m]),
                                                           key=f"fb_{m}")
    row2 = st.columns(len(MONTHS))
    for i,m in enumerate(MONTHS):
        with row2[i]:
            st.session_state.chem_target_kgN_ha[m] = st.number_input(f"{m} Target kg N/ha", 0.0, 400.0,
                                                                     value=float(st.session_state.chem_target_kgN_ha[m]),
                                                                     step=1.0, key=f"tg_{m}")
    chem_rows = []
    for m in MONTHS:
        fert = st.session_state.fert_choice[m]; N_frac,_ = FERTILISERS[fert]; target = st.session_state.chem_target_kgN_ha[m]
        if N_frac>0:
            kg_fert_per_ha = target / N_frac
            kg_fert_per_ac = kg_fert_per_ha * ACRE_TO_HA
            bags_per_ac = kg_fert_per_ac / bag_weight
        else:
            bags_per_ac = 0.0
        chem_rows.append({"Product": fert, "Target kg N/ha": round(target,1),
                          "Bags/acre": round(bags_per_ac,2), "Units N/acre": round(target*units_factor,1)})
    chem_df = pd.DataFrame(chem_rows, index=MONTHS)
    chem_df["Total Chemical kg N/ha YTD"] = ytd(chem_df["Target kg N/ha"])
    st.dataframe(chem_df, use_container_width=True, height=320)

    # Totals & Compliance
    st.markdown("#### Totals & Compliance")
    total_mo = [organic_df.loc[m,"kg N/ha"] + chem_df.loc[m,"Target kg N/ha"] for m in MONTHS]
    total_df = pd.DataFrame({
        "Month": MONTHS,
        "Organic kg N/ha":[organic_df.loc[m,"kg N/ha"] for m in MONTHS],
        "Chemical kg N/ha":[chem_df.loc[m,"Target kg N/ha"] for m in MONTHS],
        "Total kg N/ha": total_mo,
        "Total kg N/ha (YTD)": list(ytd(total_mo)),
    })
    cA,cB = st.columns([2,1])
    with cA:
        st.plotly_chart(px.line(total_df, x="Month", y=["Organic kg N/ha","Chemical kg N/ha","Total kg N/ha"],
                                markers=True, title="Monthly N Applications (kg N/ha)"),
                        use_container_width=True, height=360)
    with cB:
        ytd_total = float(total_df["Total kg N/ha (YTD)"].iloc[-1])
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("YTD Total N (kg N/ha)", f"{ytd_total:.1f}")
        n_limit = 170 if "170" in derogation else (220 if "220" in derogation else 250)
        st.metric("Regulatory Limit (kg N/ha)", f"{n_limit}")
        st.write("🟢 Within limit" if ytd_total<=n_limit else "🔴 Above limit")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("#### Export")
    excel_download_button({"Organic N": organic_df, "Chemical N": chem_df, "Totals": total_df.set_index("Month")},
                          filename=f"NPlan_{datetime.now().strftime('%Y%m%d')}.xlsx")

    # Save full plan cache for other tabs
    st.session_state._plan_cache = {
        "n_limit": n_limit,
        "organic_by_mo": [round(organic_df.loc[m,"kg N/ha"],1) for m in MONTHS],
        "chemical_by_mo": [round(chem_df.loc[m,"Target kg N/ha"],1) for m in MONTHS],
        "ytd": ytd_total, "area_ha": float(area_ha),
        "usage": paddock_usage, "slurry_available": float(avail_slurry),
    }

elif section == "Grass & Crops":
    st.markdown("### Grass & Crops Planner")
    g1,g2,g3 = st.columns(3)
    with g1: rot_days = st.number_input("Target Rotation Length (days)", 10, 60, 21)
    with g2: avg_cover = st.number_input("Average Farm Cover (kg DM/ha)", 300, 3000, 1100, step=50)
    with g3: growth = st.number_input("Growth Rate (kg DM/ha/day)", 0, 120, 50)
    st.caption("Add paddocks with area and current cover; the tool computes grazing order and next graze date.")
    if "paddocks_df" not in st.session_state:
        st.session_state.paddocks_df = pd.DataFrame({
            "Paddock":["P1","P2","P3","P4"], "Area (ha)":[3.2,2.8,2.5,3.0],
            "Cover (kg DM/ha)":[1100,950,1250,1000], "Priority":["Med","Low","High","Med"]
        })
    pad_df = st.dataframe(st.session_state.paddocks_df, use_container_width=True, height=240)
    df = st.session_state.paddocks_df.copy()
    df["Graze Score"] = 0.6*df["Cover (kg DM/ha)"] + 0.4*df["Area (ha)"]*100
    df["Days to Target"] = np.maximum(0, (1500 - df["Cover (kg DM/ha)"])/max(growth,1))
    df["Next Graze (est)"] = [(datetime.now()+timedelta(days=float(x))).date() for x in df["Days to Target"]]
    order_df = df.sort_values(["Graze Score","Cover (kg DM/ha)"], ascending=False)[
        ["Paddock","Area (ha)","Cover (kg DM/ha)","Priority","Days to Target","Next Graze (est)"]]
    st.plotly_chart(px.bar(order_df, x="Paddock", y="Cover (kg DM/ha)", title="Paddock Covers"),
                    use_container_width=True, height=350)
    st.dataframe(order_df, use_container_width=True, height=280)
    st.session_state._grass_cache = {"rotation_days": rot_days, "avg_cover": avg_cover,
                                     "growth": growth, "table": order_df}

elif section == "Dairy":
    st.markdown("### Dairy: Milk Yield Forecasting (OLS) & THI Check")
    st.caption("Upload daily records with columns: Date, MilkYield (L), Feed (kgDM), DIM, TempC, RH.")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up: dairy = pd.read_csv(up)
    else:
        days = pd.date_range("2025-03-01", periods=90, freq="D")
        rng = np.random.default_rng(42)
        feed = 6 + rng.normal(0,0.5,len(days)); dim = np.arange(50, 50+len(days))
        temp = 12 + 8*np.sin(np.linspace(0,3*np.pi,len(days))) + rng.normal(0,1,len(days))
        rh = 70 + 10*np.sin(np.linspace(0,2*np.pi,len(days)+3))[:len(days)]
        thi = np.array([calc_thi(temp[i], rh[i]) for i in range(len(days))])
        stress = np.maximum(0, thi-68)
        milk = 12 + 1.2*feed - 0.02*dim - 0.03*stress + rng.normal(0,0.5,len(days))
        dairy = pd.DataFrame({"Date":days, "MilkYield":milk.round(2), "Feed":feed.round(2),
                              "DIM":dim, "TempC":temp.round(1), "RH":rh.round(1)})
    dairy["THI"] = [calc_thi(x,y) for x,y in zip(dairy["TempC"], dairy["RH"])]
    st.dataframe(dairy.tail(15), use_container_width=True, height=260)
    st.plotly_chart(px.line(dairy, x="Date", y="MilkYield", title="Daily Milk Yield"),
                    use_container_width=True, height=350)
    X = dairy[["Feed","DIM","THI"]].values.astype(float); y = dairy["MilkYield"].values.astype(float)
    beta, r2 = fit_ols_numpy(X,y)
    st.markdown(f"**OLS model:** Milk = β₀ + β₁·Feed + β₂·DIM + β₃·THI  |  R² = **{r2:.2f}**")
    coef_df = pd.DataFrame({"Coefficient":["Intercept","Feed","DIM","THI"],"Value":np.round(beta,3)})
    st.dataframe(coef_df, use_container_width=True, height=170)
    f1,f2,f3 = st.columns(3)
    with f1: feed_next = st.number_input("Planned Feed (kgDM)", 2.0, 15.0, 6.5, step=0.1)
    with f2: temp_next = st.number_input("Expected Temp (°C)", -5.0, 35.0, 16.0, step=0.5)
    with f3: rh_next   = st.number_input("Expected RH (%)", 10.0, 100.0, 70.0, step=1.0)
    thi_next = calc_thi(temp_next, rh_next)
    future_dim = np.arange(int(dairy["DIM"].iloc[-1])+1, int(dairy["DIM"].iloc[-1])+31)
    Xf = np.c_[np.full_like(future_dim, feed_next, dtype=float),
               future_dim.astype(float),
               np.full_like(future_dim, thi_next, dtype=float)]
    milk_pred = predict_ols_numpy(beta, Xf)
    fut_df = pd.DataFrame({"Day":np.arange(1,31), "DIM":future_dim, "PredMilk":milk_pred})
    st.plotly_chart(px.line(fut_df, x="Day", y="PredMilk", title="30-Day Forecast (L/cow/day)"),
                    use_container_width=True, height=330)
    st.session_state._dairy_cache = {"beta":beta.tolist(), "r2":r2, "next_thi":thi_next, "forecast":fut_df}

elif section == "Livestock":
    st.markdown("### Livestock: Growth & Health Analytics")
    st.caption("Upload weights with columns: AnimalID, Date, Weight (kg), Breed, Group.")
    up = st.file_uploader("Upload CSV", type=["csv"], key="lv")
    if up: lv = pd.read_csv(up)
    else:
        rng = np.random.default_rng(7)
        ids = np.repeat([f"C{i:03d}" for i in range(1,16)], 4)
        dates = []; start = datetime(2025,3,1)
        for i in range(15): dates += [(start + timedelta(days=d)).date().isoformat() for d in [0,30,60,90]]
        weights = np.concatenate([np.linspace(180, 260, 4) + np.cumsum(rng.normal(0,2,4)) for _ in range(15)])
        breeds = np.repeat(np.random.choice(["HF","JE","AA"], size=15), 4)
        groups = np.repeat(np.random.choice(["Heifer","Steer"], size=15), 4)
        lv = pd.DataFrame({"AnimalID":ids,"Date":dates,"Weight":np.round(weights,1),"Breed":breeds,"Group":groups})
    lv["Date"] = pd.to_datetime(lv["Date"])
    st.dataframe(lv.sort_values(["AnimalID","Date"]).tail(20), use_container_width=True, height=280)
    adg_list = []
    for aid, g in lv.groupby("AnimalID"):
        g = g.sort_values("Date"); days = (g["Date"].iloc[-1]-g["Date"].iloc[0]).days
        if days<=0: continue
        adg = (g["Weight"].iloc[-1]-g["Weight"].iloc[0]) / days
        adg_list.append({"AnimalID":aid,"StartW":g["Weight"].iloc[0],"EndW":g["Weight"].iloc[-1],"Days":days,"ADG (kg/d)":adg})
    adg_df = pd.DataFrame(adg_list).sort_values("ADG (kg/d)", ascending=False)
    st.dataframe(adg_df, use_container_width=True, height=260)
    st.plotly_chart(px.histogram(adg_df, x="ADG (kg/d)", nbins=20, title="ADG Distribution"),
                    use_container_width=True, height=330)
    st.session_state._livestock_cache = {"adg":adg_df}

elif section == "Weather & Soil":
    st.markdown("### Weather & Soil (Manual / File Inputs)")
    w1,w2,w3,w4 = st.columns(4)
    with w1: temp_c = st.number_input("Temp (°C)", -5.0, 35.0, 15.0, step=0.5)
    with w2: rh     = st.number_input("RH (%)", 10.0, 100.0, 70.0, step=1.0)
    with w3: pH     = st.number_input("Soil pH", 4.0, 8.5, 6.2, step=0.1)
    with w4: pkstat = st.selectbox("P/K Index", ["Low P/K","Adequate","High P/K"], index=1)
    thi_now = calc_thi(temp_c, rh); st.metric("Current THI", f"{thi_now:.1f}")
    st.write("**Soil Advisory:** " + ("Apply lime to lift pH ≥6.3; target slurry to low P/K paddocks." if (pH<6.2 or pkstat=='Low P/K') else "Indices adequate; maintain pH and monitor."))
    st.session_state._wx_cache = {"temp":temp_c,"rh":rh,"thi":thi_now,"pH":pH,"PK":pkstat}

# ========= NEW: Sensors (Camera & Edge AI) =========
elif section == "Sensors (Camera & Edge AI)":
    st.markdown("### Sensors: Device Camera & Edge/Offline Inference")
    st.caption("All processing is **local**. By default, images are not saved (EU-AI Act/GDPR-friendly).")
    save_frames = st.toggle("Allow local save of annotated frames (off = privacy-first)", value=False)
    task = st.selectbox("Select Analyzer", ["Leaf Stress (VARI/ExG + Lesions)", "Crack/Surface Stress (Canny + Skeleton)"], index=0)

    tab_photo, tab_realtime = st.tabs(["📸 Photo Capture", "🎥 Realtime Webcam (optional)"])

    # ---- Photo mode (works everywhere) ----
    with tab_photo:
        img_file = st.camera_input("Capture image", help="Use phone or laptop camera.")
        sensitivity = st.slider("Sensitivity / Edge Gain", 0.5, 2.0, 1.0, 0.05)
        if img_file is not None:
            rgb = _np_img_from_upload(img_file)
            if task.startswith("Leaf"):
                res = analyze_leaf_rgb(rgb, vari_thresh=0.02*sensitivity, lesion_canny=(int(60*sensitivity), int(140*sensitivity)))
                if not res:
                    st.error("Analysis failed (check dependencies).")
                else:
                    c1, c2 = st.columns([1.4, 1])
                    with c1:
                        st.image(res["overlay"], caption="Leaf VARI Heatmap Overlay (offline)", use_column_width=True)
                    with c2:
                        st.markdown("**Leaf Metrics (offline)**")
                        st.metric("Mean VARI", res["mean_VARI"])
                        st.metric("Lesion Density", res["lesion_density"])
                        st.metric("Stress Score (0–1)", res["stress_score"])
                        st.write(f"Status: **{res['status']}**")
                    if save_frames:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        outp = f"leaf_{ts}.png"; Image.fromarray(res["overlay"]).save(outp)
                        st.success(f"Saved annotated frame: `{outp}`")
            else:
                if not HAS_CV2: st.error("OpenCV required. Install: `pip install opencv-python`")
                else:
                    res = analyze_crack_rgb(rgb, canny=(int(100/sensitivity), int(200/sensitivity)), use_sato=True)
                    c1, c2 = st.columns([1.4, 1])
                    with c1:
                        st.image(res["overlay"], caption="Surface Stress Overlay (offline)", use_column_width=True)
                    with c2:
                        st.markdown("**Surface Metrics (offline)**")
                        st.metric("Edge Density", res["edge_density"])
                        st.metric("Skeleton Length (px)", res["skeleton_length"] if res["skeleton_length"] is not None else 0)
                        st.write(f"Risk: **{res['risk']}**")
                    if save_frames:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        outp = f"surface_{ts}.png"; Image.fromarray(res["overlay"]).save(outp)
                        st.success(f"Saved annotated frame: `{outp}`")

    # ---- Realtime mode (optional) ----
    with tab_realtime:
        if not HAS_WEBRTC or not HAS_CV2:
            st.warning("Realtime requires `streamlit-webrtc`, `av`, and `opencv-python`. Install and reload.")
        else:
            st.caption("Realtime runs **entirely local**. Close the stream to stop processing.")
            mode_desc = "Leaf" if task.startswith("Leaf") else "Crack"

            class EdgeTransformer(VideoTransformerBase):
                def __init__(self):
                    self.sensitivity = 1.0
                    self.task = mode_desc

                def recv(self, frame):
                    img = frame.to_ndarray(format="bgr24")
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    if self.task == "Leaf":
                        res = analyze_leaf_rgb(rgb, vari_thresh=0.02*self.sensitivity,
                                               lesion_canny=(int(60*self.sensitivity), int(140*self.sensitivity)))
                        out = res["overlay"]
                    else:
                        res = analyze_crack_rgb(rgb, canny=(int(100/self.sensitivity), int(200/self.sensitivity)), use_sato=True)
                        out = res["overlay"]
                    return av.VideoFrame.from_ndarray(cv2.cvtColor(out, cv2.COLOR_RGB2BGR), format="bgr24")

            rtc_config = RTCConfiguration({"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]})
            st.slider("Sensitivity (realtime)", 0.5, 2.0, 1.0, 0.05, key="rt_sens")
            ctx = webrtc_streamer(
                key="edge-rt",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=rtc_config,
                video_transformer_factory=EdgeTransformer,
                media_stream_constraints={"video": True, "audio": False},
            )
            if ctx.video_transformer:
                ctx.video_transformer.sensitivity = st.session_state["rt_sens"]
                ctx.video_transformer.task = "Leaf" if task.startswith("Leaf") else "Crack"

elif section == "Finance":
    st.markdown("### Finance: Costs, Returns & N Response")

    # Inputs
    f1, f2, f3, f4 = st.columns(4)
    with f1: milk_price = st.number_input("Milk Price (€/L)", 0.10, 2.00, 0.42, step=0.01)
    with f2: fert_cost  = st.number_input("Fertiliser Cost (€/kg N)", 0.10, 5.00, 1.45, step=0.05)
    with f3: herd_size  = st.number_input("Herd Size (cows)", 1, 10000, 120)
    with f4: conv_eff   = st.number_input("N→Milk Response (L/kg N)", 0.0, 25.0, 8.0, step=0.5)

    # --- Recompute N directly from current session values (no cache dependency) ---
    kgN1000 = float(st.session_state.get("kgN_per_1000gal", 10.6))
    area_ha = float(st.session_state.get("area_ha_header", 72.59))

    # Monthly organic N (slurry) and chemical N (targets) -> kg N/ha
    organic_by_mo  = [(float(st.session_state.slurry_gal_ac.get(m, 0.0))/1000.0)*kgN1000 for m in MONTHS]
    chemical_by_mo = [float(st.session_state.chem_target_kgN_ha.get(m, 0.0)) for m in MONTHS]

    totalN_kg_per_ha = float(np.sum(organic_by_mo) + np.sum(chemical_by_mo))
    totalN_kg = totalN_kg_per_ha * area_ha

    # Economics (whole farm)
    fert_cost_eur = totalN_kg * fert_cost
    added_milk_L  = totalN_kg * conv_eff
    added_rev_eur = added_milk_L * milk_price
    margin_eur    = added_rev_eur - fert_cost_eur

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Total N Plan (kg N)", f"{totalN_kg:,.0f}")
    with c2: st.metric("Fertiliser Cost (€)", f"{fert_cost_eur:,.0f}")
    with c3: st.metric("Est. Added Revenue (€)", f"{added_rev_eur:,.0f}")
    st.metric("Estimated Margin (€)", f"{margin_eur:,.0f}")

    # Response curve (€/ha) – reactive to milk_price & fert_cost
    Ngrid = np.linspace(0, 300, 61)
    euro_per_ha = (conv_eff * Ngrid) * milk_price - Ngrid * fert_cost
    curve = pd.DataFrame({"kg N/ha": Ngrid, "Margin €/ha": euro_per_ha})
    st.plotly_chart(px.line(curve, x="kg N/ha", y="Margin €/ha", title="Simple N Response Margin Curve"),
                    use_container_width=True, height=350)

    # Optional: show the N used for the calculation so it's obvious why zeros occur
    with st.expander("Show N inputs used"):
        dbg = pd.DataFrame({"Month": MONTHS,
                            "Organic kg N/ha": np.round(organic_by_mo, 1),
                            "Chemical kg N/ha": np.round(chemical_by_mo, 1)})
        dbg["Total kg N/ha"] = dbg["Organic kg N/ha"] + dbg["Chemical kg N/ha"]
        st.dataframe(dbg, use_container_width=True, height=260)
        st.caption(f"Area: {area_ha:.2f} ha • YTD total kg N/ha: {totalN_kg_per_ha:.1f}")

    # Keep for Reports (doesn't affect finance KPIs)
    st.session_state._finance_cache = {"price": milk_price, "costN": fert_cost, "margin_curve": curve}


elif section == "AI Advisor":
    st.markdown("### Contextual AI Advisor")
    plan = st.session_state.get("_plan_cache") or get_current_plan_from_state()
    grass = st.session_state.get("_grass_cache", {})
    dairy = st.session_state.get("_dairy_cache", {})
    wx = st.session_state.get("_wx_cache", {})
    default_prompt = (
        f"Plan details:\n- N limit: {plan.get('n_limit','n/a')} kg N/ha; YTD total N: {plan.get('ytd','n/a')} kg N/ha\n"
        f"- Organic N by month: {plan.get('organic_by_mo','[]')}\n- Chemical N by month: {plan.get('chemical_by_mo','[]')}\n"
        f"- Grass rotation: {grass.get('rotation_days','n/a')} d; cover {grass.get('avg_cover','n/a')} kg DM/ha; growth {grass.get('growth','n/a')} kg DM/ha/d\n"
        f"- Weather/Soil: THI {wx.get('thi','n/a')}; pH {wx.get('pH','n/a')}\n"
        f"- Dairy model R² {dairy.get('r2','n/a')}\n\n"
        "Advise for next 2–4 weeks on slurry/chemical N timing, product choice, grazing order, heat-stress mitigations, and compliance risks. Max 6 bullets.")
    query = st.text_area("Prompt (editable):", value=default_prompt, height=220)
    if st.button("Generate Advice"):
        with st.spinner("Generating advice..."): st.write(contextual_ai(query, plan))
    st.caption(f"AI backend: {'OpenAI ('+_openai_mode+')' if USE_OPENAI else 'Rule-based (offline)'}")

elif section == "Reports":
    st.markdown("### Reports & Exports")
    exports: Dict[str, pd.DataFrame] = {}

    # Always build a plan (cache or live state) so exports work even if Nitrogen tab wasn't opened
    plan = st.session_state.get("_plan_cache") or get_current_plan_from_state()
    if plan:
        exports["Organic N (kg ha)"]  = pd.DataFrame({"kg N/ha": plan["organic_by_mo"]}, index=MONTHS)
        exports["Chemical N (kg ha)"] = pd.DataFrame({"kg N/ha": plan["chemical_by_mo"]}, index=MONTHS)
        tot = (np.array(plan["organic_by_mo"]) + np.array(plan["chemical_by_mo"]))
        exports["Total N (kg ha)"]    = pd.DataFrame({"kg N/ha": tot}, index=MONTHS)

    if "_dairy_cache" in st.session_state:
        exports["Dairy Forecast"] = st.session_state["_dairy_cache"]["forecast"].set_index("Day")
    if "_livestock_cache" in st.session_state:
        exports["Livestock ADG"] = st.session_state["_livestock_cache"]["adg"].set_index("AnimalID")
    if "_grass_cache" in st.session_state:
        exports["Grazing Order"] = st.session_state["_grass_cache"]["table"].set_index("Paddock")

    if exports:
        excel_download_button(exports, filename=f"FarmSuite_{datetime.now().strftime('%Y%m%d')}.xlsx")
    else:
        st.warning("No data yet. Visit other tabs to generate content for export.")

else:
    st.markdown("### Settings")
    st.write("Future: units, default prices, data retention options, themes.")
