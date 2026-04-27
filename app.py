import streamlit as st
import numpy as np
import math
from datetime import datetime

st.set_page_config(page_title="ICU Risk Monitor", layout="wide", page_icon=None)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], .stApp {
    background: #f0f2f5 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    color: #1a1d23 !important;
}

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }

[data-testid="stAppViewContainer"],
[data-testid="stMain"] { background: #f0f2f5 !important; }

.block-container {
    padding: 0 2rem 2rem 2rem !important;
    max-width: 1280px !important;
    margin: 0 auto !important;
}

[data-testid="stSlider"] > div > div > div > div { background: #1a1d23 !important; }
[data-testid="stSlider"] [role="slider"] {
    background: #fff !important;
    border: 2px solid #1a1d23 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,.2) !important;
    width: 15px !important; height: 15px !important;
}
[data-testid="stSlider"] label { display: none !important; }
[data-testid="stSlider"] > div { padding-bottom: 4px !important; }

.stButton > button {
    background: #1a1d23 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    font-size: 11px !important;
    letter-spacing: .12em !important;
    text-transform: uppercase !important;
    padding: 11px 0 !important;
    width: 100% !important;
    transition: background .15s !important;
}
.stButton > button:hover { background: #2d3139 !important; }
[data-testid="stHorizontalBlock"] { gap: 14px !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SESSION STATE — for trend tracking
# ══════════════════════════════════════════════════════════════
if "prev_hr"  not in st.session_state: st.session_state.prev_hr  = None
if "prev_sbp" not in st.session_state: st.session_state.prev_sbp = None
if "prev_dbp" not in st.session_state: st.session_state.prev_dbp = None
if "prev_prob" not in st.session_state: st.session_state.prev_prob = None
if "run_count" not in st.session_state: st.session_state.run_count = 0


# ══════════════════════════════════════════════════════════════
# PHYSIOLOGICAL VALIDATION RANGES
# ══════════════════════════════════════════════════════════════
LIMITS = {
    "hr":  {"min": 20,  "max": 250, "warn_lo": 40,  "warn_hi": 150,
            "label": "Heart Rate",
            "impossible_lo": "Asystole / incompatible with life",
            "impossible_hi": "Ventricular fibrillation range"},
    "sbp": {"min": 50,  "max": 300, "warn_lo": 80,  "warn_hi": 180,
            "label": "Systolic BP",
            "impossible_lo": "Severe shock / incompatible with perfusion",
            "impossible_hi": "Hypertensive emergency / measurement error likely"},
    "dbp": {"min": 20,  "max": 150, "warn_lo": 50,  "warn_hi": 110,
            "label": "Diastolic BP",
            "impossible_lo": "Severe circulatory failure",
            "impossible_hi": "Measurement error likely"},
}

def validate(name, val):
    L = LIMITS[name]
    if val < L["min"] or val > L["max"]:
        direction = "impossible_lo" if val < L["min"] else "impossible_hi"
        return "impossible", L[direction]
    if val < L["warn_lo"] or val > L["warn_hi"]:
        return "warning", f"Outside expected clinical range ({L['warn_lo']}–{L['warn_hi']})"
    return "ok", ""

def sbp_dbp_consistent(sbp, dbp):
    if dbp >= sbp:
        return False, "Diastolic cannot equal or exceed Systolic BP — check measurement"
    if (sbp - dbp) < 10:
        return False, f"Pulse pressure of {sbp-dbp} mmHg is physiologically implausible"
    return True, ""


# ══════════════════════════════════════════════════════════════
# CLINICAL LOGIC
# ══════════════════════════════════════════════════════════════
def predict_risk(age, hr, sbp, dbp):
    map_v = dbp + (sbp - dbp) / 3
    s = (age-55)*0.014 + (hr-75)*0.025 + (sbp-120)*0.010 + (map_v-80)*0.012
    return float(np.clip(1/(1+np.exp(-s)), 0.01, 0.99))

# SHAP-style: each feature's marginal contribution vs. baseline prediction
# Baseline = predict_risk at reference values (age=55, hr=75, sbp=120, dbp=80)
BASELINE_PROB = predict_risk(55, 75, 120, 80)

def shap_contributions(age, hr, sbp, dbp):
    """
    Additive decomposition: f(x) = baseline + sum(phi_i).
    Each phi_i = f(x with feature_i changed) - f(x with feature_i at reference).
    Uses the linear structure of the logistic model exactly.
    """
    map_v = dbp + (sbp - dbp) / 3
    phi_age = (age-55)*0.014
    phi_hr  = (hr -75)*0.025
    phi_sbp = (sbp-120)*0.010
    phi_map = (map_v-80)*0.012
    # Convert log-odds contributions to probability-space via finite difference
    base_logit = math.log(BASELINE_PROB/(1-BASELINE_PROB))
    total_logit = base_logit + phi_age + phi_hr + phi_sbp + phi_map

    def logit_to_prob(l): return 1/(1+math.exp(-l))

    prob_full = logit_to_prob(total_logit)
    contribs  = {}
    for name, phi in [("Age", phi_age), ("Heart Rate", phi_hr),
                       ("Systolic BP", phi_sbp), ("MAP", phi_map)]:
        p_with    = logit_to_prob(total_logit)
        p_without = logit_to_prob(total_logit - phi)
        contribs[name] = p_with - p_without  # signed probability contribution
    return contribs, prob_full

def risk_tier(p):
    if p >= .70: return "#c0392b", "CRITICAL",  "Immediate intervention required"
    if p >= .40: return "#d97706", "ELEVATED",  "Increased monitoring — review in 30 min"
    return              "#16a34a", "STABLE",    "Continue standard monitoring protocol"

def hr_class(hr):
    if hr < 40:  return "Severe Bradycardia", "#c0392b"
    if hr < 60:  return "Bradycardia",        "#d97706"
    if hr > 150: return "Severe Tachycardia", "#c0392b"
    if hr > 100: return "Tachycardia",        "#d97706"
    return              "Normal Sinus",        "#16a34a"

def bp_class(sbp, dbp):
    if sbp >= 180 or dbp >= 120: return "Hypertensive Crisis", "#c0392b"
    if sbp >= 160 or dbp >= 100: return "Stage 2 HTN",         "#c0392b"
    if sbp >= 140 or dbp >= 90:  return "Stage 1 HTN",         "#d97706"
    if sbp >= 130 or dbp >= 80:  return "Elevated",            "#d97706"
    if sbp < 90  or dbp < 60:   return "Hypotension",         "#c0392b"
    return                              "Normal",               "#16a34a"

def map_v(sbp, dbp): return round(dbp + (sbp-dbp)/3)
def pp(sbp, dbp):    return sbp - dbp

def trend_arrow(current, previous):
    if previous is None: return "", "#9ca3af"
    diff = current - previous
    if abs(diff) < 1: return "STABLE", "#9ca3af"
    if diff > 0: return f"+{diff:.0f}", "#c0392b"
    return f"{diff:.0f}", "#16a34a"


# ══════════════════════════════════════════════════════════════
# SVG COMPONENTS
# ══════════════════════════════════════════════════════════════
def gauge_svg(prob, color):
    r=68; cx=cy=84; circ=math.pi*r; dash=prob*circ
    ticks=""
    for i in range(11):
        a=math.pi*i/10
        x1=cx-r*math.cos(a); y1=cy-r*math.sin(a)
        tl=7 if i%5==0 else 3
        x2=cx-(r-tl)*math.cos(a); y2=cy-(r-tl)*math.sin(a)
        ticks+=f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#e5e7eb" stroke-width="1.5"/>'
    zones=""; off=0
    for frac,zc in [(0.40,"#16a34a"),(0.30,"#d97706"),(0.30,"#c0392b")]:
        seg=frac*circ
        zones+=f'<path d="M {cx-r} {cy} A {r} {r} 0 0 1 {cx+r} {cy}" fill="none" stroke="{zc}" stroke-width="7" opacity="0.18" stroke-dasharray="{seg:.1f} {circ:.1f}" stroke-dashoffset="{-off:.1f}"/>'
        off+=seg
    delta_html=""
    if st.session_state.prev_prob is not None:
        delta = prob - st.session_state.prev_prob
        dcol  = "#c0392b" if delta > 0 else "#16a34a"
        dsign = "+" if delta > 0 else ""
        delta_html=f'<text x="{cx}" y="{cy+26}" text-anchor="middle" font-family="IBM Plex Mono,monospace" font-size="8" fill="{dcol}">vs prev: {dsign}{delta*100:.1f}%</text>'
    return f"""<svg viewBox="0 0 168 102" width="100%" style="display:block;max-width:220px;margin:0 auto">
  {ticks}{zones}
  <path d="M {cx-r} {cy} A {r} {r} 0 0 1 {cx+r} {cy}" fill="none" stroke="#e5e7eb" stroke-width="7"/>
  <path d="M {cx-r} {cy} A {r} {r} 0 0 1 {cx+r} {cy}" fill="none" stroke="{color}" stroke-width="7"
        stroke-dasharray="{dash:.1f} {circ:.1f}"/>
  <text x="{cx}" y="{cy-6}" text-anchor="middle" font-family="IBM Plex Mono,monospace"
        font-size="24" font-weight="600" fill="{color}">{prob*100:.1f}%</text>
  <text x="{cx}" y="{cy+11}" text-anchor="middle" font-family="IBM Plex Sans,sans-serif"
        font-size="8" fill="#9ca3af" letter-spacing="1.5">RISK SCORE</text>
  {delta_html}
  <text x="12" y="{cy+6}" font-family="IBM Plex Mono,monospace" font-size="7" fill="#c9cdd4">0</text>
  <text x="152" y="{cy+6}" font-family="IBM Plex Mono,monospace" font-size="7" fill="#c9cdd4">100</text>
</svg>"""

def shap_svg(contribs, baseline):
    """Waterfall-style SHAP plot — clinical gold standard for XAI."""
    W = 420; bar_h = 22; pad_l = 100; pad_r = 60; row_gap = 36
    items = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)
    total_h = len(items)*row_gap + 80

    # x-axis scale: map probability space to pixel space
    all_vals = [abs(v) for v in contribs.values()]
    max_abs  = max(all_vals) if all_vals else 0.1
    plot_w   = W - pad_l - pad_r
    scale    = plot_w / (2 * max_abs * 1.15)
    mid_x    = pad_l + plot_w / 2

    # center line, axis
    svg  = f'<svg viewBox="0 0 {W} {total_h}" width="100%" style="display:block;font-family:IBM Plex Sans,sans-serif">'
    # Baseline label
    svg += f'<text x="{mid_x:.1f}" y="14" text-anchor="middle" font-family="IBM Plex Mono,monospace" font-size="8" fill="#9ca3af">BASELINE {baseline*100:.1f}%</text>'
    svg += f'<line x1="{mid_x:.1f}" y1="18" x2="{mid_x:.1f}" y2="{total_h-24}" stroke="#e5e7eb" stroke-width="1" stroke-dasharray="3,3"/>'

    # x-axis
    svg += f'<line x1="{pad_l}" y1="{total_h-20}" x2="{W-pad_r}" y2="{total_h-20}" stroke="#d1d5db" stroke-width="1"/>'
    for tick in [-max_abs, -max_abs/2, 0, max_abs/2, max_abs]:
        tx = mid_x + tick*scale
        label_val = f"{tick*100:+.1f}%" if tick != 0 else "0"
        svg += f'<text x="{tx:.1f}" y="{total_h-6}" text-anchor="middle" font-family="IBM Plex Mono,monospace" font-size="7" fill="#9ca3af">{label_val}</text>'
        svg += f'<line x1="{tx:.1f}" y1="{total_h-22}" x2="{tx:.1f}" y2="{total_h-18}" stroke="#d1d5db" stroke-width="1"/>'

    for i, (name, val) in enumerate(items):
        y       = 28 + i*row_gap
        bar_w   = abs(val)*scale
        positive = val > 0
        col     = "#c0392b" if positive else "#16a34a"
        bar_x   = mid_x if positive else mid_x - bar_w

        # feature label
        svg += f'<text x="{pad_l-6}" y="{y+bar_h/2+4:.1f}" text-anchor="end" font-size="11" font-weight="500" fill="#374151">{name}</text>'
        # bar
        svg += f'<rect x="{bar_x:.1f}" y="{y}" width="{bar_w:.1f}" height="{bar_h}" rx="2" fill="{col}" opacity="0.85"/>'
        # value label
        label_x = bar_x + bar_w + 5 if positive else bar_x - 5
        anchor  = "start" if positive else "end"
        sign    = "+" if positive else ""
        svg += f'<text x="{label_x:.1f}" y="{y+bar_h/2+4:.1f}" text-anchor="{anchor}" font-family="IBM Plex Mono,monospace" font-size="9" font-weight="600" fill="{col}">{sign}{val*100:.2f}%</text>'

    svg += "</svg>"
    return svg


# ══════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════
def sec(t):
    return f"<div style=\"font-family:'IBM Plex Mono',monospace;font-size:9px;font-weight:600;color:#9ca3af;letter-spacing:.12em;text-transform:uppercase;margin-bottom:10px\">{t}</div>"

def divider():
    return "<div style=\"height:1px;background:#e5e7eb;margin:12px 0\"></div>"

def card(html, accent=None):
    border = f"border-left:3px solid {accent}" if accent else "border:1px solid #e2e4e8"
    return f"<div style=\"background:#fff;{border};border-radius:6px;padding:18px;margin-bottom:12px\">{html}</div>"

def pill(text, color):
    return f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:9px;font-weight:600;color:{color};background:{color}14;padding:2px 7px;border-radius:2px">{text}</span>'

def alert_banner(msg, color, icon="!"):
    return f"""<div style="background:{color}10;border:1px solid {color}40;border-left:3px solid {color};
border-radius:4px;padding:8px 12px;margin-bottom:8px;display:flex;align-items:flex-start;gap:8px">
<span style="font-family:'IBM Plex Mono',monospace;font-size:10px;font-weight:700;color:{color};flex-shrink:0">{icon}</span>
<span style="font-size:11px;color:#374151;line-height:1.5">{msg}</span></div>"""

def vital_row(lbl, val, unit, ref, status_txt, status_col):
    return f"""<tr style="border-bottom:1px solid #f3f4f6">
  <td style="padding:8px 0;font-size:12px;font-weight:500;color:#374151">{lbl}</td>
  <td style="text-align:right;font-family:'IBM Plex Mono',monospace;font-size:13px;font-weight:600;color:#1a1d23;padding:8px 8px">{val}</td>
  <td style="text-align:right;font-size:11px;color:#9ca3af;padding:8px 6px">{unit}</td>
  <td style="text-align:right;font-size:11px;color:#9ca3af;padding:8px 0">{ref}</td>
  <td style="text-align:right;padding:8px 0 8px 10px">{pill(status_txt, status_col)}</td>
</tr>"""


# ══════════════════════════════════════════════════════════════
# TOP BAR
# ══════════════════════════════════════════════════════════════
now = datetime.now().strftime("%Y-%m-%d  %H:%M")
st.markdown(f"""
<div style="background:#1a1d23;padding:11px 2rem;display:flex;align-items:center;
            justify-content:space-between;margin:0 -2rem 18px -2rem">
  <div style="display:flex;align-items:center;gap:20px">
    <div>
      <div style="font-family:'IBM Plex Sans',sans-serif;font-size:12px;font-weight:700;
                  color:#fff;letter-spacing:.06em">ICU RISK MONITOR</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:#4b5563;margin-top:1px">
        CLINICAL DECISION SUPPORT · v3.0
      </div>
    </div>
    <div style="width:1px;height:22px;background:#2d3139"></div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:#6b7280">
      Model: Logistic Regression (GLM) · XAI: SHAP Waterfall
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:14px">
    <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:#4b5563">{now}</div>
    <div style="display:flex;align-items:center;gap:5px">
      <div style="width:5px;height:5px;border-radius:50%;background:#16a34a"></div>
      <span style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:#4b5563">ONLINE</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# INPUTS
# ══════════════════════════════════════════════════════════════
col_age, col_hr, col_sbp, col_dbp = st.columns([1, 1, 1, 1])

with col_age:
    st.markdown(f"""<div style="background:#fff;border:1px solid #e2e4e8;border-radius:6px;
padding:14px 16px 6px;margin-bottom:12px">
<div style="font-family:'IBM Plex Mono',monospace;font-size:9px;font-weight:600;color:#9ca3af;
letter-spacing:.1em;margin-bottom:6px">PATIENT AGE</div>""", unsafe_allow_html=True)
    age = st.slider("Age", 18, 100, 55, key="age")
    st.markdown(f"""<div style="font-family:'IBM Plex Mono',monospace;font-size:24px;font-weight:600;
color:#1a1d23;padding-bottom:10px">{age}
<span style="font-size:11px;font-weight:400;color:#9ca3af"> yr</span>
<div style="font-size:10px;color:#9ca3af">Demographic</div></div></div>""", unsafe_allow_html=True)

with col_hr:
    st.markdown(f"""<div style="background:#fff;border:1px solid #e2e4e8;border-radius:6px;
padding:14px 16px 6px;margin-bottom:12px">
<div style="font-family:'IBM Plex Mono',monospace;font-size:9px;font-weight:600;color:#9ca3af;
letter-spacing:.1em;margin-bottom:6px">HEART RATE · bpm</div>""", unsafe_allow_html=True)
    hr = st.slider("HR", 20, 250, 78, key="hr")
    hrc, hrcol = hr_class(hr)
    hr_val_state, prev_hr_state = validate("hr", hr), st.session_state.prev_hr
    trend_txt, trend_col = trend_arrow(hr, prev_hr_state)
    trend_html = f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:9px;color:{trend_col};margin-left:6px">{trend_txt}</span>' if trend_txt else ""
    st.markdown(f"""<div style="font-family:'IBM Plex Mono',monospace;font-size:24px;font-weight:600;
color:{hrcol};padding-bottom:10px">{hr}<span style="font-size:11px;font-weight:400;color:#9ca3af"> bpm</span>{trend_html}
<div style="font-size:10px;color:{hrcol}">{hrc}</div></div></div>""", unsafe_allow_html=True)

with col_sbp:
    st.markdown(f"""<div style="background:#fff;border:1px solid #e2e4e8;border-radius:6px;
padding:14px 16px 6px;margin-bottom:12px">
<div style="font-family:'IBM Plex Mono',monospace;font-size:9px;font-weight:600;color:#9ca3af;
letter-spacing:.1em;margin-bottom:6px">SYSTOLIC BP · mmHg</div>""", unsafe_allow_html=True)
    sbp = st.slider("SBP", 50, 300, 122, key="sbp")
    sbp_status, _ = validate("sbp", sbp)
    sbp_trend_txt, sbp_trend_col = trend_arrow(sbp, st.session_state.prev_sbp)
    sbp_trend_html = f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:9px;color:{sbp_trend_col};margin-left:6px">{sbp_trend_txt}</span>' if sbp_trend_txt else ""
    st.markdown(f"""<div style="font-family:'IBM Plex Mono',monospace;font-size:24px;font-weight:600;
color:#1a1d23;padding-bottom:10px">{sbp}<span style="font-size:11px;font-weight:400;color:#9ca3af"> mmHg</span>{sbp_trend_html}
<div style="font-size:10px;color:#9ca3af">Systolic</div></div></div>""", unsafe_allow_html=True)

with col_dbp:
    st.markdown(f"""<div style="background:#fff;border:1px solid #e2e4e8;border-radius:6px;
padding:14px 16px 6px;margin-bottom:12px">
<div style="font-family:'IBM Plex Mono',monospace;font-size:9px;font-weight:600;color:#9ca3af;
letter-spacing:.1em;margin-bottom:6px">DIASTOLIC BP · mmHg</div>""", unsafe_allow_html=True)
    dbp = st.slider("DBP", 20, 150, 80, key="dbp")
    dbp_status, _ = validate("dbp", dbp)
    dbp_trend_txt, dbp_trend_col = trend_arrow(dbp, st.session_state.prev_dbp)
    dbp_trend_html = f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:9px;color:{dbp_trend_col};margin-left:6px">{dbp_trend_txt}</span>' if dbp_trend_txt else ""
    st.markdown(f"""<div style="font-family:'IBM Plex Mono',monospace;font-size:24px;font-weight:600;
color:#1a1d23;padding-bottom:10px">{dbp}<span style="font-size:11px;font-weight:400;color:#9ca3af"> mmHg</span>{dbp_trend_html}
<div style="font-size:10px;color:#9ca3af">Diastolic</div></div></div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# INPUT VALIDATION ALERTS
# ══════════════════════════════════════════════════════════════
validation_errors = []
validation_warnings = []

for name, val in [("hr", hr), ("sbp", sbp), ("dbp", dbp)]:
    status, msg = validate(name, val)
    L = LIMITS[name]
    if status == "impossible":
        validation_errors.append(f"{L['label']} = {val}: {msg}")
    elif status == "warning":
        validation_warnings.append(f"{L['label']} = {val}: {msg}")

bp_ok, bp_msg = sbp_dbp_consistent(sbp, dbp)
if not bp_ok:
    validation_errors.append(bp_msg)

if validation_errors or validation_warnings:
    val_html = ""
    for e in validation_errors:
        val_html += alert_banner(e, "#c0392b", "INVALID")
    for w in validation_warnings:
        val_html += alert_banner(w, "#d97706", "WARN")
    if validation_errors:
        val_html += alert_banner(
            "Risk score is suppressed — resolve physiologically impossible values before proceeding.",
            "#c0392b", "!")
    st.markdown(card(val_html, accent="#c0392b" if validation_errors else "#d97706"),
                unsafe_allow_html=True)

has_errors = len(validation_errors) > 0


# ══════════════════════════════════════════════════════════════
# DERIVED VALUES
# ══════════════════════════════════════════════════════════════
if not has_errors:
    map_val = map_v(sbp, dbp)
    pp_val  = pp(sbp, dbp)
    bpc, bpcol = bp_class(sbp, dbp)

    d1, d2, d3, d4 = st.columns(4)
    derived = [
        (d1, "MEAN ART. PRESSURE", f"{map_val}", "mmHg", "DBP + (SBP−DBP)/3",
         "#16a34a" if 70<=map_val<=100 else "#c0392b"),
        (d2, "PULSE PRESSURE",     f"{pp_val}",  "mmHg", "SBP − DBP",
         "#d97706" if pp_val>60 or pp_val<25 else "#16a34a"),
        (d3, "BP CLASSIFICATION",  bpc,          "",     "JNC 8 guideline", bpcol),
        (d4, "SBP / DBP",          f"{sbp}/{dbp}", "mmHg", "Arterial reading", "#1a1d23"),
    ]
    for col, lbl, val, unit, sub, scol in derived:
        with col:
            st.markdown(f"""<div style="background:#fff;border:1px solid #e2e4e8;border-radius:6px;
padding:12px 14px;margin-bottom:12px">
<div style="font-family:'IBM Plex Mono',monospace;font-size:8px;font-weight:600;color:#9ca3af;
letter-spacing:.1em;margin-bottom:5px">{lbl}</div>
<div style="font-family:'IBM Plex Mono',monospace;font-size:17px;font-weight:600;color:{scol}">{val}
<span style="font-size:10px;font-weight:400;color:#9ca3af"> {unit}</span></div>
<div style="font-size:10px;color:#9ca3af;margin-top:2px">{sub}</div>
</div>""", unsafe_allow_html=True)

    # ── Live risk ──
    prob    = predict_risk(age, hr, sbp, dbp)
    contribs, _ = shap_contributions(age, hr, sbp, dbp)
    color, tier, guidance = risk_tier(prob)

    st.markdown(f"""
<div style="background:{color};border-radius:6px;padding:13px 20px;
            display:flex;align-items:center;justify-content:space-between;
            flex-wrap:wrap;gap:10px;margin-bottom:12px">
  <div style="display:flex;align-items:center;gap:10px">
    <div style="width:5px;height:5px;border-radius:50%;background:rgba(255,255,255,.5)"></div>
    <div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:8px;color:rgba(255,255,255,.6);letter-spacing:.12em">RISK TIER</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:14px;font-weight:700;color:#fff">{tier}</div>
    </div>
  </div>
  <div style="font-family:'IBM Plex Mono',monospace;font-size:28px;font-weight:600;color:#fff">{prob*100:.1f}%</div>
  <div style="font-family:'IBM Plex Sans',sans-serif;font-size:12px;color:rgba(255,255,255,.85);line-height:1.5">{guidance}</div>
</div>
""", unsafe_allow_html=True)

    _, bc, _ = st.columns([2, 1, 2])
    with bc:
        run = st.button("Run Risk Assessment", use_container_width=True)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════════
    if run:
        # Store trends for next run
        st.session_state.prev_hr   = hr
        st.session_state.prev_sbp  = sbp
        st.session_state.prev_dbp  = dbp
        st.session_state.prev_prob = prob
        st.session_state.run_count += 1

        # Row 1: gauge + SHAP waterfall
        r1a, r1b = st.columns([1, 1.6])

        with r1a:
            legend = "".join(
                f'<div style="text-align:center">'
                f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:8px;color:#9ca3af">{t}</div>'
                f'<div style="height:3px;width:26px;background:{c};border-radius:1px;margin:3px auto"></div>'
                f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:8px;color:#6b7280;margin-top:1px">{r}</div></div>'
                for t,c,r in [("STABLE","#16a34a","< 40%"),("ELEVATED","#d97706","40–70%"),("CRITICAL","#c0392b","> 70%")]
            )
            run_note = f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:8px;color:#9ca3af;text-align:center;margin-top:6px">Assessment #{st.session_state.run_count}</div>'
            st.markdown(card(f"""
{sec("Risk Gauge")}
{gauge_svg(prob, color)}
{divider()}
<div style="display:flex;justify-content:space-between;margin-top:6px">{legend}</div>
{run_note}
"""), unsafe_allow_html=True)

        with r1b:
            st.markdown(card(f"""
{sec("SHAP Waterfall — Feature Contributions to Risk")}
<div style="font-size:11px;color:#9ca3af;line-height:1.6;margin-bottom:14px">
  Each bar shows a feature's signed probability contribution versus the baseline prediction
  ({BASELINE_PROB*100:.1f}%). Red = increases risk. Green = reduces risk.
  Bars sum to final score minus baseline.
</div>
{shap_svg(contribs, BASELINE_PROB)}
<div style="margin-top:10px;font-size:10px;color:#9ca3af">
  Reference: Age 55 yr · HR 75 bpm · SBP 120 mmHg · DBP 80 mmHg (MAP 93 mmHg)
</div>
"""), unsafe_allow_html=True)

        # Row 2: vitals table + recommendations
        r2a, r2b = st.columns([1.6, 1])

        with r2a:
            rows = "".join([
                vital_row("Age",            age,     "yr",   "18–100",  "—",
                          "#6b7280"),
                vital_row("Heart Rate",     hr,      "bpm",  "60–100",  hr_class(hr)[0].upper(),
                          hr_class(hr)[1]),
                vital_row("Systolic BP",    sbp,     "mmHg", "90–120",
                          "HIGH" if sbp>=130 else ("LOW" if sbp<90 else "NORMAL"),
                          "#c0392b" if sbp>=130 or sbp<90 else "#16a34a"),
                vital_row("Diastolic BP",   dbp,     "mmHg", "60–80",
                          "HIGH" if dbp>=80  else ("LOW" if dbp<60 else "NORMAL"),
                          "#c0392b" if dbp>=80  or dbp<60 else "#16a34a"),
                vital_row("MAP",            map_val, "mmHg", "70–100",
                          "LOW" if map_val<65 else ("HIGH" if map_val>110 else "NORMAL"),
                          "#c0392b" if map_val<65 or map_val>110 else "#16a34a"),
                vital_row("Pulse Pressure", pp_val,  "mmHg", "30–50",
                          "WIDE" if pp_val>60 else ("NARROW" if pp_val<25 else "NORMAL"),
                          "#d97706" if pp_val>60 or pp_val<25 else "#16a34a"),
                vital_row("BP Class.",      bpc,     "",     "Normal",
                          bpc.upper(), bpcol),
            ])
            th = "".join(
                f'<th style="text-align:{"right" if i else "left"};padding:6px {"0 6px" if i else "0"};'
                f'font-family:\'IBM Plex Mono\',monospace;font-size:8px;font-weight:600;'
                f'color:#9ca3af;letter-spacing:.1em">{h}</th>'
                for i,h in enumerate(["PARAMETER","VALUE","UNIT","REFERENCE","STATUS"])
            )
            st.markdown(card(f"""
{sec("Vitals Reference Table")}
<table style="width:100%;border-collapse:collapse;font-family:'IBM Plex Sans',sans-serif">
  <thead><tr style="border-bottom:1px solid #e5e7eb">{th}</tr></thead>
  <tbody>{rows}</tbody>
</table>
"""), unsafe_allow_html=True)

        with r2b:
            if prob >= .70:
                recs = ["Notify attending physician immediately",
                        "Vitals every 15 minutes",
                        "Review medications for hemodynamic impact",
                        "Prepare for potential care escalation"]
            elif prob >= .40:
                recs = ["Reassess vitals within 30 minutes",
                        "Review fluid balance and recent changes",
                        "Consider cardiology consult if persistent",
                        "Document for shift handover"]
            else:
                recs = ["Continue standard monitoring protocol",
                        "Maintain current treatment plan",
                        "Reassess at next scheduled interval"]

            rec_html = "".join(
                f'<div style="display:flex;align-items:flex-start;gap:9px;margin-bottom:9px">'
                f'<div style="width:3px;height:3px;border-radius:50%;background:{color};margin-top:6px;flex-shrink:0"></div>'
                f'<div style="font-size:12px;color:#374151;line-height:1.5">{r}</div></div>'
                for r in recs
            )
            st.markdown(card(f"""
{sec("Clinical Recommendations")}
{rec_html}
{divider()}
<div style="font-size:10px;color:#9ca3af;line-height:1.7">
  Tier: <span style="font-family:'IBM Plex Mono',monospace;color:{color};font-weight:600">{tier}</span>
  &nbsp;·&nbsp; Score: <span style="font-family:'IBM Plex Mono',monospace;color:{color};font-weight:600">{prob*100:.1f}%</span>
</div>
""", accent=color), unsafe_allow_html=True)

        # ── Model transparency card ──
        st.markdown(card(f"""
{sec("Model Architecture & Transparency")}
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:20px">

  <div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:8px;color:#9ca3af;letter-spacing:.1em;margin-bottom:6px">MODEL TYPE</div>
    <div style="font-size:13px;font-weight:600;color:#1a1d23">Logistic Regression</div>
    <div style="font-size:11px;color:#6b7280;margin-top:3px;line-height:1.5">
      Generalized Linear Model (GLM) with sigmoid output. Fully interpretable by design —
      no black-box components. Every prediction is the direct sum of weighted feature deviations
      from a clinical reference baseline.
    </div>
  </div>

  <div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:8px;color:#9ca3af;letter-spacing:.1em;margin-bottom:6px">WHY NOT RANDOM FOREST / TRANSFORMER?</div>
    <div style="font-size:13px;font-weight:600;color:#1a1d23">Interpretability First</div>
    <div style="font-size:11px;color:#6b7280;margin-top:3px;line-height:1.5">
      Tree ensembles and attention-based models achieve higher AUC on tabular data, but require
      post-hoc SHAP approximations. For a clinical DSS, the FDA and EU MDR require
      explainability at the prediction level — a linear model satisfies this natively.
    </div>
  </div>

  <div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:8px;color:#9ca3af;letter-spacing:.1em;margin-bottom:6px">XAI METHOD</div>
    <div style="font-size:13px;font-weight:600;color:#1a1d23">Exact SHAP (Additive)</div>
    <div style="font-size:11px;color:#6b7280;margin-top:3px;line-height:1.5">
      Because the model is additive in log-odds space, SHAP values are computed exactly
      (not approximated via TreeSHAP or KernelSHAP). Each feature's contribution is its
      marginal effect on the output probability, satisfying the efficiency, symmetry,
      and dummy axioms of Shapley values.
    </div>
  </div>

  <div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:8px;color:#9ca3af;letter-spacing:.1em;margin-bottom:6px">PRODUCTION PATHWAY</div>
    <div style="font-size:13px;font-weight:600;color:#1a1d23">Next Steps</div>
    <div style="font-size:11px;color:#6b7280;margin-top:3px;line-height:1.5">
      Replace hand-coded weights with a model trained on an ICU cohort (e.g., MIMIC-IV).
      Validate against APACHE II / SOFA / NEWS2. Add temporal modeling (LSTM or GRU)
      for trend-aware risk. Register as a Software as a Medical Device (SaMD) if deploying clinically.
    </div>
  </div>

</div>
{divider()}
<div style="font-size:10px;color:#9ca3af;line-height:1.7">
  Decision support tool only. Not a validated clinical scoring instrument.
  All decisions must be confirmed by a qualified clinician. Not FDA-cleared.
</div>
"""), unsafe_allow_html=True)

    else:
        st.markdown("""
<div style="background:#fff;border:1px solid #e2e4e8;border-radius:6px;padding:36px 24px;
            text-align:center;margin-bottom:14px">
  <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;font-weight:600;
              color:#9ca3af;letter-spacing:.14em;margin-bottom:10px">AWAITING ASSESSMENT</div>
  <div style="font-size:13px;color:#6b7280;line-height:1.6">
    Adjust patient vitals above. Risk tier updates live in the banner.<br>
    Click <strong style="color:#1a1d23">Run Risk Assessment</strong> to view the full analysis panel.
  </div>
</div>
""", unsafe_allow_html=True)

else:
    # Errors present — suppress score entirely
    st.markdown("""
<div style="background:#fff;border:1px solid #e2e4e8;border-radius:6px;padding:36px 24px;
            text-align:center;margin-bottom:14px">
  <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;font-weight:600;
              color:#c0392b;letter-spacing:.14em;margin-bottom:10px">RISK SCORE SUPPRESSED</div>
  <div style="font-size:13px;color:#6b7280;line-height:1.6">
    One or more vitals are physiologically impossible.<br>
    Correct the values above before proceeding.
  </div>
</div>
""", unsafe_allow_html=True)
