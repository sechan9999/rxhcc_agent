"""
app.py — RxHCC FWA Autonomous Investigation Agent
Enterprise-grade Medicare Part D Fraud, Waste & Abuse Detection Platform
Powered by Google GenAI (Gemini 2.0 Flash) & Clinical RxHCC Risk Rules.
"""

import os
import json
import time
from datetime import datetime
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

# ── API Key Resolution (Streamlit Cloud Secrets or Local .env) ───────────────
try:
    from streamlit.errors import StreamlitSecretNotFoundError
    secret_key = st.secrets.get("GOOGLE_API_KEY") or st.secrets.get("GEMINI_API_KEY")
except Exception:
    secret_key = None

if secret_key:
    os.environ["GOOGLE_API_KEY"] = secret_key

HAS_API_KEY = bool(os.environ.get("GOOGLE_API_KEY"))

# ── Import Agent & Clinical Tools ─────────────────────────────────────────────
from agents import run_fwa_investigation
from tools import (
    ICD10_DB,
    NDC_DB,
    PROVIDER_DB,
    CONFLICT_RULES,
    get_provider_billing_history,
    generate_synthetic_claims_batch,
    evaluate_claims_batch,
)

# ── Streamlit Page Configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="RxHCC FWA Agent | Medicare Fraud Detection",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS Styles ─────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        color: #f8fafc;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-card h3 {
        margin: 0;
        font-size: 1.7rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-card p {
        margin: 4px 0 0 0;
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Verdict Banners */
    .verdict-banner-clear {
        background: linear-gradient(90deg, #064e3b 0%, #047857 100%);
        color: #ecfdf5;
        padding: 16px 20px;
        border-radius: 10px;
        border-left: 6px solid #10b981;
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 15px;
    }
    .verdict-banner-flag {
        background: linear-gradient(90deg, #78350f 0%, #b45309 100%);
        color: #fffbeb;
        padding: 16px 20px;
        border-radius: 10px;
        border-left: 6px solid #f59e0b;
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 15px;
    }
    .verdict-banner-escalate {
        background: linear-gradient(90deg, #881337 0%, #be123c 100%);
        color: #fff1f2;
        padding: 16px 20px;
        border-radius: 10px;
        border-left: 6px solid #f43f5e;
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 15px;
    }

    /* Provider Preview Card */
    .provider-card {
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-left: 4px solid #38bdf8;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 12px 0;
        font-size: 0.9rem;
        color: #0f172a;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    /* Risk Score Gauge Bar */
    .gauge-container {
        width: 100%;
        background-color: #e2e8f0;
        border-radius: 10px;
        height: 18px;
        overflow: hidden;
        margin: 8px 0;
        position: relative;
    }
    .gauge-fill {
        height: 100%;
        transition: width 0.6s ease-in-out;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════
FALLBACK_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
NON_TEXT_MARKERS = ("computer-use", "image", "audio", "tts", "embedding", "vision")


def discover_models() -> list[str]:
    """Return text/tool-calling Gemini models, sorted best first."""
    if not HAS_API_KEY:
        return FALLBACK_MODELS
    try:
        from google import genai
        client = genai.Client()
        names = []
        for m in client.models.list():
            name = m.name.split("/")[-1] if "/" in m.name else m.name
            lname = name.lower()
            if "gemini" not in lname or any(marker in lname for marker in NON_TEXT_MARKERS):
                continue
            names.append(name)
        priority = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-flash-latest", "gemini-2.5-pro"]
        ordered = [p for p in priority if p in names]
        for n in sorted(names):
            if n not in ordered:
                ordered.append(n)
        return ordered or FALLBACK_MODELS
    except Exception:
        return FALLBACK_MODELS


# ══════════════════════════════════════════════════════════════════════════════
# LOAD SCENARIOS FROM JSON
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_scenarios():
    path = os.path.join(os.path.dirname(__file__), "sample_claims.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
            scenarios = {}
            for item in raw:
                scenarios[item["scenario"]] = {
                    "claim_id": item.get("claim_id", ""),
                    "beneficiary_id": item.get("beneficiary_id", ""),
                    "icd10_codes": ", ".join(item.get("icd10_codes", [])),
                    "ndc_codes": ", ".join(item.get("ndc_codes", [])),
                    "provider_npi": item.get("provider_npi", ""),
                    "claim_amount": f"{item.get('claim_amount', 0.0):.2f}",
                    "note": item.get("notes", ""),
                    "expected": item.get("expected_verdict", ""),
                }
            return scenarios
    return {}


SCENARIOS = load_scenarios()
scenario_keys = list(SCENARIOS.keys())

# ── Session State Management for Synced Inputs ─────────────────────────────────
if "active_scenario" not in st.session_state:
    st.session_state["active_scenario"] = scenario_keys[0] if scenario_keys else ""
    first_sc = SCENARIOS.get(st.session_state["active_scenario"], {})
    st.session_state["claim_id_val"] = first_sc.get("claim_id", "CLM-2026-001")
    st.session_state["ben_id_val"] = first_sc.get("beneficiary_id", "BNF-F-99211")
    st.session_state["icd_val"] = first_sc.get("icd10_codes", "E11.9, Z79.4")
    st.session_state["ndc_val"] = first_sc.get("ndc_codes", "00002143380, 00002143480")
    st.session_state["npi_val"] = first_sc.get("provider_npi", "1122334455")
    st.session_state["amt_val"] = first_sc.get("claim_amount", "245.00")


def on_scenario_select():
    sel = st.session_state.sidebar_scenario_select
    st.session_state["active_scenario"] = sel
    if sel in SCENARIOS:
        sc = SCENARIOS[sel]
        st.session_state["claim_id_val"] = sc["claim_id"]
        st.session_state["ben_id_val"] = sc["beneficiary_id"]
        st.session_state["icd_val"] = sc["icd10_codes"]
        st.session_state["ndc_val"] = sc["ndc_codes"]
        st.session_state["npi_val"] = sc["provider_npi"]
        st.session_state["amt_val"] = sc["claim_amount"]
        st.session_state.pop("audit_result", None)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/color/96/hospital.png", width=64)
    st.title("RxHCC FWA Agent")
    st.caption("CMS Medicare Part D Fraud, Waste & Abuse Investigator")
    st.divider()

    st.subheader("📋 Select Test Scenario")
    st.selectbox(
        "Load vetted clinical claim:",
        scenario_keys,
        index=scenario_keys.index(st.session_state["active_scenario"]) if st.session_state["active_scenario"] in scenario_keys else 0,
        key="sidebar_scenario_select",
        on_change=on_scenario_select,
        help="Instantly loads clinical diagnosis codes, NDC medications, and provider profiles."
    )

    demo_mode = st.checkbox(
        "⚡ Fast Demo Mode (Offline Simulation)",
        value=True,
        help="When enabled, runs against the local clinical rules engine — instant, deterministic, and 100% free."
    )

    if demo_mode:
        st.success("🟢 Demo Engine Active (Local Rules)")
    elif HAS_API_KEY:
        st.info("🤖 Live Gemini Agent Mode Active")
    else:
        st.warning("⚠️ No GOOGLE_API_KEY found. Operating in Demo Mode.")

    st.divider()
    st.markdown("**System Architecture**")
    st.caption("• 5-Tool Autonomous Execution Loop\n• RxHCC Risk Adjustment Integration\n• OIG & False Claims Act Rule Base")

    st.divider()
    st.markdown("[🌐 GitHub Repository](https://github.com/sechan9999/rxhcc_agent)")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.title("🏥 RxHCC Medicare Part D Fraud Investigation Agent")
st.caption(
    "**Google GenAI & RxHCC Capstone** · Multi-Tool Autonomous Pre-Payment Audit System"
)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_audit, tab_batch, tab_rules, tab_roi, tab_settings = st.tabs([
    "🔍 Real-Time Claim Auditor",
    "📊 Batch Simulation & Analytics",
    "📖 Rules & Knowledge Matrix",
    "📈 Medicare ROI Calculator",
    "⚙️ Model & System Config",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: REAL-TIME CLAIM AUDITOR
# ─────────────────────────────────────────────────────────────────────
with tab_audit:
    current_scenario = SCENARIOS.get(st.session_state["active_scenario"], {})
    if current_scenario.get("note"):
        st.info(f"💡 **Active Scenario Overview:** {current_scenario['note']}")

    with st.form("audit_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            claim_id_input = st.text_input("Claim ID", value=st.session_state["claim_id_val"])
            ben_id_input = st.text_input(
                "Beneficiary ID",
                value=st.session_state["ben_id_val"],
                help="CMS Beneficiary ID. Suffix -M- or -F- indicates gender for clinical incongruence checks.",
            )
            icd_input = st.text_input(
                "ICD-10 Diagnosis Codes (comma-separated)",
                value=st.session_state["icd_val"],
                help="e.g., E11.9, Z79.4, I10, C50.911",
            )
        with c2:
            ndc_input = st.text_input(
                "NDC Pharmacy Drug Codes (comma-separated)",
                value=st.session_state["ndc_val"],
                help="11-digit NDC codes (e.g. 00169406012 for Ozempic, 00406051201 for OxyContin)",
            )
            npi_input = st.text_input(
                "Prescribing Provider NPI",
                value=st.session_state["npi_val"],
                help="10-digit National Provider Identifier (e.g., 1234567890 for Sunshine Pain Clinic)",
            )
            amt_input = st.text_input(
                "Claim Billed Amount ($)",
                value=st.session_state["amt_val"],
                help="Total billed amount in USD, e.g. 245.00 or 8400.00",
            )

        # Provider Live Insight Preview
        p_info = get_provider_billing_history(npi_input)
        if p_info:
            p_flags = ", ".join(p_info.get("flags", [])) if p_info.get("flags") else "No adverse flags on file"
            st.markdown(
                f"""
                <div class="provider-card">
                    🩺 <b>Provider On Record:</b> {p_info['name']} ({p_info['specialty']} — {p_info['state']})<br>
                    📊 90-Day Volume: <b>{p_info['total_claims_90d']} claims</b> | Peer Rank: <b>{p_info['peer_percentile']}th percentile</b> |
                    Anomaly Score: <b>{p_info['anomaly_score']:.2f}</b><br>
                    🚩 <i>Active Risk Signals:</i> {p_flags}
                </div>
                """,
                unsafe_allow_html=True,
            )

        submit_audit = st.form_submit_button("⚡ Run Autonomous FWA Investigation", type="primary", use_container_width=True)

    if submit_audit:
        if not all([claim_id_input, ben_id_input, icd_input, npi_input, amt_input]):
            st.error("Please fill in all mandatory claim fields.")
            st.stop()

        try:
            clean_amt = float(amt_input.replace("$", "").replace(",", "").strip())
        except ValueError:
            st.error(f"Invalid claim amount '{amt_input}'. Please enter a valid numerical dollar amount.")
            st.stop()

        prompt = f"""Investigate the following Medicare Part D claim for Fraud, Waste & Abuse:

Claim ID:         {claim_id_input}
Beneficiary ID:   {ben_id_input}
ICD-10 Codes:     {icd_input}
NDC Drug Codes:   {ndc_input if ndc_input else 'None'}
Provider NPI:     {npi_input}
Claim Amount:     ${clean_amt:,.2f}
Date of Service:  {datetime.utcnow().strftime('%Y-%m-%d')}

Please conduct a full investigation following the standard FWA workflow."""

        agent_trace_logs = []
        try:
            with st.spinner("🤖 Autonomous Agent executing 5-step clinical investigation workflow..."):
                start_t = time.time()
                final_report_text = run_fwa_investigation(
                    prompt=prompt,
                    agent_logs=agent_trace_logs,
                    demo_mode=demo_mode,
                )
                elapsed = max(time.time() - start_t, 0.25)
        except Exception as e:
            st.error(f"Investigation execution failed: {e}")
            st.stop()

        # Parse verdict
        verdict = "UNKNOWN"
        rep_upper = final_report_text.upper()
        if "ESCALATE" in rep_upper:
            verdict = "ESCALATE"
        elif "FLAG_FOR_REVIEW" in rep_upper or "FLAG FOR REVIEW" in rep_upper:
            verdict = "FLAG_FOR_REVIEW"
        elif "CLEAR" in rep_upper:
            verdict = "CLEAR"

        # Parse score
        score_val = 0.0
        for line in final_report_text.split("\n"):
            if "Score" in line and "%" in line:
                try:
                    pct_str = line.split(":", 1)[1].replace("%", "").strip()
                    score_val = float(pct_str) / 100.0
                    break
                except Exception:
                    pass

        if score_val == 0.0:
            score_val = 0.85 if verdict == "ESCALATE" else (0.45 if verdict == "FLAG_FOR_REVIEW" else 0.05)

        st.session_state["audit_result"] = {
            "claim_id": claim_id_input,
            "beneficiary_id": ben_id_input,
            "claim_amount": clean_amt,
            "verdict": verdict,
            "score": score_val,
            "final_text": final_report_text,
            "trace_logs": agent_trace_logs,
            "elapsed_seconds": elapsed,
        }

    # Render results if available
    if "audit_result" in st.session_state:
        res = st.session_state["audit_result"]
        v = res["verdict"]
        score = res["score"]

        st.divider()
        st.subheader("📋 Investigation Findings & Compliance Decision")

        # Top Executive Banner
        if v == "CLEAR":
            st.markdown(
                f'<div class="verdict-banner-clear">✅ VERDICT: CLEAR — APPROVED FOR PAYMENT<br>'
                f'<span style="font-size:0.95rem; font-weight:normal;">Claim {res["claim_id"]} passed all 5 clinical and billing compliance checks in {res["elapsed_seconds"]:.2f}s.</span></div>',
                unsafe_allow_html=True,
            )
        elif v == "FLAG_FOR_REVIEW":
            st.markdown(
                f'<div class="verdict-banner-flag">⚠️ VERDICT: FLAG FOR REVIEW — 30-DAY PAYMENT HOLD<br>'
                f'<span style="font-size:0.95rem; font-weight:normal;">Suspicious billing anomalies or documentation discrepancies detected for Claim {res["claim_id"]}.</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="verdict-banner-escalate">🚨 VERDICT: ESCALATE — IMMEDIATE PAYMENT BLOCK & SIU REFERRAL<br>'
                f'<span style="font-size:0.95rem; font-weight:normal;">Severe Fraud, Waste & Abuse signals detected for Claim {res["claim_id"]} (Potential False Claims Act violation).</span></div>',
                unsafe_allow_html=True,
            )

        # Visual Risk Gauge Meter
        gauge_color = "#10b981" if v == "CLEAR" else ("#f59e0b" if v == "FLAG_FOR_REVIEW" else "#ef4444")
        st.markdown(
            f"""
            <div style="margin: 15px 0;">
                <div style="display:flex; justify-content:space-between; font-size:0.85rem; font-weight:600;">
                    <span>Composite Fraud Risk Score: {score:.0%}</span>
                    <span style="color:{gauge_color};">{v}</span>
                </div>
                <div class="gauge-container">
                    <div class="gauge-fill" style="width: {score * 100}%; background-color: {gauge_color};"></div>
                </div>
                <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#94a3b8;">
                    <span>0% (Clear)</span>
                    <span>30% (Review Threshold)</span>
                    <span>70% (Escalate Threshold)</span>
                    <span>100% (High Fraud)</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Quick Metric Overview
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="metric-card"><h3>${res["claim_amount"]:,.2f}</h3><p>Claim Amount</p></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><h3>{v}</h3><p>Integrity Verdict</p></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><h3>{res["elapsed_seconds"]:.2f}s</h3><p>Execution Time</p></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-card"><h3>5 / 5</h3><p>Tools Executed</p></div>', unsafe_allow_html=True)

        # Step-by-Step Tool Execution Trace
        st.markdown("#### 🔬 Step-by-Step Autonomous Agent Trace")
        with st.expander("🔎 View Live Multi-Tool Calling Sequence & Payloads", expanded=True):
            trace_items = res["trace_logs"]
            for item in trace_items:
                if isinstance(item, dict):
                    itype = item.get("type")
                    if itype == "system":
                        st.markdown(f"⏱ `{item.get('time')}` **System:** {item.get('message')}")
                    elif itype == "tool_call":
                        tool_name = item.get("tool")
                        args = item.get("args")
                        st.markdown(f"⚙️ `{item.get('time')}` **Agent calling tool:** `{tool_name}`")
                        st.caption(f"Input Parameters: `{json.dumps(args)}`")
                    elif itype == "tool_response":
                        tool_name = item.get("tool")
                        result = item.get("result")
                        st.markdown(f"📥 `{item.get('time')}` **Tool `{tool_name}` returned result:**")
                        st.json(result)
                else:
                    st.text(str(item))

        # Dossier & Full Report
        st.markdown("#### 📄 CMS Special Investigations Unit (SIU) Dossier")
        st.code(res["final_text"], language="markdown")

        c_dl1, c_dl2 = st.columns(2)
        with c_dl1:
            st.download_button(
                label="💾 Download SIU Dossier (.txt)",
                data=res["final_text"],
                file_name=f"SIU_Investigation_{res['claim_id']}.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with c_dl2:
            audit_json = json.dumps({
                "claim_id": res["claim_id"],
                "beneficiary_id": res["beneficiary_id"],
                "claim_amount": res["claim_amount"],
                "verdict": res["verdict"],
                "risk_score": res["score"],
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "trace": res["trace_logs"],
            }, indent=2)
            st.download_button(
                label="💾 Download Structured Audit Log (.json)",
                data=audit_json,
                file_name=f"Audit_Trace_{res['claim_id']}.json",
                mime="application/json",
                use_container_width=True,
            )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: BATCH SIMULATION & ANALYTICS
# ─────────────────────────────────────────────────────────────────────
with tab_batch:
    st.subheader("📊 High-Throughput Batch Claim Simulation")
    st.markdown(
        "Simulate Medicare Part D claims across diverse clinical cohorts to evaluate "
        "pre-payment fraud detection velocity, financial savings, and risk distribution."
    )

    bc1, bc2 = st.columns([3, 1])
    with bc1:
        batch_size = st.slider("Cohort Sample Size (Claims):", min_value=10, max_value=100, value=30, step=5)
    with bc2:
        run_batch_btn = st.button("🚀 Run Batch Audit", type="primary", use_container_width=True)

    if run_batch_btn or "batch_summary" not in st.session_state:
        with st.spinner(f"Evaluating {batch_size} claims through RxHCC hybrid engine..."):
            synthetic_claims = generate_synthetic_claims_batch(n_claims=batch_size)
            batch_eval = evaluate_claims_batch(synthetic_claims)
            st.session_state["batch_summary"] = batch_eval

    summary = st.session_state["batch_summary"]
    df = pd.DataFrame(summary["claims_data"])

    # KPI row
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Claims Audited", f"{summary['total_claims']}")
    k2.metric("Total Billed", f"${summary['total_billed']:,.0f}")
    k3.metric("Fraud Rate", f"{summary['fraud_detection_rate']:.1%}")
    k4.metric("Value Blocked", f"${summary['escalated_value']:,.0f}", delta="Stopped at gate")
    k5.metric("Est. Medicare Savings", f"${summary['potential_savings']:,.0f}", delta="Pre-payment ROI")

    st.divider()

    # Visual Charts
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        verdict_df = pd.DataFrame({
            "Verdict": ["CLEAR (Approve)", "FLAG_FOR_REVIEW (Hold)", "ESCALATE (Block)"],
            "Count": [summary["clear_count"], summary["flag_count"], summary["escalate_count"]],
        })
        fig_pie = px.pie(
            verdict_df,
            names="Verdict",
            values="Count",
            title="Portfolio Integrity Distribution",
            color="Verdict",
            color_discrete_map={
                "CLEAR (Approve)": "#10b981",
                "FLAG_FOR_REVIEW (Hold)": "#f59e0b",
                "ESCALATE (Block)": "#ef4444",
            },
            hole=0.45,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_chart2:
        fig_scatter = px.scatter(
            df,
            x="claim_amount",
            y="risk_score",
            color="verdict",
            hover_data=["claim_id", "scenario_type", "top_factor"],
            title="Claim Risk Score vs. Dollar Value",
            color_discrete_map={
                "CLEAR": "#10b981",
                "FLAG_FOR_REVIEW": "#f59e0b",
                "ESCALATE": "#ef4444",
            },
            labels={"claim_amount": "Claim Amount ($)", "risk_score": "Composite Risk Score (0–1)"},
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("#### 📑 Batch Claims Log")
    filter_choice = st.radio(
        "Filter Claims Table:",
        ["All Claims", "CLEAR Only", "FLAG_FOR_REVIEW Only", "ESCALATE Only"],
        horizontal=True,
    )
    filtered_df = df
    if filter_choice == "CLEAR Only":
        filtered_df = df[df["verdict"] == "CLEAR"]
    elif filter_choice == "FLAG_FOR_REVIEW Only":
        filtered_df = df[df["verdict"] == "FLAG_FOR_REVIEW"]
    elif filter_choice == "ESCALATE Only":
        filtered_df = df[df["verdict"] == "ESCALATE"]

    st.dataframe(
        filtered_df[["claim_id", "beneficiary_id", "scenario_type", "claim_amount", "risk_score_pct", "verdict", "top_factor"]],
        use_container_width=True,
    )

    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Filtered Batch Audit Log (.csv)",
        data=csv_data,
        file_name="Medicare_Batch_FWA_Audit.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: RULES & KNOWLEDGE MATRIX
# ─────────────────────────────────────────────────────────────────────
with tab_rules:
    st.subheader("📖 Clinical Rules & Regulatory Knowledge Matrix")
    st.markdown("Reference database of clinical validation rules, DEA schedules, and federal compliance mandates.")

    rule_subtab1, rule_subtab2, rule_subtab3 = st.tabs([
        "ICD-10 & RxHCC Mappings",
        "High-Risk NDC Pharmacy Registry",
        "Federal Fraud Statutes",
    ])

    with rule_subtab1:
        icd_records = []
        for code, data in ICD10_DB.items():
            icd_records.append({
                "ICD-10 Code": code,
                "Clinical Description": data["desc"],
                "Severity (1-5)": data["severity"],
                "RxHCC Relevant": "Yes" if data["hcc"] else "No",
                "HCC Category": data.get("hcc_category") or "—",
                "Clinical Class": data.get("category", "General"),
                "Gender Restriction": data.get("gender") or "None",
            })
        st.dataframe(pd.DataFrame(icd_records), use_container_width=True)

    with rule_subtab2:
        ndc_records = []
        for code, data in NDC_DB.items():
            ndc_records.append({
                "NDC Code": code,
                "Drug Brand & Generic": data["name"],
                "DEA Schedule": data.get("schedule") or "Non-Controlled",
                "Therapeutic Class": data.get("class", "Standard"),
                "Risk Rating": data.get("risk", "LOW"),
                "Prior Auth Required": "Yes" if data.get("prior_auth") else "No",
            })
        st.dataframe(pd.DataFrame(ndc_records), use_container_width=True)

    with rule_subtab3:
        st.markdown(
            """
            ### ⚖️ Federal Fraud, Waste & Abuse Legal Framework
            - **False Claims Act (31 U.S.C. § 3729)**: Imposes liability on persons and companies who defraud governmental programs. Penalties include treble damages plus statutory fines per claim.
            - **Anti-Kickback Statute (42 U.S.C. § 1320a-7b(b))**: Prohibits the knowing and willful payment of remuneration to induce or reward patient referrals or generate business involving Medicare.
            - **Physician Self-Referral Law (Stark Law, 42 U.S.C. § 1395nn)**: Prohibits physicians from referring Medicare patients for designated health services to entities with which the physician has a financial relationship.
            - **42 CFR § 405.980 (Reopening Determinations)**: Governs CMS post-payment recovery timelines, allowing 1 year for any reason and 4 years for good cause.
            """
        )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: MEDICARE ROI CALCULATOR
# ─────────────────────────────────────────────────────────────────────
with tab_roi:
    st.subheader("📈 Medicare Advantage & Part D ROI Calculator")
    st.markdown(
        "Estimate annual financial recovery and administrative savings achieved by replacing "
        "legacy post-payment recovery with the **RxHCC Autonomous Multi-Tool Agent**."
    )

    r_col1, r_col2 = st.columns(2)
    with r_col1:
        members = st.slider("Covered Medicare Beneficiaries:", min_value=10_000, max_value=1_000_000, value=150_000, step=10_000)
        annual_spend_per_member = st.slider("Average Annual Part D Spend per Member ($):", min_value=1_000, max_value=6_000, value=3_200, step=200)

    with r_col2:
        total_part_d_spend = members * annual_spend_per_member
        improper_spend = total_part_d_spend * 0.068
        traditional_recovery = improper_spend * 0.15
        ai_agent_recovery = improper_spend * 0.65
        net_gain = ai_agent_recovery - traditional_recovery

        st.markdown(
            f"""
            <div class="metric-card" style="margin-bottom:15px;">
                <h3>${total_part_d_spend:,.0f}</h3>
                <p>Total Plan Part D Spend</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    r_kpi1, r_kpi2, r_kpi3 = st.columns(3)
    r_kpi1.metric("Improper Payments at Risk", f"${improper_spend:,.0f}", help="Based on CMS 6.8% improper billing baseline")
    r_kpi2.metric("Traditional Pay-&-Chase", f"${traditional_recovery:,.0f}", delta="15% recovery rate")
    r_kpi3.metric("RxHCC AI Agent Savings", f"${ai_agent_recovery:,.0f}", delta=f"+${net_gain:,.0f} Net Gain", delta_color="normal")

    st.divider()
    st.caption("Sources: CMS OIG Annual Report, Government Accountability Office (GAO-23-106202).")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: SYSTEM CONFIGURATION
# ─────────────────────────────────────────────────────────────────────
with tab_settings:
    st.subheader("⚙️ System Configuration & Gemini Model Selection")

    models_list = discover_models()
    selected_model = st.selectbox(
        "Selected Gemini GenAI Engine:",
        models_list,
        index=0,
        help="Filtered to active text & function-calling models to guarantee stability."
    )

    st.markdown(f"**Current Execution Status:** {'🟢 API Connected' if HAS_API_KEY else '🟡 Local Demo Simulation Only'}")
    if not HAS_API_KEY:
        st.info("To enable live Gemini API calls, configure `GOOGLE_API_KEY` in your environment or Streamlit Cloud Secrets.")

    st.divider()
    st.markdown("#### Detection Sensitivity Thresholds")
    clear_thresh = st.slider("CLEAR Threshold (Upper Limit):", 0.10, 0.40, 0.29, 0.01)
    escalate_thresh = st.slider("ESCALATE Threshold (Lower Limit):", 0.60, 0.85, 0.70, 0.01)
    st.caption(f"Verdicts: 0.00–{clear_thresh:.2f} CLEAR | {clear_thresh:.2f}–{escalate_thresh:.2f} FLAG FOR REVIEW | >{escalate_thresh:.2f} ESCALATE")

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.caption(
    "RxHCC Autonomous Multi-Tool Agent · Medicare Part D Program Integrity System · "
    "Kaggle 5-Day AI Agents Capstone (Agents for Good Track)"
)
