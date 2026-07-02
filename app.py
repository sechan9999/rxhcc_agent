"""
app.py — RxHCC FWA Agent | Streamlit Demo UI
Kaggle Capstone: 5-Day AI Agents Intensive Vibe Coding (Agents for Good)
"""

import os
import json
import time
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
from google.genai import types

# ── Load API key from Streamlit secrets (Cloud) or environment (local) ────────
from streamlit.errors import StreamlitSecretNotFoundError

# Safely load API key from Streamlit secrets if available, otherwise rely on environment variable
try:
    secret_key = st.secrets.get("GOOGLE_API_KEY") or st.secrets.get("GEMINI_API_KEY")
except StreamlitSecretNotFoundError:
    secret_key = None

if secret_key:
    os.environ["GOOGLE_API_KEY"] = secret_key

if not os.environ.get("GOOGLE_API_KEY"):
    st.warning(
        "🔑 **GOOGLE_API_KEY not set.**\n\n- The app will operate in **demo mode only**. To enable real Gemini calls, add a secret in Streamlit Cloud Settings → Secrets."
    )

# ── Import FWA investigation agent ────────────────────────────────────────────
import importlib
import agents
importlib.reload(agents)
from agents import run_fwa_investigation

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RxHCC FWA Agent",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .verdict-clear    { background:#d4edda; color:#155724; padding:12px 18px; border-radius:8px; font-weight:bold; font-size:1.1rem; }
    .verdict-flag     { background:#fff3cd; color:#856404; padding:12px 18px; border-radius:8px; font-weight:bold; font-size:1.1rem; }
    .verdict-escalate { background:#f8d7da; color:#721c24; padding:12px 18px; border-radius:8px; font-weight:bold; font-size:1.1rem; }
    .metric-box       { background:#f8f9fa; border:1px solid #dee2e6; border-radius:8px; padding:14px; text-align:center; }
    .agent-trace      { background:#1e1e2e; color:#cdd6f4; font-family:monospace; font-size:0.82rem; padding:12px; border-radius:6px; }
    .report-section   { font-size:18px; line-height:1.6; }
    div[data-testid="stMarkdown"] { font-size: 1.05rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏥 RxHCC FWA Investigation Agent")
st.caption(
    "**Agents for Good** · Google Gemini 2.0 Flash · "
    "Kaggle 5-Day AI Agents Capstone"
)
st.markdown(
    "An autonomous multi-tool AI agent that investigates Medicare Part D claims "
    "for **Fraud, Waste & Abuse** before payment is released."
)
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Sample claims & info
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("📋 Sample Scenarios")
    scenario = st.radio(
        "Load a pre-built claim:",
        [
            "✅ Clean — Diabetic Maintenance",
            "⚠️ Suspicious — Opioid/Benzo Combo",
            "🚨 Escalate — Gender Mismatch + High-Risk Provider",
        ],
        index=0,
    )
    demo_mode = st.checkbox("🧪 Demo mode (use mock data)", value=True, help="When enabled, the agent runs against simulated data for faster demos.")
    if demo_mode:
        st.info("Demo mode is ON: tool calls will return simulated responses.")

    st.divider()

    st.markdown("**Navigation**")
    st.markdown(
        """
    1. Sample Scenarios  
    2. Settings  
    3. How It Works  
    4. Impact  
    5. Investigation Report  
    """
    )
    
    # Interactive navigation (select box)
    nav_step = st.sidebar.selectbox(
        "Go to step",
        ["Sample Scenarios", "Settings", "How It Works", "Impact", "Investigation Report"],
        index=0,
        help="Select a step to jump to its section in the main view"
    )
    st.session_state.nav_step = nav_step

    st.divider()
    st.header("⚙️ Settings")
    model_options = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
    if os.environ.get("GOOGLE_API_KEY"):
        try:
            from google import genai
            temp_client = genai.Client()
            api_models = temp_client.models.list()
            fetched_models = []
            for m in api_models:
                m_name = m.name.split("/")[-1] if "/" in m.name else m.name
                supports_gen = True
                if hasattr(m, "supported_methods"):
                    supports_gen = any("generateContent" in method for method in m.supported_methods)
                elif hasattr(m, "supported_generation_methods"):
                    supports_gen = any("generateContent" in method for method in m.supported_generation_methods)
                if supports_gen and "gemini" in m_name.lower():
                    fetched_models.append(m_name)
            if fetched_models:
                unique_models = list(set(fetched_models))
                if "gemini-2.0-flash" in unique_models:
                    unique_models.remove("gemini-2.0-flash")
                    model_options = ["gemini-2.0-flash"] + sorted(unique_models)
                else:
                    model_options = sorted(unique_models)
        except Exception:
            pass
    selected_model = st.selectbox(
        "Select Gemini Model:",
        model_options,
        index=0,
        help="Only models supported by your Google API Key are listed. Switch models if you hit rate limits.",
    )
    st.divider()
    st.header("ℹ️ How It Works")
    st.markdown(
        """
        **5 tools called autonomously:**

        1. 🔍 **lookup_icd10_code**
        2. 💊 **check_drug_combination**
        3. 🏥 **get_provider_billing_history**
        4. 📊 **calculate_rxhcc_risk_score**
        5. 📄 **generate_fwa_report**
        """
    )
    st.divider()
    st.header("📈 FWA Impact")
    st.metric("Annual Medicare Fraud", "$60–100B", delta="-$12B saved by AI detection")
    st.metric("False Positive Rate (rule-based)", "~40%", delta="-15% with AI agents", delta_color="inverse")
    st.caption("Sources: CMS OIG, GAO 2023")

# ══════════════════════════════════════════════════════════════════════════════
# PRE-BUILT SCENARIOS
# ══════════════════════════════════════════════════════════════════════════════
SCENARIOS = {
    "✅ Clean — Diabetic Maintenance": {
        "claim_id": "CLM-2024-001",
        "beneficiary_id": "BNF-F-99211",
        "icd10_codes": "E11.9, Z79.4",
        "ndc_codes": "00002143480",
        "provider_npi": "1111111111",
        "claim_amount": "320.00",
        "note": "Routine Type 2 diabetes claim with insulin. Expect CLEAR verdict.",
    },
    "⚠️ Suspicious — Opioid/Benzo Combo": {
        "claim_id": "CLM-2024-002",
        "beneficiary_id": "BNF-M-44831",
        "icd10_codes": "G89.29, F11.10",
        "ndc_codes": "00406051201, 59011049010",
        "provider_npi": "9876543210",
        "claim_amount": "1250.00",
        "note": "Chronic pain + opioid abuse + OxyContin/Xanax combo from a flagged pharmacy. Expect FLAG or ESCALATE.",
    },
    "🚨 Escalate — Gender Mismatch + High-Risk Provider": {
        "claim_id": "CLM-2024-003",
        "beneficiary_id": "BNF-M-77542",
        "icd10_codes": "C50.911, E11.9",
        "ndc_codes": "00406051201, 59011049010, 65162010850",
        "provider_npi": "1234567890",
        "claim_amount": "8400.00",
        "note": "MALE patient billed for female breast cancer + opioid triad + watchlisted provider. Expect ESCALATE.",
    },
}

defaults = SCENARIOS[scenario]

# ══════════════════════════════════════════════════════════════════════════════
# CLAIM INPUT FORM
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("📝 Claim Details")
if defaults.get("note"):
    st.info(f"💡 {defaults['note']}")
with st.form("claim_form", clear_on_submit=False):
    col1, col2 = st.columns(2)
    with col1:
        claim_id = st.text_input("Claim ID", value=defaults["claim_id"])
        beneficiary_id = st.text_input(
            "Beneficiary ID",
            value=defaults["beneficiary_id"],
            help="Suffix -M- or -F- is used for gender inference",
        )
        icd10_input = st.text_input(
            "ICD-10 Codes (comma-separated)",
            value=defaults["icd10_codes"],
            help="e.g. E11.9, Z79.4, I10",
        )
        st.divider()
    with col2:
        ndc_input = st.text_input(
            "NDC Drug Codes (comma-separated)",
            value=defaults["ndc_codes"],
            help="11-digit NDC codes from pharmacy claim",
        )
        provider_npi = st.text_input("Provider NPI", value=defaults["provider_npi"])
        claim_amount = st.text_input("Claim Amount ($)", value=defaults["claim_amount"])
    submitted = st.form_submit_button(
        "🔍 Investigate Claim", type="primary", use_container_width=True
    )

# ══════════════════════════════════════════════════════════════════════════════
# AGENT EXECUTION
# ══════════════════════════════════════════════════════════════════════════════
if submitted:
    if not all([claim_id, beneficiary_id, icd10_input, provider_npi, claim_amount]):
        st.error("Please fill in all required fields.")
        st.stop()
    prompt = f"""Investigate the following Medicare Part D claim for Fraud, Waste & Abuse:\n\nClaim ID:         {claim_id}\nBeneficiary ID:   {beneficiary_id}\nICD-10 Codes:     {icd10_input}\nNDC Drug Codes:   {ndc_input if ndc_input else 'None'}\nProvider NPI:     {provider_npi}\nClaim Amount:     ${claim_amount}\nDate of Service:  {time.strftime('%Y-%m-%d')}\n\nPlease conduct a full investigation following the standard FWA workflow."""
    agent_logs = []
    events = []  # placeholder for agent events; actual agent returns trace inside run_fwa_investigation
    # Execute investigation and build UI
    try:
        final_text = run_fwa_investigation(prompt, agent_logs, model=selected_model)
    except Exception as e:
        st.error(f"Agent error: {e}")
        st.exception(e)
        final_text = ""
    # Collect sub‑agent outputs for the trace (if any additional events are populated)
    for event in events:
        author = getattr(event, "author", "agent")
        if hasattr(event, "content") and event.content:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text and part.text.strip():
                    agent_logs.append(f"[{author}]\n{part.text.strip()}")
        if event.is_final_response():
            pass
    # Trace UI
    trace_placeholder = st.expander(f"🔧 Agent Tool Trace ({len(agent_logs)} calls)", expanded=False)
    with trace_placeholder:
        for log in agent_logs:
            st.markdown(f"```\n{log}\n```")
    trace_json = json.dumps(agent_logs, indent=2)
    st.download_button(
        label="Download Trace JSON",
        data=trace_json,
        file_name="agent_trace.json",
        mime="application/json",
        key="download_trace",
    )
    # Verdict parsing
    verdict = "UNKNOWN"
    if "ESCALATE" in final_text.upper():
        verdict = "ESCALATE"
    elif "FLAG_FOR_REVIEW" in final_text.upper() or "FLAG FOR REVIEW" in final_text.upper():
        verdict = "FLAG_FOR_REVIEW"
    elif "CLEAR" in final_text.upper():
        verdict = "CLEAR"
    # Verdict banner
    VERDICT_CONFIG = {
        "CLEAR": ("verdict-clear", "✅ CLEAR — Approve for Payment"),
        "FLAG_FOR_REVIEW": ("verdict-flag", "⚠️ FLAG FOR REVIEW — Hold Claim"),
        "ESCALATE": ("verdict-escalate", "🚨 ESCALATE — Refer to SIU"),
        "UNKNOWN": ("metric-box", "❓ Verdict Undetermined"),
    }
    css_class, label = VERDICT_CONFIG[verdict]
    st.markdown(f'<div class="{css_class}">{label}</div>', unsafe_allow_html=True)
    st.markdown("")
    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Claim ID", claim_id)
    m2.metric("Beneficiary", beneficiary_id)
    m3.metric("Claim Amount", f"${float(claim_amount.replace(',','')):.2f}")
    m4.metric("Verdict", verdict)
    st.divider()
    # Full investigation report
    st.subheader("📄 Full Investigation Report")
    st.markdown(f"<div class='report-section'>{final_text}</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.caption(
    "🏥 RxHCC FWA Agent · Built with Google Gemini 2.0 Flash · "
    "Kaggle 5-Day AI Agents Intensive Vibe Coding Capstone · Agents for Good Track"
)
