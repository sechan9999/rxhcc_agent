"""
app.py — RxHCC FWA Agent | Streamlit Demo UI
Kaggle Capstone: 5-Day AI Agents Intensive Vibe Coding (Agents for Good)

Run:  streamlit run app.py
"""

import os
import time
import streamlit as st
from google import genai as _genai
from google.genai import types as _gtypes

# ── Load API key from Streamlit secrets (Cloud) or environment (local) ────────
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
elif "GEMINI_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]

if not os.environ.get("GOOGLE_API_KEY"):
    st.error(
        "🔑 **GOOGLE_API_KEY not set.**\n\n"
        "- **Streamlit Cloud:** Go to App Settings → Secrets and add:\n"
        "  ```\n  GOOGLE_API_KEY = \"your_key_here\"\n  ```\n"
        "- **Local:** `export GOOGLE_API_KEY=your_key_here`\n\n"
        "Get a free key at https://aistudio.google.com/app/apikey"
    )
    st.stop()

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
st.markdown("""
<style>
.verdict-clear    { background:#d4edda; color:#155724; padding:12px 18px;
                    border-radius:8px; font-weight:bold; font-size:1.1rem; }
.verdict-flag     { background:#fff3cd; color:#856404; padding:12px 18px;
                    border-radius:8px; font-weight:bold; font-size:1.1rem; }
.verdict-escalate { background:#f8d7da; color:#721c24; padding:12px 18px;
                    border-radius:8px; font-weight:bold; font-size:1.1rem; }
.metric-box       { background:#f8f9fa; border:1px solid #dee2e6;
                    border-radius:8px; padding:14px; text-align:center; }
.agent-trace      { background:#1e1e2e; color:#cdd6f4; font-family:monospace;
                    font-size:0.82rem; padding:12px; border-radius:6px; }
</style>
""", unsafe_allow_html=True)

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

    st.divider()
    st.header("⚙️ Settings")
    
    # Query available models dynamically from Gemini API based on user credentials
    model_options = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
    if os.environ.get("GOOGLE_API_KEY"):
        try:
            from google import genai
            temp_client = genai.Client()
            api_models = temp_client.models.list()
            fetched_models = []
            for m in api_models:
                m_name = m.name.split("/")[-1] if "/" in m.name else m.name
                
                # Check if model supports content generation
                supports_gen = False
                if hasattr(m, "supported_methods"):
                    supports_gen = any("generateContent" in method for method in m.supported_methods)
                elif hasattr(m, "supported_generation_methods"):
                    supports_gen = any("generateContent" in method for method in m.supported_generation_methods)
                else:
                    supports_gen = True
                    
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
        help="Only models supported by your Google API Key are listed. Switch models if you hit rate limits."
    )

    st.divider()
    st.header("ℹ️ How It Works")
    st.markdown("""
**5 tools called autonomously:**

1. 🔍 **lookup_icd10_code**
   Validates diagnosis codes, flags gender restrictions

2. 💊 **check_drug_combination**
   Detects opioid+benzo, pill-mill patterns

3. 🏥 **get_provider_billing_history**
   Provider anomaly score vs peer benchmarks

4. 📊 **calculate_rxhcc_risk_score**
   Composite RxHCC fraud probability (0–100%)

5. 📄 **generate_fwa_report**
   SIU-ready compliance report
""")

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
        claim_id       = st.text_input("Claim ID",                    value=defaults["claim_id"])
        beneficiary_id = st.text_input("Beneficiary ID",              value=defaults["beneficiary_id"],
                                        help="Suffix -M- or -F- is used for gender inference")
        icd10_input    = st.text_input("ICD-10 Codes (comma-separated)", value=defaults["icd10_codes"],
                                        help="e.g. E11.9, Z79.4, I10")

    with col2:
        ndc_input      = st.text_input("NDC Drug Codes (comma-separated)", value=defaults["ndc_codes"],
                                        help="11-digit NDC codes from pharmacy claim")
        provider_npi   = st.text_input("Provider NPI",                value=defaults["provider_npi"])
        claim_amount   = st.text_input("Claim Amount ($)",            value=defaults["claim_amount"])

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

    prompt = f"""Investigate the following Medicare Part D claim for Fraud, Waste & Abuse:

Claim ID:         {claim_id}
Beneficiary ID:   {beneficiary_id}
ICD-10 Codes:     {icd10_input}
NDC Drug Codes:   {ndc_input if ndc_input else 'None'}
Provider NPI:     {provider_npi}
Claim Amount:     ${claim_amount}
Date of Service:  {time.strftime('%Y-%m-%d')}

Please conduct a full investigation following the standard FWA workflow."""

    st.divider()
    st.subheader("🤖 Agent Investigation in Progress")

    trace_placeholder = st.empty()
    agent_logs: list[str] = []

    with st.spinner("Agent working... (validating codes → scoring risk → generating report)"):
        try:
            final_text = run_fwa_investigation(prompt, agent_logs, model=selected_model)

            if not final_text:
                final_text = (
                    "⚠️ The agent did not produce a text response after all tool calls.\n\n"
                    "Tool calls executed: " + str(len(agent_logs)) + "\n\n"
                    "Check that your GOOGLE_API_KEY has quota remaining."
                )

            # ── Agent Trace ────────────────────────────────────────────────────
            with trace_placeholder.expander(
                f"🔧 Agent Tool Trace ({len(agent_logs)} calls)", expanded=False
            ):
                for log in agent_logs:
                    st.markdown(f"```\n{log}\n```")

            # ── Parse verdict ──────────────────────────────────────────────────
            verdict = "UNKNOWN"
            if "ESCALATE" in final_text.upper():
                verdict = "ESCALATE"
            elif "FLAG_FOR_REVIEW" in final_text.upper() or "FLAG FOR REVIEW" in final_text.upper():
                verdict = "FLAG_FOR_REVIEW"
            elif "CLEAR" in final_text.upper():
                verdict = "CLEAR"

            # ── Verdict banner ─────────────────────────────────────────────────
            VERDICT_CONFIG = {
                "CLEAR":          ("verdict-clear",    "✅ CLEAR — Approve for Payment"),
                "FLAG_FOR_REVIEW":("verdict-flag",     "⚠️ FLAG FOR REVIEW — Hold Claim"),
                "ESCALATE":       ("verdict-escalate", "🚨 ESCALATE — Refer to SIU"),
                "UNKNOWN":        ("metric-box",       "❓ Verdict Undetermined"),
            }
            css_class, label = VERDICT_CONFIG[verdict]
            st.markdown(f'<div class="{css_class}">{label}</div>', unsafe_allow_html=True)
            st.markdown("")

            # ── Metrics row ────────────────────────────────────────────────────
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Claim ID", claim_id)
            m2.metric("Beneficiary", beneficiary_id)
            m3.metric("Claim Amount", f"${float(claim_amount.replace(',','')):.2f}")
            m4.metric("Verdict", verdict)

            st.divider()

            # ── Full investigation report ──────────────────────────────────────
            st.subheader("📄 Full Investigation Report")
            st.markdown(final_text)

        except Exception as e:
            st.error(f"Agent error: {e}")
            st.exception(e)
            st.info(
                "💡 Common causes: invalid API key, quota exceeded, "
                "or network issue. Check https://aistudio.google.com/app/apikey"
            )

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.caption(
    "🏥 RxHCC FWA Agent · Built with Google Gemini 2.0 Flash · "
    "Kaggle 5-Day AI Agents Intensive Vibe Coding Capstone · Agents for Good Track"
)
