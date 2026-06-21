"""
app.py — RxHCC FWA Agent | Streamlit Demo UI
Kaggle Capstone: 5-Day AI Agents Intensive Vibe Coding (Agents for Good)

Run:  streamlit run app.py
"""

import asyncio
import os
import time
import streamlit as st
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agents import fwa_orchestrator, create_runner

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
    "**Agents for Good** · Google ADK + Gemini 2.0 Flash · "
    "Kaggle 5-Day AI Agents Capstone"
)
st.markdown(
    "An autonomous multi-agent system that investigates Medicare Part D claims "
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
    st.header("ℹ️ How It Works")
    st.markdown("""
**3 specialized agents work in sequence:**

1. 🔍 **Claim Analyzer**
   Validates ICD-10 codes, extracts structured fields

2. 📊 **Risk Scorer**
   RxHCC model + drug combo check + provider history

3. 📄 **Report Writer**
   Generates SIU-ready compliance report

**Orchestrator** coordinates all three and delivers the final verdict.
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
    # Validate inputs
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

    # Agent trace expander
    trace_placeholder = st.empty()
    result_placeholder = st.empty()
    final_placeholder  = st.empty()

    agent_logs: list[str] = []

    with st.spinner("Agents working... (Claim Analyzer → Risk Scorer → Report Writer)"):
        try:
            runner, session_service = create_runner()

            async def run_investigation():
                session = await session_service.create_session(
                    app_name="rxhcc_fwa_agent", user_id="streamlit_user"
                )
                events = runner.run(
                    user_id="streamlit_user",
                    session_id=session.id,
                    new_message=types.Content(
                        role="user",
                        parts=[types.Part(text=prompt)],
                    ),
                )
                final_text = ""
                for event in events:
                    # Collect sub-agent outputs for the trace
                    author = getattr(event, "author", "agent")
                    if hasattr(event, "content") and event.content:
                        for part in event.content.parts:
                            if hasattr(part, "text") and part.text and part.text.strip():
                                agent_logs.append(f"[{author}]\n{part.text.strip()}")
                    if event.is_final_response():
                        final_text = event.content.parts[0].text
                return final_text

            final_response = asyncio.run(run_investigation())

            # ── Agent Trace ──────────────────────────────────────────────────
            with trace_placeholder.expander("🔧 Agent Reasoning Trace", expanded=False):
                for log in agent_logs:
                    st.markdown(f"```\n{log}\n```")

            # ── Parse verdict from response ──────────────────────────────────
            verdict = "UNKNOWN"
            score_str = "N/A"
            if "ESCALATE" in final_response.upper():
                verdict = "ESCALATE"
            elif "FLAG_FOR_REVIEW" in final_response.upper() or "FLAG FOR REVIEW" in final_response.upper():
                verdict = "FLAG_FOR_REVIEW"
            elif "CLEAR" in final_response.upper():
                verdict = "CLEAR"

            # ── Verdict banner ───────────────────────────────────────────────
            VERDICT_CONFIG = {
                "CLEAR":          ("verdict-clear",    "✅ CLEAR — Approve for Payment"),
                "FLAG_FOR_REVIEW":("verdict-flag",     "⚠️ FLAG FOR REVIEW — Hold Claim"),
                "ESCALATE":       ("verdict-escalate", "🚨 ESCALATE — Refer to SIU"),
                "UNKNOWN":        ("metric-box",       "❓ Verdict Undetermined"),
            }
            css_class, label = VERDICT_CONFIG[verdict]
            st.markdown(f'<div class="{css_class}">{label}</div>', unsafe_allow_html=True)
            st.markdown("")

            # ── Metrics row ──────────────────────────────────────────────────
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Claim ID", claim_id)
            m2.metric("Beneficiary", beneficiary_id)
            m3.metric("Claim Amount", f"${float(claim_amount.replace(',','')):.2f}")
            m4.metric("Verdict", verdict)

            st.divider()

            # ── Full investigation report ────────────────────────────────────
            st.subheader("📄 Full Investigation Report")
            st.markdown(final_response)

        except Exception as e:
            st.error(f"Agent error: {e}")
            st.exception(e)
            st.info(
                "💡 Make sure GOOGLE_API_KEY is set in your environment: "
                "`export GOOGLE_API_KEY=your_key_here`"
            )

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.caption(
    "🏥 RxHCC FWA Agent · Built with Google ADK + Gemini 2.0 Flash · "
    "Kaggle 5-Day AI Agents Intensive Vibe Coding Capstone · Agents for Good Track"
)
