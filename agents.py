"""
agents.py — RxHCC FWA Agent
Built with Google ADK + Gemini 2.0 Flash for the Kaggle
"5-Day AI Agents: Intensive Vibe Coding" Capstone (Agents for Good track).

Single-agent design: fwa_agent holds all 5 tools and uses Gemini's native
function-calling to sequence through them autonomously — no sub-agent
delegation, maximum reliability across ADK versions.
"""

import os
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from tools import (
    lookup_icd10_code,
    get_provider_billing_history,
    check_drug_combination,
    calculate_rxhcc_risk_score,
    generate_fwa_report,
)

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

SYSTEM_INSTRUCTION = """You are RxHCC-FWA, an autonomous Medicare Part D
Fraud, Waste & Abuse investigator powered by the RxHCC (Prescription Drug
Hierarchical Condition Category) risk-adjustment model.

Medicare fraud costs Americans $60–100 billion per year. Your job is to
catch it BEFORE payment is made, using your 5 tools in sequence.

═══ MANDATORY INVESTIGATION WORKFLOW ═══

STEP 1 — VALIDATE ICD-10 CODES
  Call lookup_icd10_code once for EACH diagnosis code in the claim.
  Record: validity, description, severity (1–5), gender restriction.

STEP 2 — CHECK DRUG COMBINATIONS
  Call check_drug_combination with ALL NDC codes as a list.
  Detect: opioid+benzo, opioid+soma, multiple Schedule II drugs.

STEP 3 — PULL PROVIDER RISK PROFILE
  Call get_provider_billing_history with the provider NPI.
  Record: anomaly_score, peer_percentile, risk_flags.

STEP 4 — CALCULATE COMPOSITE RISK SCORE
  Call calculate_rxhcc_risk_score with:
    beneficiary_id  — exact string from claim
    icd10_codes     — Python list of strings
    ndc_codes       — Python list of strings ([] if none)
    claim_amount    — float (strip $ and commas)
    provider_npi    — exact NPI string

STEP 5 — GENERATE COMPLIANCE REPORT
  Call generate_fwa_report with all findings collected above.
  Pass provider_name, provider_npi, provider_anomaly_score, provider_flags,
  drugs_prescribed, drug_combination_risk, drug_flags, risk_score (float 0–1),
  verdict, risk_factors (list), recommendation.

═══ VERDICT THRESHOLDS ═══
  CLEAR           (risk_score 0.00–0.29) → Approve for payment
  FLAG_FOR_REVIEW (risk_score 0.30–0.69) → Hold; request documentation
  ESCALATE        (risk_score 0.70–1.00) → Block payment; refer to SIU

═══ FINAL OUTPUT FORMAT ═══
After all 5 tool calls, present:

  INVESTIGATION COMPLETE
  Claim: [ID] | Beneficiary: [ID] | Amount: $[X]
  Risk Score: [X%] | Verdict: [CLEAR / FLAG_FOR_REVIEW / ESCALATE]

  Top Risk Signals:
  1. [highest-risk finding]
  2. [second finding if present]
  3. [third finding if present]

  [Full report as returned by generate_fwa_report]

IMPORTANT: You MUST call all 5 tools before writing your final answer.
Do not skip any tool. Do not guess — always call the tool to get real data."""

# ── Single agent with all 5 tools ─────────────────────────────────────────────
fwa_agent = Agent(
    model=MODEL,
    name="fwa_agent",
    description=(
        "Autonomous Medicare Part D FWA investigator using RxHCC risk scoring. "
        "Validates ICD-10 codes, checks drug combinations, profiles providers, "
        "calculates fraud risk, and generates SIU-ready compliance reports."
    ),
    instruction=SYSTEM_INSTRUCTION,
    tools=[
        lookup_icd10_code,
        get_provider_billing_history,
        check_drug_combination,
        calculate_rxhcc_risk_score,
        generate_fwa_report,
    ],
)

# Keep fwa_orchestrator as an alias so app.py imports don't break
fwa_orchestrator = fwa_agent


# ── Runner factory ─────────────────────────────────────────────────────────────
def create_runner() -> tuple[Runner, InMemorySessionService]:
    session_service = InMemorySessionService()
    runner = Runner(
        agent=fwa_agent,
        app_name="rxhcc_fwa_agent",
        session_service=session_service,
    )
    return runner, session_service


# ── Quick CLI test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import asyncio

    TEST_CLAIM = """
Investigate this Medicare Part D claim for FWA:

Claim ID:        CLM-TEST-003
Beneficiary ID:  BNF-M-77542
ICD-10 Codes:    C50.911, E11.9
NDC Codes:       00406051201, 59011049010
Provider NPI:    1234567890
Claim Amount:    $8,400.00
Date of Service: 2024-11-15
"""

    async def run_test():
        runner, svc = create_runner()
        session = await svc.create_session(
            app_name="rxhcc_fwa_agent", user_id="test_user"
        )
        events = runner.run(
            user_id="test_user",
            session_id=session.id,
            new_message=types.Content(
                role="user",
                parts=[types.Part(text=TEST_CLAIM)],
            ),
        )
        for event in events:
            if event.is_final_response():
                if event.content and event.content.parts:
                    print(event.content.parts[0].text)

    asyncio.run(run_test())
