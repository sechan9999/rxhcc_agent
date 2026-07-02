"""
agents.py — RxHCC FWA Multi-Agent System
Built with Google ADK + Gemini 2.0 Flash for the Kaggle
"5-Day AI Agents: Intensive Vibe Coding" Capstone (Agents for Good track).

Agent topology:
  fwa_orchestrator
    ├── claim_analyzer     (validates ICD-10 codes, extracts claim fields)
    ├── risk_scorer        (calculates RxHCC fraud probability + drug check)
    └── report_writer      (synthesizes findings → compliance report)
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

# ── Gemini model used across all agents ───────────────────────────────────────
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")


# ══════════════════════════════════════════════════════════════════════════════
# SUB-AGENT 1 — Claim Analyzer
# Responsibility: parse the raw claim, validate every ICD-10 and NDC code,
# flag any structurally suspicious fields before scoring begins.
# ══════════════════════════════════════════════════════════════════════════════
claim_analyzer = Agent(
    model=MODEL,
    name="claim_analyzer",
    description=(
        "Validates and parses Medicare Part D claims. "
        "Checks ICD-10 codes for validity, gender restrictions, and HCC relevance."
    ),
    instruction="""You are a Medicare claims validation specialist with expertise in
ICD-10-CM coding and CMS billing rules.

When you receive a claim, perform these steps IN ORDER:

1. EXTRACT structured fields from the claim text:
   - Claim ID, Beneficiary ID, date of service
   - All ICD-10-CM diagnosis codes (comma-separated)
   - All NDC drug codes if present
   - Prescribing / dispensing provider NPI
   - Total claim amount

2. VALIDATE each ICD-10 code using lookup_icd10_code:
   - Call the tool once per code
   - Note: validity, description, severity (1-5), gender restriction

3. FLAG clinical anomalies:
   - Any invalid codes → mark as INVALID
   - Codes with gender restrictions → note for gender mismatch check
   - Severity 4-5 codes → flag as HIGH-SEVERITY (require strong documentation)

4. OUTPUT a structured summary:
   ```
   CLAIM PARSING COMPLETE
   Claim ID: ...
   Beneficiary ID: ...  Inferred Gender: M/F/Unknown
   Codes Validated: X/Y valid
   High-Severity Diagnoses: [list]
   Gender-Restricted Codes: [list]
   NDC Codes: [list]
   Provider NPI: ...
   Claim Amount: $...
   Structural Flags: [any immediate red flags]
   ```

Be precise. Every code must be checked individually.""",
    tools=[lookup_icd10_code],
    output_key="claim_analysis",
)


# ══════════════════════════════════════════════════════════════════════════════
# SUB-AGENT 2 — Risk Scorer
# Responsibility: run the RxHCC fraud probability model, check drug combos,
# and pull provider billing history to build a complete risk picture.
# ══════════════════════════════════════════════════════════════════════════════
risk_scorer = Agent(
    model=MODEL,
    name="risk_scorer",
    description=(
        "Calculates FWA fraud risk score using the RxHCC model, "
        "drug combination analysis, and provider billing history."
    ),
    instruction="""You are an FWA risk analyst specializing in Medicare Part D
fraud detection using predictive modeling and claims analytics.

Using the validated claim data from the claim analyzer, perform:

STEP A — DRUG SAFETY CHECK (if NDC codes present)
  Call check_drug_combination with the list of NDC codes.
  Document: drugs identified, DEA schedules, combination risk, any flags.

STEP B — PROVIDER RISK PROFILE
  Call get_provider_billing_history with the provider NPI.
  Note: anomaly score, peer percentile, controlled-substance %, and flags.

STEP C — COMPOSITE RISK SCORE
  Call calculate_rxhcc_risk_score with ALL of the following:
    - beneficiary_id (exact string from claim)
    - icd10_codes (Python list of strings)
    - ndc_codes (Python list of strings, or [] if none)
    - claim_amount (float, no $ sign)
    - provider_npi (exact NPI string)

STEP D — RISK SUMMARY
  Report clearly:
  ```
  RISK ASSESSMENT COMPLETE
  Risk Score:   X% (0-100%)
  Verdict:      CLEAR / FLAG_FOR_REVIEW / ESCALATE
  Drug Risk:    LOW / MEDIUM / HIGH
  Provider Risk: [anomaly score]
  Risk Factors Identified:
    [1] ...
    [2] ...
  Recommendation: [one sentence action]
  ```

Common FWA patterns that increase score:
  • Gender-diagnosis mismatch (e.g., male billed for ovarian cyst)
  • Opioid + benzodiazepine + muscle relaxant "holy trinity" combo
  • Provider in 95th+ percentile for controlled substance billing
  • Claim amount >3x provider's own average
  • Multiple Schedule II drugs co-prescribed""",
    tools=[check_drug_combination, get_provider_billing_history, calculate_rxhcc_risk_score],
    output_key="risk_assessment",
)


# ══════════════════════════════════════════════════════════════════════════════
# SUB-AGENT 3 — Report Writer
# Responsibility: synthesize claim analysis + risk assessment into a
# formal compliance report suitable for SIU investigators.
# ══════════════════════════════════════════════════════════════════════════════
report_writer = Agent(
    model=MODEL,
    name="report_writer",
    description=(
        "Generates structured FWA investigation reports for compliance officers "
        "and the CMS Special Investigations Unit."
    ),
    instruction="""You are a compliance report writer for Medicare fraud investigations.
Your reports may be used in legal proceedings — be precise and factual.

Using the claim analysis and risk assessment outputs:

1. COLLECT all required fields:
   - claim_id, risk_score (float 0-1), verdict, risk_factors (list)
   - provider_name, provider_npi, provider_anomaly_score, provider_flags (list)
   - drugs_prescribed (list), drug_combination_risk, drug_flags (list)
   - recommendation

2. CALL generate_fwa_report with all collected fields.
   Pass lists as Python lists, floats as floats (not strings).

3. PRESENT the complete report exactly as returned by the tool.

4. APPEND an EXECUTIVE SUMMARY (3 sentences max) at the top:
   "EXECUTIVE SUMMARY: [One sentence on what was found].
    [One sentence on the highest-risk signal].
    [One sentence on the immediate next step.]"

Tone: formal, factual, no speculation. Every flag must be traceable to
a specific tool output, not inferred.""",
    tools=[generate_fwa_report],
    output_key="final_report",
)


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR — FWA Investigation Orchestrator
# The root agent that drives the entire investigation pipeline.
# Delegates to sub-agents in sequence, then delivers the final verdict.
# ══════════════════════════════════════════════════════════════════════════════
fwa_orchestrator = Agent(
    model=MODEL,
    name="fwa_orchestrator",
    description=(
        "Root orchestrator for Medicare Part D FWA investigations. "
        "Coordinates claim validation, risk scoring, and report generation."
    ),
    instruction="""You are the RxHCC FWA Investigation Orchestrator — the AI system
protecting Medicare beneficiaries and U.S. taxpayers from healthcare fraud.

Medicare fraud costs Americans $60-100 billion per year.
Your job is to catch it before payment is made.

═══ INVESTIGATION WORKFLOW ═══

PHASE 1 — CLAIM VALIDATION
  Delegate to claim_analyzer.
  Wait for: structured claim data + ICD-10 validation results.

PHASE 2 — RISK SCORING
  Delegate to risk_scorer with the validated claim data.
  Wait for: composite risk score, verdict, and all risk factors.

PHASE 3 — INVESTIGATION REPORT
  Delegate to report_writer with full findings from Phases 1 & 2.
  Wait for: complete formatted compliance report.

PHASE 4 — FINAL VERDICT DELIVERY
  Present to the user:

  ┌─────────────────────────────────────────┐
  │  INVESTIGATION COMPLETE                 │
  │  Claim: [ID]                            │
  │  Risk Score: [X%]  Verdict: [VERDICT]   │
  │  Top Flags: [1-3 most critical]         │
  │  Next Action: [one clear sentence]      │
  └─────────────────────────────────────────┘

  Then include the full report from report_writer.

═══ VERDICT THRESHOLDS ═══
  CLEAR          (0–29%)  → Approve for payment
  FLAG_FOR_REVIEW (30–69%) → Hold; request documentation
  ESCALATE       (70–100%) → Block payment; refer to SIU

You protect the program. Be thorough. Be precise.""",
    sub_agents=[claim_analyzer, risk_scorer, report_writer],
)


# ══════════════════════════════════════════════════════════════════════════════
# Runner factory — call this to get a configured Runner + session
# ══════════════════════════════════════════════════════════════════════════════
def create_runner() -> tuple[Runner, InMemorySessionService]:
    """
    Create and return a (Runner, SessionService) pair for the FWA orchestrator.
    Call session_service.create_session(...) then runner.run(...) to investigate.
    """
    session_service = InMemorySessionService()
    runner = Runner(
        agent=fwa_orchestrator,
        app_name="rxhcc_fwa_agent",
        session_service=session_service,
    )
    return runner, session_service


# ══════════════════════════════════════════════════════════════════════════════
# Quick CLI test
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import asyncio

    TEST_CLAIM = """
Investigate this Medicare Part D claim for FWA:

Claim ID:       CLM-TEST-003
Beneficiary ID: BNF-M-77542
ICD-10 Codes:   C50.911, E11.9
NDC Codes:      00406051201, 59011049010
Provider NPI:   1234567890
Claim Amount:   $8,400.00
Date of Service: 2024-11-15
"""

    async def run_test():
        # Respect demo mode: set environment variable for downstream tools if needed
        if demo_mode:
            os.environ["DEMO_MODE"] = "1"
        else:
            os.environ["DEMO_MODE"] = "0"
        runner, svc = create_runner()
        session = await svc.create_session(app_name="rxhcc_fwa_agent", user_id="test_user")
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
                # Safely extract final response text; some events may have empty content
                if hasattr(event, "content") and event.content and getattr(event.content, "parts", None):
                    part = event.content.parts[0]
                    if hasattr(part, "text") and part.text:
                        print(part.text)
                    else:
                        print("[No text in final response]")
                else:
                    print("[Final response missing content]")

    asyncio.run(run_test())
