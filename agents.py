"""
agents.py — RxHCC FWA Autonomous Multi-Tool Agent
Autonomous Medicare Part D Fraud, Waste & Abuse Investigator
Powered by Google GenAI (Gemini 2.0 / 2.5) with local deterministic simulation fallback.
"""

import os
import time
import json
from datetime import datetime
from typing import Optional, List, Dict, Any

from tools import (
    lookup_icd10_code,
    get_provider_billing_history,
    check_drug_combination,
    calculate_rxhcc_risk_score,
    generate_fwa_report,
)

SYSTEM_INSTRUCTION = """You are RxHCC-FWA, an autonomous Medicare Part D Fraud, Waste & Abuse investigator powered by the RxHCC (Prescription Drug Hierarchical Condition Category) risk-adjustment model.

Medicare fraud costs American taxpayers $60–100 billion per year. Your objective is to detect, investigate, and stop fraudulent claims BEFORE payment release, executing all 5 clinical tools in strict sequence.

═══ MANDATORY INVESTIGATION WORKFLOW ═══
STEP 1 — VALIDATE ICD-10 DIAGNOSIS CODES
Call lookup_icd10_code once for EACH diagnosis code in the claim.
Verify clinical validity, CMS HCC category, severity (1–5), and gender restrictions.

STEP 2 — CHECK DRUG COMBINATIONS & GLP-1 OFF-LABEL ABUSE
Call check_drug_combination with ALL NDC codes as a list.
Detect lethal drug triads (opioid+benzo+soma 'Holy Trinity'), Schedule II clusters, and high-cost GLP-1 weight-loss patterns.

STEP 3 — PROFILE PRESCRIBING PROVIDER
Call get_provider_billing_history with the provider's 10-digit NPI.
Review billing anomalies, peer percentile rank, and historical enforcement flags.

STEP 4 — CALCULATE COMPOSITE RxHCC FRAUD RISK SCORE
Call calculate_rxhcc_risk_score with:
  beneficiary_id — exact string from claim (includes gender suffix)
  icd10_codes — Python list of diagnosis strings
  ndc_codes — Python list of pharmacy NDC strings
  claim_amount — float dollar amount
  provider_npi — exact 10-digit NPI string

STEP 5 — GENERATE COMPLIANCE REPORT & ENFORCEMENT ACTION
Call generate_fwa_report with all findings collected above.
Pass claim_id, risk_score, verdict, risk_factors, provider_name, provider_npi, provider_anomaly_score,
provider_flags, drugs_prescribed, drug_combination_risk, drug_flags, and recommendation.

═══ VERDICT THRESHOLDS ═══
CLEAR (risk_score 0.00–0.29) → Approve claim for electronic payment.
FLAG_FOR_REVIEW (risk_score 0.30–0.69) → Place 30-day payment hold; request medical records.
ESCALATE (risk_score 0.70–1.00) → Block payment; refer to CMS Special Investigations Unit (SIU).

IMPORTANT: You MUST call all 5 tools before formulating your final report. Do not skip any tool. Always verify data through tools."""


def run_fwa_investigation(
    prompt: str,
    agent_logs: list,
    model: str = None,
    demo_mode: bool = None,
) -> str:
    """
    Execute the FWA investigation workflow.

    Args:
        prompt: Claim details prompt string
        agent_logs: Mutable list to accumulate trace messages & tool steps
        model: Gemini model name (e.g. 'gemini-2.0-flash')
        demo_mode: True forces fast local deterministic tool execution; False forces live Gemini API;
                   None auto-detects from environment.

    Returns:
        Final compliance investigation report string.
    """
    if demo_mode is None:
        demo_active = (os.environ.get("DEMO_MODE") == "1") or not os.environ.get("GOOGLE_API_KEY")
    else:
        demo_active = demo_mode

    # ─────────────────────────────────────────────────────────────────────────
    # PATH A: LOCAL DETERMINISTIC SIMULATION (DEMO MODE)
    # ─────────────────────────────────────────────────────────────────────────
    if demo_active:
        agent_logs.append({
            "type": "system",
            "time": datetime.utcnow().strftime("%H:%M:%S"),
            "message": "⚡ Initializing Clinical Rule Engine & Local Agent Simulation (Demo Mode)..."
        })

        # Parse claim attributes from prompt
        claim_id = "CLM-UNKNOWN"
        beneficiary_id = "BEN-UNKNOWN"
        icd10_codes: List[str] = []
        ndc_codes: List[str] = []
        provider_npi = "0000000000"
        claim_amount_val = 0.0

        for line in prompt.split("\n"):
            line_str = line.strip()
            if line_str.startswith("Claim ID:"):
                claim_id = line_str.split(":", 1)[1].strip()
            elif line_str.startswith("Beneficiary ID:"):
                beneficiary_id = line_str.split(":", 1)[1].strip()
            elif line_str.startswith("ICD-10 Codes:"):
                raw_codes = line_str.split(":", 1)[1].strip()
                icd10_codes = [c.strip() for c in raw_codes.split(",") if c.strip()]
            elif line_str.startswith("NDC Drug Codes:"):
                raw_drugs = line_str.split(":", 1)[1].strip()
                if raw_drugs and raw_drugs.lower() != "none":
                    ndc_codes = [d.strip() for d in raw_drugs.split(",") if d.strip()]
            elif line_str.startswith("Provider NPI:"):
                provider_npi = line_str.split(":", 1)[1].strip()
            elif line_str.startswith("Claim Amount:"):
                amt_str = line_str.split(":", 1)[1].strip().replace("$", "").replace(",", "")
                try:
                    claim_amount_val = float(amt_str)
                except ValueError:
                    claim_amount_val = 0.0

        # Step 1: Validate ICD-10 codes
        icd10_results = []
        for code in icd10_codes:
            agent_logs.append({
                "type": "tool_call",
                "tool": "lookup_icd10_code",
                "args": {"code": code},
                "time": datetime.utcnow().strftime("%H:%M:%S"),
            })
            res = lookup_icd10_code(code)
            icd10_results.append(res)
            agent_logs.append({
                "type": "tool_response",
                "tool": "lookup_icd10_code",
                "result": res,
                "time": datetime.utcnow().strftime("%H:%M:%S"),
            })

        # Step 2: Check drug combination
        agent_logs.append({
            "type": "tool_call",
            "tool": "check_drug_combination",
            "args": {"ndc_codes": ndc_codes},
            "time": datetime.utcnow().strftime("%H:%M:%S"),
        })
        drug_res = check_drug_combination(ndc_codes)
        agent_logs.append({
            "type": "tool_response",
            "tool": "check_drug_combination",
            "result": drug_res,
            "time": datetime.utcnow().strftime("%H:%M:%S"),
        })

        # Step 3: Pull provider billing history
        agent_logs.append({
            "type": "tool_call",
            "tool": "get_provider_billing_history",
            "args": {"npi": provider_npi},
            "time": datetime.utcnow().strftime("%H:%M:%S"),
        })
        provider_res = get_provider_billing_history(provider_npi)
        agent_logs.append({
            "type": "tool_response",
            "tool": "get_provider_billing_history",
            "result": provider_res,
            "time": datetime.utcnow().strftime("%H:%M:%S"),
        })

        # Step 4: Calculate composite risk score
        agent_logs.append({
            "type": "tool_call",
            "tool": "calculate_rxhcc_risk_score",
            "args": {
                "beneficiary_id": beneficiary_id,
                "icd10_codes": icd10_codes,
                "ndc_codes": ndc_codes,
                "claim_amount": claim_amount_val,
                "provider_npi": provider_npi,
            },
            "time": datetime.utcnow().strftime("%H:%M:%S"),
        })
        risk_res = calculate_rxhcc_risk_score(
            beneficiary_id=beneficiary_id,
            icd10_codes=icd10_codes,
            ndc_codes=ndc_codes,
            claim_amount=claim_amount_val,
            provider_npi=provider_npi,
        )
        agent_logs.append({
            "type": "tool_response",
            "tool": "calculate_rxhcc_risk_score",
            "result": risk_res,
            "time": datetime.utcnow().strftime("%H:%M:%S"),
        })

        # Step 5: Generate compliance report
        agent_logs.append({
            "type": "tool_call",
            "tool": "generate_fwa_report",
            "args": {"claim_id": claim_id, "risk_score": risk_res["risk_score"], "verdict": risk_res["verdict"]},
            "time": datetime.utcnow().strftime("%H:%M:%S"),
        })
        report_text = generate_fwa_report(
            claim_id=claim_id,
            risk_score=risk_res["risk_score"],
            verdict=risk_res["verdict"],
            risk_factors=risk_res["risk_factors"],
            provider_name=provider_res["name"],
            provider_npi=provider_npi,
            provider_anomaly_score=provider_res["anomaly_score"],
            provider_flags=provider_res.get("flags", []),
            drugs_prescribed=drug_res.get("drugs", []),
            drug_combination_risk=drug_res.get("combination_risk", "LOW"),
            drug_flags=drug_res.get("flags", []),
            recommendation=risk_res["recommendation"],
        )
        agent_logs.append({
            "type": "tool_response",
            "tool": "generate_fwa_report",
            "result": {"status": "SUCCESS", "report_length": len(report_text)},
            "time": datetime.utcnow().strftime("%H:%M:%S"),
        })

        top_signals = "\n".join(f"{i+1}. {f}" for i, f in enumerate(risk_res["risk_factors"][:3])) or "1. None identified."

        final_response = f"""INVESTIGATION COMPLETE
Claim: {claim_id} | Beneficiary: {beneficiary_id} | Amount: ${claim_amount_val:,.2f}
Risk Score: {risk_res['risk_score']:.0%} | Verdict: {risk_res['verdict']}

Top Risk Signals:
{top_signals}

{report_text}"""
        return final_response

    # ─────────────────────────────────────────────────────────────────────────
    # PATH B: LIVE GOOGLE GEMINI GENAI AGENT
    # ─────────────────────────────────────────────────────────────────────────
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise RuntimeError("google-genai package is required for live agent runs. Please install google-genai.") from e

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not configured.")

    active_model = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    client = genai.Client(api_key=api_key)

    agent_logs.append({
        "type": "system",
        "time": datetime.utcnow().strftime("%H:%M:%S"),
        "message": f"🤖 Initializing Gemini Multi-Tool Agent [Model: {active_model}]..."
    })

    tool_declarations = [
        lookup_icd10_code,
        check_drug_combination,
        get_provider_billing_history,
        calculate_rxhcc_risk_score,
        generate_fwa_report,
    ]

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)]
        )
    ]

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        tools=tool_declarations,
        temperature=0.1,
    )

    max_turns = 12
    for turn in range(max_turns):
        response = client.models.generate_content(
            model=active_model,
            contents=contents,
            config=config,
        )

        candidate = response.candidates[0]
        model_content = candidate.content
        contents.append(model_content)

        # Check for function calls
        function_calls = []
        for part in model_content.parts:
            if part.function_call:
                function_calls.append(part.function_call)

        if not function_calls:
            # Model produced final text response
            final_text = ""
            for part in model_content.parts:
                if part.text:
                    final_text += part.text
            agent_logs.append({
                "type": "system",
                "time": datetime.utcnow().strftime("%H:%M:%S"),
                "message": "✅ Agent completed investigation and delivered final synthesis."
            })
            return final_text

        # Execute function calls
        tool_response_parts = []
        for call in function_calls:
            name = call.name
            args = dict(call.args.items()) if hasattr(call.args, "items") else (call.args or {})

            agent_logs.append({
                "type": "tool_call",
                "tool": name,
                "args": args,
                "time": datetime.utcnow().strftime("%H:%M:%S"),
            })

            # Map function name to implementation
            func_map = {
                "lookup_icd10_code": lookup_icd10_code,
                "check_drug_combination": check_drug_combination,
                "get_provider_billing_history": get_provider_billing_history,
                "calculate_rxhcc_risk_score": calculate_rxhcc_risk_score,
                "generate_fwa_report": generate_fwa_report,
            }

            if name in func_map:
                try:
                    result = func_map[name](**args)
                except Exception as ex:
                    result = {"error": str(ex)}
            else:
                result = {"error": f"Unknown tool: {name}"}

            agent_logs.append({
                "type": "tool_response",
                "tool": name,
                "result": result if not isinstance(result, str) else {"status": "SUCCESS", "text_len": len(result)},
                "time": datetime.utcnow().strftime("%H:%M:%S"),
            })

            # Send function response back as role="user" with function_response Part
            res_dict = result if isinstance(result, dict) else {"result": str(result)}
            tool_response_parts.append(
                types.Part.from_function_response(name=name, response=res_dict)
            )

        contents.append(
            types.Content(
                role="user",
                parts=tool_response_parts,
            )
        )

    raise RuntimeError(f"Agent reached maximum execution depth ({max_turns} turns) without terminating.")


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    test_prompt = """Investigate the following Medicare Part D claim for Fraud, Waste & Abuse:

Claim ID:         CLM-2026-TEST
Beneficiary ID:   BEN-M-98765
ICD-10 Codes:     E11.9
NDC Drug Codes:   00002143380
Provider NPI:     1122334455
Claim Amount:     $185.00
Date of Service:  2026-09-11

Please conduct a full investigation following the standard FWA workflow."""

    logs = []
    print("Running CLI test in Demo Mode...")
    out = run_fwa_investigation(test_prompt, logs, demo_mode=True)
    print("\n--- AGENT OUTPUT ---")
    try:
        print(out)
    except UnicodeEncodeError:
        print(out.encode("ascii", errors="replace").decode("ascii"))
    print(f"\nCollected {len(logs)} trace events.")
