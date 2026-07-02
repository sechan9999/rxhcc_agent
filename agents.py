"""
agents.py — RxHCC FWA Agent
Built with Google GenAI SDK (direct call) for the Kaggle "5-Day AI Agents: Intensive Vibe Coding" Capstone (Agents for Good track).
Bypasses Google ADK, invoking google-genai directly with a manual tool loop.
"""

import os
import time
from google import genai
from google.genai import types
from tools import (
    lookup_icd10_code,
    get_provider_billing_history,
    check_drug_combination,
    calculate_rxhcc_risk_score,
    generate_fwa_report,
)

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

SYSTEM_INSTRUCTION = """You are RxHCC-FWA, an autonomous Medicare Part D Fraud, Waste & Abuse investigator powered by the RxHCC (Prescription Drug Hierarchical Condition Category) risk-adjustment model.

Medicare fraud costs Americans $60–100 billion per year. Your job is to catch it BEFORE payment is made, using your 5 tools in sequence.

═♐═ MANDATORY INVESTIGATION WORKFLOW ═══
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
  beneficiary_id — exact string from claim
  icd10_codes — Python list of strings
  ndc_codes — Python list of strings ([] if none)
  claim_amount — float (strip $ and commas)
  provider_npi — exact NPI string

STEP 5 — GENERATE COMPLIANCE REPORT
Call generate_fwa_report with all findings collected above.
Pass provider_name, provider_npi, provider_anomaly_score, provider_flags, drugs_prescribed, drugs_prescribed, drug_combination_risk, drug_flags, risk_score (float 0–1), verdict, risk_factors (list), recommendation.

═══ VERDICT THRESHOLDS ═══
CLEAR (risk_score 0.00–0.29) → Approve for payment
FLAG_FOR_REVIEW (risk_score 0.30–0.69) → Hold; request documentation
ESCALATE (risk_score 0.70–1.00) → Block payment; refer to SIU

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

IMPORTANT: You MUST call all 5 tools before writing your final answer. Do not skip any tool. Do not guess — always call the tool to get real data."""

def run_fwa_investigation(prompt: str, agent_logs: list, model: str = None) -> str:
    """
    Execute the FWA investigation using the google-genai client directly, running a manual loop to handle tool calls / function calling.
    """
    # Force/check demo mode
    demo_active = (os.environ.get("DEMO_MODE") == "1") or not os.environ.get("GOOGLE_API_KEY")
    
    if demo_active:
        agent_logs.append("[system]\nInitializing local simulation run (Demo Mode)...")
        # Parse fields from the prompt
        claim_id = ""
        beneficiary_id = ""
        icd10_codes = []
        ndc_codes = []
        provider_npi = ""
        claim_amount_val = 0.0

        for line in prompt.split("\n"):
            if line.startswith("Claim ID:"):
                claim_id = line.split(":", 1)[1].strip()
            elif line.startswith("Beneficiary ID:"):
                beneficiary_id = line.split(":", 1)[1].strip()
            elif line.startswith("ICD-10 Codes:"):
                codes_str = line.split(":", 1)[1].strip()
                icd10_codes = [c.strip() for c in codes_str.split(",") if c.strip()]
            elif line.startswith("NDC Drug Codes:"):
                drugs_str = line.split(":", 1)[1].strip()
                if drugs_str and drugs_str.lower() != "none":
                    ndc_codes = [d.strip() for d in drugs_str.split(",") if d.strip()]
            elif line.startswith("Provider NPI:"):
                provider_npi = line.split(":", 1)[1].strip()
            elif line.startswith("Claim Amount:"):
                amt_str = line.split(":", 1)[1].strip()
                amt_str = amt_str.replace("$", "").replace(",", "")
                try:
                    claim_amount_val = float(amt_str)
                except ValueError:
                    claim_amount_val = 0.0

        # Step 1: Validate ICD-10 codes
        icd10_results = []
        for code in icd10_codes:
            agent_logs.append(f"[agent calling tool]\nTool: lookup_icd10_code\nArgs: {{'code': '{code}'}}")
            res = lookup_icd10_code(code)
            agent_logs.append(f"[tool result]\n{res}")
            icd10_results.append(res)
            time.sleep(0.1)

        # Step 2: Check drug combinations
        agent_logs.append(f"[agent calling tool]\nTool: check_drug_combination\nArgs: {{'ndc_codes': {ndc_codes}}}")
        drug_res = check_drug_combination(ndc_codes)
        agent_logs.append(f"[tool result]\n{drug_res}")
        time.sleep(0.1)

        # Step 3: Pull provider risk profile
        agent_logs.append(f"[agent calling tool]\nTool: get_provider_billing_history\nArgs: {{'npi': '{provider_npi}'}}")
        provider_res = get_provider_billing_history(provider_npi)
        agent_logs.append(f"[tool result]\n{provider_res}")
        time.sleep(0.1)

        # Step 4: Calculate composite risk score
        agent_logs.append(f"[agent calling tool]\nTool: calculate_rxhcc_risk_score\nArgs: {{'beneficiary_id': '{beneficiary_id}', 'icd10_codes': {icd10_codes}, 'ndc_codes': {ndc_codes}, 'claim_amount': {claim_amount_val}, 'provider_npi': '{provider_npi}'}}")
        risk_res = calculate_rxhcc_risk_score(
            beneficiary_id=beneficiary_id,
            icd10_codes=icd10_codes,
            ndc_codes=ndc_codes,
            claim_amount=claim_amount_val,
            provider_npi=provider_npi
        )
        agent_logs.append(f"[tool result]\n{risk_res}")
        time.sleep(0.1)

        # Step 5: Generate compliance report
        agent_logs.append(f"[agent calling tool]\nTool: generate_fwa_report\nArgs: {{'claim_id': '{claim_id}', ...}}")
        report = generate_fwa_report(
            claim_id=claim_id,
            risk_score=risk_res["risk_score"],
            verdict=risk_res["verdict"],
            risk_factors=risk_res["risk_factors"],
            provider_name=provider_res["name"],
            provider_npi=provider_res["npi"],
            provider_anomaly_score=provider_res["anomaly_score"],
            provider_flags=provider_res["flags"],
            drugs_prescribed=drug_res["drugs"],
            drug_combination_risk=drug_res["combination_risk"],
            drug_flags=drug_res["flags"],
            recommendation=risk_res["recommendation"]
        )
        agent_logs.append(f"[tool result]\n{report}")
        time.sleep(0.1)

        # Format final text matching instructions
        verdict = risk_res["verdict"]
        risk_pct = int(risk_res["risk_score"] * 100)
        signals_str = "\n".join([f"{i+1}. {factor}" for i, factor in enumerate(risk_res["risk_factors"][:3])])
        if not signals_str:
            signals_str = "None"
        
        final_text = f"""INVESTIGATION COMPLETE
Claim: {claim_id} | Beneficiary: {beneficiary_id} | Amount: ${claim_amount_val:,.2f}
Risk Score: {risk_pct}% | Verdict: {verdict}

Top Risk Signals:
{signals_str}

{report}"""
        return final_text

    # Use selected model or default fallback
    target_model = model or MODEL

    # Create the GenAI client.
    # It automatically picks up GOOGLE_API_KEY from environment variables.
    client = genai.Client()

    # Define tools list
    tools_list = [
        lookup_icd10_code,
        get_provider_billing_history,
        check_drug_combination,
        calculate_rxhcc_risk_score,
        generate_fwa_report,
    ]

    # Initialize the history with the user prompt
    history = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)]
        )
    ]
    agent_logs.append(f"[system]\nInitializing Gemini client using {target_model} and starting manual tool execution loop...")

    max_turns = 10
    for turn in range(max_turns):
        agent_logs.append(f"[system]\nCalling Gemini (Turn {turn + 1})...")
        
        # Exponential backoff retry loop for rate limiting/429
        max_retries = 3
        response = None
        for retry in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=target_model,
                    contents=history,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        tools=tools_list,
                        temperature=0.0,
                    )
                )
                break
            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str
                if is_rate_limit and retry < max_retries - 1:
                    wait_time = (retry + 1) * 6
                    agent_logs.append(
                        f"[system]\nRate limit hit (429/Resource Exhausted). "
                        f"Retrying in {wait_time}s... (Attempt {retry+1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                    continue
                raise e

        # Append assistant response to history
        if response.candidates and response.candidates[0].content:
            history.append(response.candidates[0].content)

        # Log agent thoughts if any text was returned
        if response.text:
            agent_logs.append(f"[agent]\n{response.text.strip()}")

        # Check for function calls
        function_calls = response.function_calls
        if not function_calls:
            # Loop ends when no more tools are requested
            return response.text or ""

        tool_response_parts = []
        for call in function_calls:
            name = call.name
            args = call.args
            agent_logs.append(f"[agent calling tool]\nTool: {name}\nArgs: {args}")
            
            # Match and execute the correct function
            func = None
            if name == "lookup_icd10_code":
                func = lookup_icd10_code
            elif name == "get_provider_billing_history":
                func = get_provider_billing_history
            elif name == "check_drug_combination":
                func = check_drug_combination
            elif name == "calculate_rxhcc_risk_score":
                func = calculate_rxhcc_risk_score
            elif name == "generate_fwa_report":
                func = generate_fwa_report

            if func:
                try:
                    result = func(**args)
                    agent_logs.append(f"[tool result]\n{result}")
                except Exception as e:
                    result = {"error": str(e)}
                    agent_logs.append(f"[tool error]\n{e}")
            else:
                result = {"error": f"Tool {name} not found"}
                agent_logs.append(f"[tool error]\nTool {name} not found")

            tool_response_parts.append(
                types.Part.from_function_response(
                    name=name,
                    response=result
                )
            )

        # Add the tool execution results to history
        history.append(
            types.Content(
                role="tool",
                parts=tool_response_parts
            )
        )

    return "Error: Maximum tool execution turns exceeded."

if __name__ == "__main__":
    # Test script for manual CLI testing
    TEST_CLAIM = """
    Investigate this Medicare Part D claim for FWA:
    Claim ID: CLM-TEST-003
    Beneficiary ID: BNF-M-77542
    ICD-10 Codes: C50.911, E11.9
    NDC Codes: 00406051201, 59011049010
    Provider NPI: 1234567890
    Claim Amount: $8,400.00
    Date of Service: 2024-11-15
    """
    # Set demo mode environment variable for testing
    os.environ["DEMO_MODE"] = os.getenv("DEMO_MODE", "1")
    print("Running FWA Investigation test...")
    logs = []
    try:
        report = run_fwa_investigation(TEST_CLAIM, logs)
        print("\n=== Investigation Report ===")
        print(report)
        print("\n=== Logs ===")
        for log in logs:
            print(log)
    except Exception as e:
        print(f"Error executing investigation: {e}")
