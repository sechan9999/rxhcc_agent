"""
tools.py — RxHCC FWA Detection Tools
All functions here are registered as Google ADK tools used by the sub-agents.

Fraud, Waste & Abuse (FWA) patterns detected:
  1. Invalid / impossible ICD-10 code combinations
  2. Gender-diagnosis mismatches
  3. Poly-pharmacy & controlled-substance abuse (pill mills)
  4. High-risk provider billing anomalies (vs. peer benchmarks)
  5. Claim amount outliers
"""

from datetime import datetime
from typing import Optional

# ── ICD-10 Reference ──────────────────────────────────────────────────────────
ICD10_DB: dict[str, dict] = {
    "E11.9":   {"desc": "Type 2 diabetes mellitus without complications",           "severity": 1, "gender": None,   "hcc": True},
    "E11.65":  {"desc": "Type 2 diabetes mellitus with hyperglycemia",              "severity": 2, "gender": None,   "hcc": True},
    "E11.40":  {"desc": "Type 2 diabetes mellitus with diabetic neuropathy",        "severity": 3, "gender": None,   "hcc": True},
    "I10":     {"desc": "Essential (primary) hypertension",                         "severity": 1, "gender": None,   "hcc": False},
    "I50.9":   {"desc": "Heart failure, unspecified",                               "severity": 4, "gender": None,   "hcc": True},
    "N18.6":   {"desc": "End-stage renal disease",                                  "severity": 5, "gender": None,   "hcc": True},
    "Z79.4":   {"desc": "Long-term current use of insulin",                         "severity": 1, "gender": None,   "hcc": False},
    "C50.911": {"desc": "Malignant neoplasm, right female breast",                  "severity": 5, "gender": "F",    "hcc": True},
    "C50.912": {"desc": "Malignant neoplasm, left female breast",                   "severity": 5, "gender": "F",    "hcc": True},
    "N81.10":  {"desc": "Cystocele, unspecified (female pelvic floor disorder)",    "severity": 2, "gender": "F",    "hcc": False},
    "N40.0":   {"desc": "Benign prostatic hyperplasia (BPH)",                       "severity": 2, "gender": "M",    "hcc": False},
    "F11.10":  {"desc": "Opioid abuse, uncomplicated",                              "severity": 3, "gender": None,   "hcc": True},
    "F14.10":  {"desc": "Cocaine abuse, uncomplicated",                             "severity": 3, "gender": None,   "hcc": True},
    "G89.29":  {"desc": "Other chronic pain",                                       "severity": 2, "gender": None,   "hcc": False},
    "Z87.891": {"desc": "Personal history of nicotine dependence",                  "severity": 1, "gender": None,   "hcc": False},
}

# ── NDC Drug Reference ────────────────────────────────────────────────────────
NDC_DB: dict[str, dict] = {
    "00406051201": {"name": "OxyContin (oxycodone) 80 mg",    "schedule": "II",  "risk": "HIGH"},
    "00406051301": {"name": "OxyContin (oxycodone) 160 mg",   "schedule": "II",  "risk": "HIGH"},
    "59011049010": {"name": "Alprazolam (Xanax) 2 mg",        "schedule": "IV",  "risk": "MEDIUM"},
    "00093733056": {"name": "Methadone HCl 10 mg",            "schedule": "II",  "risk": "HIGH"},
    "00591058105": {"name": "Gabapentin 800 mg",               "schedule": "V",   "risk": "LOW"},
    "00074334713": {"name": "Adalimumab (Humira)",             "schedule": None,  "risk": "HIGH"},
    "00088502905": {"name": "Semaglutide (Ozempic)",           "schedule": None,  "risk": "LOW"},
    "00002143480": {"name": "Insulin glargine (Lantus)",       "schedule": None,  "risk": "LOW"},
    "65162010850": {"name": "Carisoprodol (Soma) 350 mg",     "schedule": "IV",  "risk": "MEDIUM"},
}

# ── Simulated High-Risk Provider Registry ─────────────────────────────────────
PROVIDER_DB: dict[str, dict] = {
    "1234567890": {
        "name": "Dr. James Rutherford MD",
        "specialty": "Pain Management",
        "state": "FL",
        "total_claims_90d": 1842,
        "avg_claim_amount": 1640.00,
        "peer_percentile": 99,
        "controlled_substance_pct": 0.78,
        "anomaly_score": 0.91,
        "flags": ["TOP_1PCT_OPIOID_PRESCRIBER", "MULTI_PATIENT_OVERLAP", "CASH_PAY_ONLY"],
    },
    "9876543210": {
        "name": "Sunrise Specialty Pharmacy LLC",
        "specialty": "Retail Pharmacy",
        "state": "TX",
        "total_claims_90d": 3210,
        "avg_claim_amount": 820.00,
        "peer_percentile": 98,
        "controlled_substance_pct": 0.65,
        "anomaly_score": 0.87,
        "flags": ["DISPENSING_WITHOUT_VALID_RX", "UNUSUALLY_HIGH_REFILL_RATE", "OUT_OF_NETWORK_BILLING"],
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 1 — ICD-10 Lookup
# ═══════════════════════════════════════════════════════════════════════════════
def lookup_icd10_code(code: str) -> dict:
    """
    Validate an ICD-10-CM diagnosis code and return its clinical details.

    Args:
        code: ICD-10-CM code string, e.g. "E11.9" or "C50.911"

    Returns:
        dict with keys: valid (bool), code, description, severity (1-5),
        gender_restriction (M/F/None), hcc_relevant (bool).
    """
    code = code.upper().strip()
    entry = ICD10_DB.get(code)
    if entry:
        return {
            "valid": True,
            "code": code,
            "description": entry["desc"],
            "severity": entry["severity"],
            "gender_restriction": entry["gender"],
            "hcc_relevant": entry["hcc"],
        }
    return {
        "valid": False,
        "code": code,
        "description": "Code not found in ICD-10-CM database — may be invalid or unbillable",
        "severity": 0,
        "gender_restriction": None,
        "hcc_relevant": False,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 2 — Provider Billing History
# ═══════════════════════════════════════════════════════════════════════════════
def get_provider_billing_history(npi: str) -> dict:
    """
    Retrieve a provider's 90-day billing statistics and risk flags from the
    NPI registry and peer-comparison database.

    Args:
        npi: 10-digit National Provider Identifier string

    Returns:
        dict with provider info, billing stats, peer percentile, anomaly score,
        and a list of specific risk flags.
    """
    if npi in PROVIDER_DB:
        p = PROVIDER_DB[npi]
        return {"npi": npi, **p}

    # Default: low-risk provider
    return {
        "npi": npi,
        "name": f"Provider NPI {npi}",
        "specialty": "General Practice",
        "state": "N/A",
        "total_claims_90d": 87,
        "avg_claim_amount": 310.00,
        "peer_percentile": 42,
        "controlled_substance_pct": 0.04,
        "anomaly_score": 0.08,
        "flags": [],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 3 — Drug Combination Check
# ═══════════════════════════════════════════════════════════════════════════════
def check_drug_combination(ndc_codes: list[str]) -> dict:
    """
    Analyze a set of NDC drug codes for dangerous or suspicious combinations.
    Detects: poly-pharmacy abuse, opioid+benzo lethal combos, pill-mill patterns.

    Args:
        ndc_codes: list of 11-digit NDC strings, e.g. ["00406051201", "59011049010"]

    Returns:
        dict with resolved drug names, DEA schedules, combination_risk level,
        requires_prior_auth flag, and a list of specific flags.
    """
    resolved = []
    for ndc in ndc_codes:
        resolved.append(NDC_DB.get(ndc, {
            "name": f"Unknown drug ({ndc})",
            "schedule": None,
            "risk": "UNKNOWN",
        }))

    sched_ii   = [d for d in resolved if d.get("schedule") == "II"]
    sched_iv   = [d for d in resolved if d.get("schedule") == "IV"]
    high_risk  = [d for d in resolved if d.get("risk") == "HIGH"]

    flags = []
    has_opioid = any("oxycodone" in d["name"].lower() or "methadone" in d["name"].lower()
                     for d in resolved)
    has_benzo  = any("alprazolam" in d["name"].lower() for d in resolved)
    has_soma   = any("carisoprodol" in d["name"].lower() for d in resolved)

    if has_opioid and has_benzo:
        flags.append("OPIOID_BENZO_COMBINATION — HIGH OVERDOSE MORTALITY RISK (FDA Black Box)")
    if has_opioid and has_soma:
        flags.append("OPIOID_MUSCLE_RELAXANT_COMBINATION — Documented abuse triad")
    if len(sched_ii) >= 2:
        flags.append("MULTIPLE_SCHEDULE_II_CONTROLLED_SUBSTANCES — Requires DEA review")
    if len(high_risk) >= 3:
        flags.append("POLY_PHARMACY_HIGH_RISK — 3+ high-risk drugs co-prescribed")

    return {
        "drugs": [d["name"] for d in resolved],
        "schedules_present": sorted(set(d["schedule"] for d in resolved if d.get("schedule"))),
        "combination_risk": "HIGH" if flags else ("MEDIUM" if sched_ii else "LOW"),
        "requires_prior_auth": bool(sched_ii or sched_iv),
        "flags": flags,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 4 — RxHCC Risk Scorer
# ═══════════════════════════════════════════════════════════════════════════════
def calculate_rxhcc_risk_score(
    beneficiary_id: str,
    icd10_codes: list[str],
    ndc_codes: list[str],
    claim_amount: float,
    provider_npi: str,
) -> dict:
    """
    Calculate the composite FWA fraud risk score for a Medicare Part D claim
    using the RxHCC model logic + rule-based anomaly detection.

    Args:
        beneficiary_id: CMS beneficiary identifier (suffix M/F used for gender inference)
        icd10_codes:    list of ICD-10-CM diagnosis codes on the claim
        ndc_codes:      list of NDC drug codes dispensed
        claim_amount:   total dollar amount billed
        provider_npi:   10-digit NPI of the prescribing/dispensing provider

    Returns:
        dict with risk_score (0.0-1.0), verdict, recommendation, and risk_factors list.
    """
    score = 0.0
    risk_factors: list[str] = []

    # Infer gender from beneficiary ID suffix (M/F convention)
    beneficiary_gender: Optional[str] = None
    if "-M-" in beneficiary_id.upper():
        beneficiary_gender = "M"
    elif "-F-" in beneficiary_id.upper():
        beneficiary_gender = "F"

    # ── Rule 1: ICD-10 validity & severity ───────────────────────────────────
    codes_upper = [c.upper().strip() for c in icd10_codes]
    for code in codes_upper:
        info = ICD10_DB.get(code, {})
        severity = info.get("severity", 0)
        if not info:
            score += 0.05
            risk_factors.append(f"INVALID_ICD10_CODE: {code} — not billable")
        elif severity >= 4:
            score += 0.10
            risk_factors.append(f"HIGH_SEVERITY_DIAGNOSIS: {code} ({info['desc']})")

        # Rule 1b: Gender-diagnosis mismatch
        gender_req = info.get("gender")
        if gender_req and beneficiary_gender and gender_req != beneficiary_gender:
            score += 0.40
            risk_factors.append(
                f"GENDER_DIAGNOSIS_MISMATCH: {code} ({info['desc']}) "
                f"requires patient gender={gender_req}, beneficiary is {beneficiary_gender}"
            )

    # ── Rule 2: Drug combination risk ────────────────────────────────────────
    if ndc_codes:
        drug_check = check_drug_combination(ndc_codes)
        if drug_check["combination_risk"] == "HIGH":
            score += 0.25
            risk_factors.extend(drug_check["flags"])
        elif drug_check["combination_risk"] == "MEDIUM":
            score += 0.10
            if drug_check["flags"]:
                risk_factors.extend(drug_check["flags"])

    # ── Rule 3: Provider anomaly ──────────────────────────────────────────────
    provider = get_provider_billing_history(provider_npi)
    if provider["anomaly_score"] > 0.75:
        score += 0.30
        risk_factors.append(
            f"HIGH_RISK_PROVIDER: NPI {provider_npi} — {provider['name']} "
            f"(anomaly score {provider['anomaly_score']:.2f}, "
            f"{provider['peer_percentile']}th percentile)"
        )
        risk_factors.extend(provider["flags"])
    elif provider["anomaly_score"] > 0.40:
        score += 0.10
        risk_factors.append(f"ELEVATED_PROVIDER_RISK: NPI {provider_npi} — {provider['name']}")

    # ── Rule 4: Claim amount anomaly ─────────────────────────────────────────
    expected_max = provider.get("avg_claim_amount", 500) * 3
    if claim_amount > expected_max:
        score += 0.15
        risk_factors.append(
            f"CLAIM_AMOUNT_OUTLIER: ${claim_amount:,.2f} is "
            f"{claim_amount / max(provider['avg_claim_amount'], 1):.1f}x provider average"
        )
    elif claim_amount > 5000:
        score += 0.05
        risk_factors.append(f"HIGH_CLAIM_AMOUNT: ${claim_amount:,.2f}")

    # ── Final verdict ─────────────────────────────────────────────────────────
    score = min(round(score, 2), 1.0)

    if score < 0.30:
        verdict = "CLEAR"
        recommendation = "Approve claim for payment. No further action required."
    elif score < 0.70:
        verdict = "FLAG_FOR_REVIEW"
        recommendation = (
            "Hold claim. Request supporting documentation from provider: "
            "medical records, prior authorizations, and patient encounter notes."
        )
    else:
        verdict = "ESCALATE"
        recommendation = (
            "Do NOT pay claim. Refer immediately to the Special Investigations Unit (SIU). "
            "Consider provider suspension pending investigation."
        )

    return {
        "beneficiary_id": beneficiary_id,
        "risk_score": score,
        "risk_score_pct": f"{score:.0%}",
        "verdict": verdict,
        "recommendation": recommendation,
        "risk_factors": risk_factors,
        "model_version": "RxHCC-FWA-v2.1",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 5 — FWA Investigation Report Generator
# ═══════════════════════════════════════════════════════════════════════════════
def generate_fwa_report(
    claim_id: str,
    risk_score: float,
    verdict: str,
    risk_factors: list[str],
    provider_name: str,
    provider_npi: str,
    provider_anomaly_score: float,
    provider_flags: list[str],
    drugs_prescribed: list[str],
    drug_combination_risk: str,
    drug_flags: list[str],
    recommendation: str,
) -> str:
    """
    Generate a structured FWA investigation report formatted for compliance officers
    and the CMS Special Investigations Unit (SIU).

    Returns a formatted plain-text report string.
    """
    bar = "=" * 65
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    verdict_emoji = {"CLEAR": "✅", "FLAG_FOR_REVIEW": "⚠️", "ESCALATE": "🚨"}.get(verdict, "❓")

    factor_lines = "\n".join(
        f"  [{i+1:02d}] {f}" for i, f in enumerate(risk_factors)
    ) if risk_factors else "  None identified."

    drug_lines = "\n".join(f"  • {d}" for d in drugs_prescribed) if drugs_prescribed else "  None"
    dflag_lines = "\n".join(f"  ⚠ {f}" for f in drug_flags) if drug_flags else "  None"
    pflag_lines = "\n".join(f"  ⚠ {f}" for f in provider_flags) if provider_flags else "  None"

    action_section = ""
    if verdict == "ESCALATE":
        action_section = """
IMMEDIATE ACTIONS REQUIRED:
  1. Place payment hold on this claim and all pending claims from NPI above
  2. Open SIU case and assign lead investigator within 24 hours
  3. Notify CMS Program Integrity (PI) contractor
  4. Preserve audit trail — do not contact provider before SIU authorization
  5. Cross-reference beneficiary against other flagged providers
"""
    elif verdict == "FLAG_FOR_REVIEW":
        action_section = """
DOCUMENTATION TO REQUEST FROM PROVIDER:
  1. Signed patient encounter notes for date of service
  2. Prior authorization documentation for controlled substances
  3. Diagnostic test results supporting high-severity ICD-10 codes
  4. Referral documentation (if applicable)
  Response deadline: 30 days (per 42 CFR § 405.980)
"""

    report = f"""
{bar}
  RxHCC FWA INVESTIGATION REPORT — {verdict_emoji} {verdict}
{bar}
  Generated : {now}
  Claim ID  : {claim_id}
  Model     : RxHCC-FWA-v2.1 | Google ADK + Gemini 2.0 Flash
{bar}

RISK SUMMARY
  Score        : {risk_score:.0%}
  Verdict      : {verdict_emoji}  {verdict}
  Recommendation: {recommendation}

RISK FACTORS IDENTIFIED  ({len(risk_factors)} total)
{factor_lines}
{action_section}
PROVIDER ANALYSIS
  Name            : {provider_name}
  NPI             : {provider_npi}
  Anomaly Score   : {provider_anomaly_score:.2f} / 1.00
  Provider Flags  :
{pflag_lines}

DRUG / PHARMACY REVIEW
  Drugs Prescribed:
{drug_lines}
  Combination Risk: {drug_combination_risk}
  Drug Flags      :
{dflag_lines}

{bar}
  This report is generated by the RxHCC FWA Agent System.
  CONFIDENTIAL — For authorized CMS / SIU personnel only.
  Unauthorized disclosure may violate 18 U.S.C. § 1905.
{bar}
""".strip()

    return report
