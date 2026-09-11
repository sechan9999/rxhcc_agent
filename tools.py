"""
tools.py — RxHCC FWA Detection Tools & Clinical Rule Engine
Integrated clinical intelligence fusing Medicare Part D guidelines,
RxHCC risk adjustment methodologies, and OIG fraud detection patterns.

FWA Patterns Detected:
  1. Invalid / unbillable ICD-10 codes
  2. Mutually exclusive diagnosis conflicts (e.g. Type 1 vs Type 2 Diabetes)
  3. Gender-diagnosis mismatches (e.g. male with female breast/ovarian neoplasm)
  4. GLP-1 off-label weight-loss abuse without approved indication
  5. Potential HCC upcoding (uncomplicated diagnosis mapped to high-risk HCC)
  6. Lethal polypharmacy & controlled substance triads (FDA Black Box warnings)
  7. High-risk provider billing anomalies & peer percentile outliers
  8. Claim amount financial outliers vs. peer benchmarks
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
import random

# ═══════════════════════════════════════════════════════════════════════════════
# 1. ICD-10 & RxHCC CLINICAL DATABASE
# ═══════════════════════════════════════════════════════════════════════════════
ICD10_DB: Dict[str, Dict[str, Any]] = {
    "E11.9": {
        "desc": "Type 2 diabetes mellitus without complications",
        "severity": 1,
        "gender": None,
        "hcc": True,
        "hcc_category": "HCC19",
        "category": "Endocrine",
    },
    "E11.65": {
        "desc": "Type 2 diabetes mellitus with hyperglycemia",
        "severity": 2,
        "gender": None,
        "hcc": True,
        "hcc_category": "HCC18",
        "category": "Endocrine",
    },
    "E11.40": {
        "desc": "Type 2 diabetes mellitus with diabetic neuropathy",
        "severity": 3,
        "gender": None,
        "hcc": True,
        "hcc_category": "HCC18",
        "category": "Endocrine",
    },
    "E10.9": {
        "desc": "Type 1 diabetes mellitus without complications",
        "severity": 2,
        "gender": None,
        "hcc": True,
        "hcc_category": "HCC19",
        "category": "Endocrine",
    },
    "E66.01": {
        "desc": "Morbid (severe) obesity due to excess calories",
        "severity": 2,
        "gender": None,
        "hcc": False,
        "hcc_category": None,
        "category": "Metabolic",
    },
    "I10": {
        "desc": "Essential (primary) hypertension",
        "severity": 1,
        "gender": None,
        "hcc": False,
        "hcc_category": None,
        "category": "Cardiovascular",
    },
    "I50.9": {
        "desc": "Heart failure, unspecified (Congestive Heart Failure)",
        "severity": 4,
        "gender": None,
        "hcc": True,
        "hcc_category": "HCC85",
        "category": "Cardiovascular",
    },
    "N18.6": {
        "desc": "End-stage renal disease (ESRD)",
        "severity": 5,
        "gender": None,
        "hcc": True,
        "hcc_category": "HCC136",
        "category": "Renal",
    },
    "Z79.4": {
        "desc": "Long-term (current) use of insulin",
        "severity": 1,
        "gender": None,
        "hcc": False,
        "hcc_category": None,
        "category": "Treatment History",
    },
    "Z86.39": {
        "desc": "Personal history of other endocrine, nutritional and metabolic disease",
        "severity": 1,
        "gender": None,
        "hcc": False,
        "hcc_category": None,
        "category": "History",
    },
    "C50.911": {
        "desc": "Malignant neoplasm, unspecified site of right female breast",
        "severity": 5,
        "gender": "F",
        "hcc": True,
        "hcc_category": "HCC8",
        "category": "Oncology",
    },
    "C50.912": {
        "desc": "Malignant neoplasm, unspecified site of left female breast",
        "severity": 5,
        "gender": "F",
        "hcc": True,
        "hcc_category": "HCC8",
        "category": "Oncology",
    },
    "N81.10": {
        "desc": "Cystocele, unspecified (female pelvic floor disorder)",
        "severity": 2,
        "gender": "F",
        "hcc": False,
        "hcc_category": None,
        "category": "Genitourinary",
    },
    "N40.0": {
        "desc": "Benign prostatic hyperplasia (BPH) without urinary obstruction",
        "severity": 2,
        "gender": "M",
        "hcc": False,
        "hcc_category": None,
        "category": "Genitourinary",
    },
    "F11.10": {
        "desc": "Opioid abuse, uncomplicated",
        "severity": 3,
        "gender": None,
        "hcc": True,
        "hcc_category": "HCC55",
        "category": "Behavioral Health",
    },
    "F14.10": {
        "desc": "Cocaine abuse, uncomplicated",
        "severity": 3,
        "gender": None,
        "hcc": True,
        "hcc_category": "HCC55",
        "category": "Behavioral Health",
    },
    "G89.29": {
        "desc": "Other chronic post-thoracotomy and post-operative pain",
        "severity": 2,
        "gender": None,
        "hcc": False,
        "hcc_category": None,
        "category": "Pain / Neurological",
    },
    "J44.9": {
        "desc": "Chronic obstructive pulmonary disease (COPD), unspecified",
        "severity": 3,
        "gender": None,
        "hcc": True,
        "hcc_category": "HCC111",
        "category": "Respiratory",
    },
    "J45.909": {
        "desc": "Unspecified asthma, uncomplicated",
        "severity": 2,
        "gender": None,
        "hcc": False,
        "hcc_category": None,
        "category": "Respiratory",
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# 2. NDC DRUG DATABASE & DEA SCHEDULES
# ═══════════════════════════════════════════════════════════════════════════════
NDC_DB: Dict[str, Dict[str, Any]] = {
    # Controlled Substances
    "00406051201": {
        "name": "OxyContin (oxycodone HCl) 80 mg",
        "schedule": "II",
        "risk": "HIGH",
        "class": "Opioid Agonist",
        "prior_auth": True,
    },
    "00406051301": {
        "name": "OxyContin (oxycodone HCl) 160 mg",
        "schedule": "II",
        "risk": "HIGH",
        "class": "Opioid Agonist",
        "prior_auth": True,
    },
    "59011049010": {
        "name": "Alprazolam (Xanax) 2 mg",
        "schedule": "IV",
        "risk": "MEDIUM",
        "class": "Benzodiazepine",
        "prior_auth": True,
    },
    "65162010850": {
        "name": "Carisoprodol (Soma) 350 mg",
        "schedule": "IV",
        "risk": "MEDIUM",
        "class": "Muscle Relaxant",
        "prior_auth": True,
    },
    "00093733056": {
        "name": "Methadone HCl 10 mg",
        "schedule": "II",
        "risk": "HIGH",
        "class": "Opioid Maintenance",
        "prior_auth": True,
    },
    "00591058105": {
        "name": "Gabapentin (Neurontin) 800 mg",
        "schedule": "V",
        "risk": "LOW",
        "class": "GABA Analog / Anticonvulsant",
        "prior_auth": False,
    },
    # GLP-1 Agonists (High-Value Off-label Fraud Target)
    "00169406012": {
        "name": "Ozempic (semaglutide) 2 mg/3 mL pen",
        "schedule": None,
        "risk": "HIGH_COST",
        "class": "GLP-1 Receptor Agonist",
        "prior_auth": True,
        "is_glp1": True,
        "approved_indications": ["E11"],
    },
    "00169406013": {
        "name": "Wegovy (semaglutide) 2.4 mg auto-injector",
        "schedule": None,
        "risk": "HIGH_COST",
        "class": "GLP-1 Receptor Agonist",
        "prior_auth": True,
        "is_glp1": True,
        "approved_indications": ["E66"],
    },
    "00002751501": {
        "name": "Trulicity (dulaglutide) 1.5 mg/0.5 mL",
        "schedule": None,
        "risk": "HIGH_COST",
        "class": "GLP-1 Receptor Agonist",
        "prior_auth": True,
        "is_glp1": True,
        "approved_indications": ["E11"],
    },
    # Standard & Chronic Medications
    "00002143380": {
        "name": "Metformin HCl (Glucophage) 500 mg",
        "schedule": None,
        "risk": "LOW",
        "class": "Biguanide (Antidiabetic)",
        "prior_auth": False,
    },
    "00002143480": {
        "name": "Insulin glargine (Lantus) 100 units/mL",
        "schedule": None,
        "risk": "LOW",
        "class": "Basal Insulin",
        "prior_auth": False,
    },
    "00071015540": {
        "name": "Amlodipine besylate (Norvasc) 10 mg",
        "schedule": None,
        "risk": "LOW",
        "class": "Calcium Channel Blocker",
        "prior_auth": False,
    },
    "00781150610": {
        "name": "Lisinopril 20 mg",
        "schedule": None,
        "risk": "LOW",
        "class": "ACE Inhibitor",
        "prior_auth": False,
    },
    "00074334713": {
        "name": "Adalimumab (Humira) 40 mg/0.8 mL",
        "schedule": None,
        "risk": "HIGH_COST",
        "class": "TNF-alpha Inhibitor Biologic",
        "prior_auth": True,
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# 3. HIGH-RISK PROVIDER PROFILING DATABASE
# ═══════════════════════════════════════════════════════════════════════════════
PROVIDER_DB: Dict[str, Dict[str, Any]] = {
    "1234567890": {
        "name": "Dr. James Rutherford MD (Sunshine Pain Clinic)",
        "specialty": "Interventional Pain Management",
        "state": "FL",
        "total_claims_90d": 1842,
        "avg_claim_amount": 1640.00,
        "peer_percentile": 99,
        "controlled_substance_pct": 0.82,
        "anomaly_score": 0.94,
        "flags": [
            "TOP_1PCT_OPIOID_PRESCRIBER",
            "MULTI_PATIENT_OVERLAP_SUSPICION",
            "HIGH_RATIO_CASH_DISPENSING",
            "FDA_TRINITY_PRESCRIBER",
        ],
    },
    "9876543210": {
        "name": "Sunrise Specialty Pharmacy LLC",
        "specialty": "Retail & Specialty Pharmacy",
        "state": "TX",
        "total_claims_90d": 3210,
        "avg_claim_amount": 820.00,
        "peer_percentile": 98,
        "controlled_substance_pct": 0.68,
        "anomaly_score": 0.88,
        "flags": [
            "DISPENSING_WITHOUT_PRIOR_AUTH",
            "UNUSUALLY_HIGH_EARLY_REFILL_RATE",
            "OUT_OF_NETWORK_BILLING_SPIKE",
        ],
    },
    "1928374650": {
        "name": "Dr. Mark Henderson MD (Apex Wellness & Aesthetics)",
        "specialty": "Anti-Aging / Wellness Clinic",
        "state": "CA",
        "total_claims_90d": 950,
        "avg_claim_amount": 1850.00,
        "peer_percentile": 96,
        "controlled_substance_pct": 0.12,
        "anomaly_score": 0.84,
        "flags": [
            "MASS_GLP1_OFF_LABEL_PRESCRIBING",
            "ABSENCE_OF_DOCUMENTED_PRIMARY_CARE_TREATMENT",
            "HIGH_BILLING_CONCENTRATION_COSMETIC",
        ],
    },
    "1122334455": {
        "name": "Dr. Sarah Jenkins MD",
        "specialty": "Internal Medicine / Primary Care",
        "state": "OH",
        "total_claims_90d": 140,
        "avg_claim_amount": 310.00,
        "peer_percentile": 42,
        "controlled_substance_pct": 0.04,
        "anomaly_score": 0.08,
        "flags": [],
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# 4. MUTUAL EXCLUSION & CLINICAL CONFLICT RULES
# ═══════════════════════════════════════════════════════════════════════════════
CONFLICT_RULES = [
    {
        "id": "CONFLICT_DIABETES_T1_T2",
        "name": "Type 1 & Type 2 Diabetes Co-billing",
        "set_a": ["E10"],
        "set_b": ["E11"],
        "severity": "CRITICAL",
        "score_penalty": 0.45,
        "message": "Mutually exclusive diagnoses: Type 1 Diabetes (E10) and Type 2 Diabetes (E11) co-billed on the same claim.",
    },
    {
        "id": "CONFLICT_ACTIVE_VS_REMISSION",
        "name": "Active Diabetes with History of Remission",
        "set_a": ["E11"],
        "set_b": ["Z86.39"],
        "severity": "WARNING",
        "score_penalty": 0.20,
        "message": "Co-existence of active Diabetes (E11) and remission/history code (Z86.39) indicates potential coding conflict.",
    },
    {
        "id": "CONFLICT_ASTHMA_COPD",
        "name": "Asthma & COPD Overlap Review",
        "set_a": ["J45"],
        "set_b": ["J44"],
        "severity": "WARNING",
        "score_penalty": 0.15,
        "message": "Asthma-COPD Overlap (ACO) requires specific clinical documentation to substantiate dual therapy.",
    },
]

# Upcoding target mappings
HCC_UPCODING_MAP = {
    "HCC18": {
        "desc": "Diabetes with Chronic Complications",
        "expected_icd": ["E11.65", "E11.40"],
        "uncomplicated_icd": ["E11.9"],
        "score_penalty": 0.35,
        "risk_weight": 0.302,
    },
    "HCC85": {
        "desc": "Congestive Heart Failure",
        "expected_icd": ["I50.9"],
        "uncomplicated_icd": ["I10"],
        "score_penalty": 0.40,
        "risk_weight": 0.331,
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 1 — ICD-10 Clinical Lookup
# ═══════════════════════════════════════════════════════════════════════════════
def lookup_icd10_code(code: str) -> dict:
    """
    Validate an ICD-10-CM diagnosis code and return its clinical & RxHCC details.

    Args:
        code: ICD-10-CM code string, e.g. "E11.9", "C50.911", "I10"

    Returns:
        dict with clinical metadata: validity, description, severity (1-5),
        gender_restriction, hcc_relevant, hcc_category, clinical category.
    """
    code_clean = code.upper().strip()
    entry = ICD10_DB.get(code_clean)
    if entry:
        return {
            "valid": True,
            "code": code_clean,
            "description": entry["desc"],
            "severity": entry["severity"],
            "gender_restriction": entry["gender"],
            "hcc_relevant": entry["hcc"],
            "hcc_category": entry["hcc_category"],
            "clinical_category": entry["category"],
        }

    return {
        "valid": False,
        "code": code_clean,
        "description": "Code not found in CMS ICD-10 database — invalid or unbillable under Medicare Part D",
        "severity": 0,
        "gender_restriction": None,
        "hcc_relevant": False,
        "hcc_category": None,
        "clinical_category": "Unknown",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 2 — Provider Billing History & Outlier Profiling
# ═══════════════════════════════════════════════════════════════════════════════
def get_provider_billing_history(npi: str) -> dict:
    """
    Retrieve provider 90-day billing statistics and peer anomaly flags from NPI registry.

    Args:
        npi: 10-digit National Provider Identifier string

    Returns:
        dict with provider profiling, peer percentile rank, anomaly score, and risk flags.
    """
    npi_clean = str(npi).strip()
    if npi_clean in PROVIDER_DB:
        p = PROVIDER_DB[npi_clean]
        return {"npi": npi_clean, **p}

    # Low-risk default profile
    return {
        "npi": npi_clean,
        "name": f"Provider NPI {npi_clean} (Community Practice)",
        "specialty": "Internal Medicine / Family Practice",
        "state": "N/A",
        "total_claims_90d": 95,
        "avg_claim_amount": 320.00,
        "peer_percentile": 45,
        "controlled_substance_pct": 0.03,
        "anomaly_score": 0.07,
        "flags": [],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 3 — Drug Combination & Lethal Interaction Review
# ═══════════════════════════════════════════════════════════════════════════════
def check_drug_combination(ndc_codes: list[str]) -> dict:
    """
    Analyze NDC drug codes for high-risk combinations, FDA Black Box warnings,
    and GLP-1 weight-loss off-label abuse patterns.

    Args:
        ndc_codes: list of NDC code strings (formatted or unhyphenated)

    Returns:
        dict with drug names, DEA schedules, combination risk rating, prior auth requirements,
        and specific risk warning flags.
    """
    resolved = []
    for raw_ndc in ndc_codes:
        clean_ndc = raw_ndc.replace("-", "").strip()
        matched = None
        for db_ndc, info in NDC_DB.items():
            if clean_ndc == db_ndc or clean_ndc.startswith(db_ndc[:9]):
                matched = info
                break
        if matched:
            resolved.append(matched)
        else:
            resolved.append({
                "name": f"NDC {raw_ndc} (Unclassified Drug)",
                "schedule": None,
                "risk": "UNKNOWN",
                "class": "General Medication",
                "prior_auth": False,
            })

    flags = []
    has_opioid = any("oxycodone" in d["name"].lower() or "methadone" in d["name"].lower() for d in resolved)
    has_benzo = any("alprazolam" in d["name"].lower() for d in resolved)
    has_soma = any("carisoprodol" in d["name"].lower() for d in resolved)
    has_glp1 = any(d.get("is_glp1", False) for d in resolved)

    sched_ii = [d for d in resolved if d.get("schedule") == "II"]
    sched_iv = [d for d in resolved if d.get("schedule") == "IV"]
    high_cost = [d for d in resolved if d.get("risk") == "HIGH_COST"]

    if has_opioid and has_benzo and has_soma:
        flags.append("PILL_MILL_HOLY_TRINITY — Lethal Opioid + Benzo + Soma combination (Highest Overdose Risk)")
    elif has_opioid and has_benzo:
        flags.append("OPIOID_BENZO_COMBINATION — FDA Black Box Warning: Profound sedation, respiratory depression, coma, death")
    elif has_opioid and has_soma:
        flags.append("OPIOID_MUSCLE_RELAXANT_SYNERGY — Documented drug abuse triad")

    if len(sched_ii) >= 2:
        flags.append("MULTIPLE_SCHEDULE_II_SUBSTANCES — Requires immediate DEA & SIU prescription verification")

    if has_glp1:
        flags.append("HIGH_VALUE_GLP1_DETECTED — Requires verification of FDA-approved clinical indication (Diabetes/Obesity)")

    combination_risk = "LOW"
    if any("HOLY_TRINITY" in f for f in flags) or (has_opioid and has_benzo) or len(sched_ii) >= 2:
        combination_risk = "HIGH"
    elif sched_ii or sched_iv or high_cost or flags:
        combination_risk = "MEDIUM"

    requires_prior_auth = any(d.get("prior_auth", False) for d in resolved)

    return {
        "drugs": [d["name"] for d in resolved],
        "schedules_present": sorted(set(d["schedule"] for d in resolved if d.get("schedule"))),
        "combination_risk": combination_risk,
        "requires_prior_auth": requires_prior_auth,
        "flags": flags,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 4 — Composite RxHCC Fraud, Waste & Abuse Risk Scorer
# ═══════════════════════════════════════════════════════════════════════════════
def calculate_rxhcc_risk_score(
    beneficiary_id: str,
    icd10_codes: list[str],
    ndc_codes: list[str],
    claim_amount: float,
    provider_npi: str,
) -> dict:
    """
    Calculate composite FWA risk score using clinical rules, ICD mutual exclusions,
    GLP-1 off-label checking, HCC upcoding tests, and provider anomaly models.

    Args:
        beneficiary_id: CMS beneficiary ID (suffix -M- or -F- indicates gender)
        icd10_codes: list of ICD-10 diagnosis strings
        ndc_codes: list of NDC pharmacy drug strings
        claim_amount: total billed dollar amount
        provider_npi: 10-digit NPI of prescribing provider

    Returns:
        dict with composite score (0.00-1.00), verdict, recommendations, and all flagged factors.
    """
    score = 0.0
    risk_factors: list[str] = []

    # 1. Infer beneficiary gender
    beneficiary_gender: Optional[str] = None
    ben_upper = beneficiary_id.upper()
    if "-M-" in ben_upper or ben_upper.endswith("-M"):
        beneficiary_gender = "M"
    elif "-F-" in ben_upper or ben_upper.endswith("-F"):
        beneficiary_gender = "F"

    # 2. ICD-10 Code Validity & Severity
    clean_icds = [c.upper().strip() for c in icd10_codes if c.strip()]
    icd_prefixes = [c.split(".")[0] for c in clean_icds]

    for code in clean_icds:
        info = lookup_icd10_code(code)
        if not info["valid"]:
            score += 0.15
            risk_factors.append(f"INVALID_ICD10_CODE: '{code}' is not a valid billable Medicare diagnosis")
        elif info["severity"] >= 4:
            score += 0.10
            risk_factors.append(f"HIGH_SEVERITY_DIAGNOSIS: {code} — {info['description']} (Severity {info['severity']}/5)")

        # Gender mismatch
        gender_req = info.get("gender_restriction")
        if gender_req and beneficiary_gender and gender_req != beneficiary_gender:
            score += 0.45
            risk_factors.append(
                f"GENDER_DIAGNOSIS_MISMATCH: {code} ({info['description']}) requires patient gender={gender_req}, "
                f"but beneficiary profile indicates {beneficiary_gender}"
            )

    # 3. Mutually Exclusive Diagnosis Conflicts
    for rule in CONFLICT_RULES:
        has_a = any(any(c.startswith(prefix) for prefix in rule["set_a"]) for c in icd_prefixes)
        has_b = any(any(c.startswith(prefix) for prefix in rule["set_b"]) for c in icd_prefixes)
        if has_a and has_b:
            score += rule["score_penalty"]
            risk_factors.append(f"DIAGNOSIS_CONFLICT: {rule['message']}")

    # 4. Drug Analysis & GLP-1 Off-Label Check
    clean_ndcs = [n.strip() for n in ndc_codes if n.strip()]
    if clean_ndcs:
        drug_res = check_drug_combination(clean_ndcs)
        if drug_res["combination_risk"] == "HIGH":
            score += 0.30
        elif drug_res["combination_risk"] == "MEDIUM":
            score += 0.12
        risk_factors.extend(drug_res["flags"])

        # Check GLP-1 specific off-label usage
        for raw_ndc in clean_ndcs:
            c_ndc = raw_ndc.replace("-", "").strip()
            for db_ndc, d_info in NDC_DB.items():
                if (c_ndc == db_ndc or c_ndc.startswith(db_ndc[:9])) and d_info.get("is_glp1"):
                    approved_cats = d_info.get("approved_indications", [])
                    has_indication = any(any(c.startswith(cat) for cat in approved_cats) for c in icd_prefixes)
                    if not has_indication:
                        score += 0.40
                        risk_factors.append(
                            f"GLP1_OFF_LABEL_ABUSE: High-cost {d_info['name']} prescribed without required clinical "
                            f"indication (requires ICD category {approved_cats}, billed: {clean_icds})"
                        )

    # 5. Upcoding Evaluation
    # Check if uncomplicated code is paired with excessive billing or unsupported severity
    if "E11.9" in clean_icds and ("HCC18" in clean_icds or claim_amount > 1200):
        score += 0.30
        risk_factors.append(
            "POTENTIAL_HCC_UPCODING: Uncomplicated Diabetes (E11.9 / HCC19) billed with inflated severity/charges "
            "typical of complicated diabetes (HCC18), artificially boosting CMS capitation payments."
        )

    # 6. Provider Anomaly Evaluation
    provider = get_provider_billing_history(provider_npi)
    anomaly_score = provider.get("anomaly_score", 0.0)
    if anomaly_score > 0.75:
        score += 0.35
        risk_factors.append(
            f"HIGH_RISK_PROVIDER_OUTLIER: NPI {provider_npi} ({provider['name']}) "
            f"ranks in the {provider['peer_percentile']}th percentile for risk with anomaly score {anomaly_score:.2f}"
        )
        risk_factors.extend(provider.get("flags", []))
    elif anomaly_score > 0.40:
        score += 0.15
        risk_factors.append(f"ELEVATED_PROVIDER_RISK: NPI {provider_npi} ({provider['name']})")

    # 7. Claim Amount Outliers
    peer_avg = provider.get("avg_claim_amount", 350.00)
    if claim_amount > peer_avg * 3.5:
        score += 0.20
        ratio = claim_amount / max(peer_avg, 1)
        risk_factors.append(f"CLAIM_AMOUNT_OUTLIER: ${claim_amount:,.2f} is {ratio:.1f}x higher than peer specialty benchmark (${peer_avg:,.2f})")
    elif claim_amount > 4000.00:
        score += 0.10
        risk_factors.append(f"HIGH_VALUE_CLAIM: Claim amount ${claim_amount:,.2f} exceeds standard outpatient threshold")

    # Final score normalization
    final_score = min(round(score, 2), 1.00)

    # Assign verdict
    if final_score < 0.30:
        verdict = "CLEAR"
        recommendation = "Approve claim for payment release. Standard post-payment quality sampling applies."
    elif final_score < 0.70:
        verdict = "FLAG_FOR_REVIEW"
        recommendation = (
            "Place payment hold. Request comprehensive clinical documentation from provider: "
            "signed chart notes, lab diagnostics, and active prior authorization."
        )
    else:
        verdict = "ESCALATE"
        recommendation = (
            "Block payment immediately. Issue CMS Program Integrity freeze and refer case to "
            "the Special Investigations Unit (SIU) for False Claims Act (31 U.S.C. § 3729) inquiry."
        )

    return {
        "beneficiary_id": beneficiary_id,
        "risk_score": final_score,
        "risk_score_pct": f"{final_score:.0%}",
        "verdict": verdict,
        "recommendation": recommendation,
        "risk_factors": risk_factors,
        "model_version": "RxHCC-FWA-v3.0-Hybrid",
        "provider_npi": provider_npi,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 5 — CMS OIG & SIU Compliance Dossier Generator
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
    Generate a formal CMS OIG / SIU Fraud, Waste & Abuse Compliance Dossier.
    """
    bar = "=" * 70
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    verdict_emoji = {"CLEAR": "✅", "FLAG_FOR_REVIEW": "⚠️", "ESCALATE": "🚨"}.get(verdict, "❓")

    factor_lines = "\n".join(f"  [{i+1:02d}] {f}" for i, f in enumerate(risk_factors)) if risk_factors else "  None identified."
    drug_lines = "\n".join(f"  • {d}" for d in drugs_prescribed) if drugs_prescribed else "  None specified"
    dflag_lines = "\n".join(f"  ⚠ {f}" for f in drug_flags) if drug_flags else "  None"
    pflag_lines = "\n".join(f"  ⚠ {f}" for f in provider_flags) if provider_flags else "  None"

    action_section = ""
    if verdict == "ESCALATE":
        action_section = """
SIU ENFORCEMENT & REMEDIATION PROTOCOL:
  1. Mandatory payment freeze on claim and all pending remits from this NPI.
  2. Docket case with CMS Unified Program Integrity Contractor (UPIC).
  3. Prepare evidentiary package under False Claims Act (31 U.S.C. § 3729) and Anti-Kickback Statute.
  4. Coordinate with DEA Diversion Control Division if Schedule II/controlled substances involved.
  5. Cross-reference beneficiary identifier for concurrent cross-provider billing rings.
"""
    elif verdict == "FLAG_FOR_REVIEW":
        action_section = """
SPECIAL INVESTIGATION HOLD & MEDICAL RECORD REQUEST:
  1. Hold remittance pending submission of supporting documentation within 30 days (42 CFR § 405.980).
  2. Issue Records Request: Signed physician encounter notes, lab diagnostics, and therapy rationale.
  3. Validate active Electronic Prior Authorization (ePA) on file.
"""
    else:
        action_section = """
CLEARANCE & SETTLEMENT PROTOCOL:
  1. Claim clears automated clinical integrity and risk adjustment thresholds.
  2. Approved for standard electronic payment processing.
"""

    return f"""
{bar}
  MEDICARE PART D COMPLIANCE DOSSIER — {verdict_emoji} {verdict}
{bar}
  Timestamp : {now}
  Claim ID  : {claim_id}
  Auditor   : RxHCC-FWA Multi-Tool Autonomous Agent (CMS v3.0)
{bar}

EXECUTIVE ASSESSMENT
  Fraud Risk Score : {risk_score:.0%}
  Integrity Verdict: {verdict_emoji} {verdict}
  Recommendation   : {recommendation}

EVIDENTIARY RISK SIGNALS ({len(risk_factors)} Identified)
{factor_lines}
{action_section}
PRESCRIBING PROVIDER METRICS
  Provider Name : {provider_name}
  National NPI  : {provider_npi}
  Anomaly Score : {provider_anomaly_score:.2f} / 1.00
  Outlier Flags :
{pflag_lines}

PHARMACEUTICAL AUDIT
  Prescribed NDCs :
{drug_lines}
  Interaction Risk: {drug_combination_risk}
  Pharmacy Flags  :
{dflag_lines}

{bar}
  CONFIDENTIAL COMPLIANCE DOCUMENT — CMS SPECIAL INVESTIGATIONS UNIT ONLY
  Protected Health Information (PHI) under HIPAA 45 CFR Part 160/164.
{bar}
""".strip()


# ═══════════════════════════════════════════════════════════════════════════════
# 6. BATCH SYNTHETIC GENERATOR & SIMULATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
def generate_synthetic_claims_batch(n_claims: int = 25) -> list[dict]:
    """Generate a realistic randomized cohort of Medicare Part D claims for batch analytics."""
    templates = [
        # Clean diabetic
        {
            "scenario": "Clean T2D Maintenance",
            "icd": ["E11.9"],
            "ndc": ["00002143380"],
            "amt_range": (80, 250),
            "npi": "1122334455",
            "gender": "F",
        },
        # Clean Hypertension
        {
            "scenario": "Clean Hypertension",
            "icd": ["I10"],
            "ndc": ["00071015540"],
            "amt_range": (50, 180),
            "npi": "1122334455",
            "gender": "M",
        },
        # Pill Mill Triad
        {
            "scenario": "Pill Mill Controlled Substance Triad",
            "icd": ["G89.29", "F11.10"],
            "ndc": ["00406051201", "59011049010", "65162010850"],
            "amt_range": (1400, 3200),
            "npi": "1234567890",
            "gender": "M",
        },
        # GLP-1 Off-label
        {
            "scenario": "GLP-1 Off-label Abuse Scheme",
            "icd": ["I10"],
            "ndc": ["00169406012"],
            "amt_range": (950, 1900),
            "npi": "1928374650",
            "gender": "F",
        },
        # HCC Upcoding
        {
            "scenario": "Uncomplicated Diabetes Upcoding",
            "icd": ["E11.9", "HCC18"],
            "ndc": ["00002143380"],
            "amt_range": (1600, 2800),
            "npi": "9876543210",
            "gender": "M",
        },
        # Mutually exclusive conflict
        {
            "scenario": "Type 1 & Type 2 Diagnosis Conflict",
            "icd": ["E10.9", "E11.9"],
            "ndc": ["00002143480", "00002143380"],
            "amt_range": (600, 1400),
            "npi": "1122334455",
            "gender": "F",
        },
        # Gender mismatch
        {
            "scenario": "Gender-Diagnosis Mismatch",
            "icd": ["C50.911"],
            "ndc": [],
            "amt_range": (2500, 4800),
            "npi": "1122334455",
            "gender": "M",
        },
    ]

    claims = []
    for i in range(n_claims):
        t = random.choice(templates)
        claim_id = f"CLM-2026-{1000 + i}"
        ben_id = f"BEN-{t['gender']}-{90000 + i}"
        amt = round(random.uniform(*t["amt_range"]), 2)
        claims.append({
            "claim_id": claim_id,
            "beneficiary_id": ben_id,
            "scenario_type": t["scenario"],
            "icd10_codes": t["icd"],
            "ndc_codes": t["ndc"],
            "claim_amount": amt,
            "provider_npi": t["npi"],
        })
    return claims


def evaluate_claims_batch(claims: list[dict]) -> dict:
    """Evaluate a batch of claims with the RxHCC risk engine and compile portfolio analytics."""
    evaluated = []
    total_billed = 0.0
    escalated_value = 0.0
    flagged_value = 0.0
    clear_value = 0.0

    verdict_counts = {"CLEAR": 0, "FLAG_FOR_REVIEW": 0, "ESCALATE": 0}

    for c in claims:
        amt = float(c.get("claim_amount", 0.0))
        total_billed += amt
        res = calculate_rxhcc_risk_score(
            beneficiary_id=c["beneficiary_id"],
            icd10_codes=c["icd10_codes"],
            ndc_codes=c["ndc_codes"],
            claim_amount=amt,
            provider_npi=c["provider_npi"],
        )
        verdict = res["verdict"]
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

        if verdict == "ESCALATE":
            escalated_value += amt
        elif verdict == "FLAG_FOR_REVIEW":
            flagged_value += amt
        else:
            clear_value += amt

        evaluated.append({
            "claim_id": c["claim_id"],
            "beneficiary_id": c["beneficiary_id"],
            "scenario_type": c.get("scenario_type", "Custom"),
            "claim_amount": amt,
            "risk_score": res["risk_score"],
            "risk_score_pct": res["risk_score_pct"],
            "verdict": verdict,
            "factors_count": len(res["risk_factors"]),
            "top_factor": res["risk_factors"][0] if res["risk_factors"] else "None",
        })

    n = max(len(claims), 1)
    fraud_rate = (verdict_counts["ESCALATE"] + verdict_counts["FLAG_FOR_REVIEW"]) / n

    return {
        "total_claims": len(claims),
        "total_billed": total_billed,
        "clear_count": verdict_counts["CLEAR"],
        "flag_count": verdict_counts["FLAG_FOR_REVIEW"],
        "escalate_count": verdict_counts["ESCALATE"],
        "clear_value": clear_value,
        "flagged_value": flagged_value,
        "escalated_value": escalated_value,
        "potential_savings": escalated_value + (flagged_value * 0.5),
        "fraud_detection_rate": fraud_rate,
        "claims_data": evaluated,
    }
