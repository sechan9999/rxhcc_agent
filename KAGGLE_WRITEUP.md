# RxHCC FWA Investigation Agent
## Catching Medicare Fraud Before It's Paid — With Multi-Agent AI

**Track:** Agents for Good
**Author:** Gyver (sechan9999)
**Live Demo:** https://rxhcc-app.vercel.app/
**GitHub:** https://github.com/sechan9999/rxhcc_risk_adjustment

---

## 1. Problem Definition (~300 words)

Medicare and Medicaid fraud, waste, and abuse (FWA) costs American taxpayers an estimated
$60–100 billion every year — roughly 10% of total program spending. Traditional rule-based
detection systems catch fraud *after* payment, forcing the government into costly "pay and chase"
recoveries. By the time an investigation concludes, the money is often gone.

The RxHCC (Prescription Drug Hierarchical Condition Category) model is used by CMS to adjust
risk scores for Medicare Part D drug plans. It maps ICD-10 diagnosis codes to risk coefficients
that determine plan reimbursements. When providers manipulate these codes — through upcoding,
unbundling, phantom billing, or impossible diagnoses — they inflate payments at the expense of
legitimate beneficiaries.

Key fraud patterns this system targets:
- **Gender-diagnosis mismatches**: Male patients billed for female-specific conditions (e.g., ovarian cysts, breast cancer)
- **Opioid "holy trinity"**: Co-prescribing opioids + benzodiazepines + muscle relaxants — the #1 pill-mill signature
- **Provider network anomalies**: Prescribers billing in the 99th percentile vs. peers for controlled substances
- **Claim amount outliers**: Charges 3–10x the provider's own average

The question this project answers: *Can a multi-agent AI system identify these patterns
autonomously, before payment is released, with explainable reasoning a compliance officer
can act on?*

---

## 2. Solution Design (~500 words)

### Architecture Overview

The system is built as a **hierarchical multi-agent pipeline** using Google ADK and Gemini 2.0 Flash.
Each agent has a single responsibility, its own tool set, and communicates through structured
output keys — mirroring how a real investigations team operates.

```
User submits claim
        │
        ▼
┌─────────────────────────────┐
│   fwa_orchestrator          │  ← Root agent. Drives the pipeline.
│   (Gemini 2.0 Flash)        │    Delivers final verdict.
└────────────┬────────────────┘
             │ delegates to (in sequence):
    ┌────────┴──────────────────────────────────┐
    │                  │                        │
    ▼                  ▼                        ▼
claim_analyzer    risk_scorer            report_writer
    │                  │                        │
    └─ lookup_icd10    ├─ check_drug_combo      └─ generate_fwa_report
                       ├─ get_provider_history
                       └─ calculate_rxhcc_score
```

### Agent Roles

**Claim Analyzer** validates every ICD-10 code in the claim using structured lookup tools.
It extracts beneficiary gender from the ID, checks code validity, flags gender restrictions,
and produces a structured claim summary passed downstream.

**Risk Scorer** runs the three scoring tools in sequence: (1) drug combination analysis to
catch pill-mill patterns, (2) provider billing history for peer-benchmarking and watchlist
checks, and (3) the composite RxHCC risk model that weighs all signals into a 0–100% fraud
probability with a CLEAR / FLAG / ESCALATE verdict.

**Report Writer** takes the structured findings from both upstream agents and calls
`generate_fwa_report` to produce a compliance-grade investigation report — including
immediate action items specific to the verdict level.

**Orchestrator** is the brain. It runs the full pipeline, passes context between agents,
and delivers a concise executive summary + full report to the user.

### Tools (Function Calling)

| Tool | Description |
|------|-------------|
| `lookup_icd10_code(code)` | Validates ICD-10-CM codes; returns description, severity, gender restriction |
| `get_provider_billing_history(npi)` | Returns 90-day billing stats, peer percentile, anomaly score, watchlist flags |
| `check_drug_combination(ndc_codes)` | Detects dangerous drug combos: opioid+benzo, poly-pharmacy, Schedule II stacking |
| `calculate_rxhcc_risk_score(...)` | Composite FWA scorer: gender mismatch + drug risk + provider risk + claim amount |
| `generate_fwa_report(...)` | Produces formatted SIU-ready compliance report with actionable next steps |

### Why Agents vs. a Single LLM Call?

A single prompt to Gemini would mix validation, scoring, and reporting — making the reasoning
opaque and the output hard to audit. The multi-agent design:
1. **Isolates failures** — a bad provider lookup doesn't corrupt the drug check
2. **Enables specialization** — each agent's system prompt is tuned for its task
3. **Creates audit trails** — every sub-agent output is a named artifact (`output_key`)
4. **Scales naturally** — new fraud signal types become new sub-agents without refactoring

---

## 3. Course Concepts Applied (~400 words)

This project directly applies the five core concepts from the 5-Day AI Agents course:

**Day 1 — Foundation Models & Prompting**
The orchestrator's instruction uses structured prompt engineering: explicit workflow steps,
output format templates, and threshold tables for verdict decisions. Each sub-agent's
instruction is a focused system prompt with chain-of-thought scaffolding.

**Day 2 — Function Calling & Tool Use**
All five detection functions are registered as ADK tools. The risk_scorer agent decides
*which* tools to call and in what order based on the claim data — it will skip drug checks
if no NDC codes are present, demonstrating conditional tool use.

**Day 3 — Agentic Pipelines**
The `output_key` mechanism chains agents: claim_analyzer writes to `claim_analysis`,
risk_scorer reads it and writes to `risk_assessment`, report_writer synthesizes both.
This is the Kaggle course's sequential pipeline pattern applied to healthcare compliance.

**Day 4 — Multi-Agent Systems**
The orchestrator-subagent topology directly implements the course's hierarchical agent
architecture. The orchestrator never calls tools directly — it delegates entirely, keeping
responsibilities cleanly separated.

**Day 5 — Production & Evaluation**
The `sample_claims.json` file defines four labeled test cases with expected verdicts,
enabling systematic evaluation. The Streamlit UI exposes the agent trace so users can
inspect sub-agent reasoning, not just the final answer.

---

## 4. Results & Impact (~300 words)

### Demo Scenarios

| Scenario | Risk Score | Verdict | Key Signal |
|----------|-----------|---------|------------|
| Diabetic maintenance claim | 0% | ✅ CLEAR | No anomalies |
| Opioid + benzo combo | 65% | ⚠️ FLAG | Drug combination + flagged pharmacy |
| Male patient + breast cancer + pill-mill provider | 95% | 🚨 ESCALATE | Gender mismatch + provider watchlist + opioid triad |
| ESRD + heart failure + wrong drug | 35% | ⚠️ FLAG | Severity mismatch + claim amount outlier |

### Value Delivered

- **Pre-payment detection**: Flags fraud *before* the claim is processed, not after
- **Explainable verdicts**: Every risk factor is named, numbered, and traceable to a tool call
- **Actionable reports**: ESCALATE reports include specific SIU referral steps;
  FLAG reports list exact documentation to request from providers
- **Extensible**: Adding a new fraud signal is a single new tool + tool registration

### Real-World Integration Path

1. Replace mock provider DB with live NPI Registry API + CMS PSV data
2. Replace simulated risk scorer with the actual SageMaker RxHCC model endpoint
3. Deploy on Google Cloud Run with ADK's built-in session management
4. Add streaming UI for real-time agent trace visibility

---

## 5. Technical Appendix (~300 words)

### File Structure
```
rxhcc_fwa_agent/
├── tools.py          # 5 ADK tool functions + reference data
├── agents.py         # 4 ADK agents + Runner factory
├── app.py            # Streamlit demo UI
├── sample_claims.json # 4 labeled test cases
└── requirements.txt
```

### Running Locally
```bash
git clone https://github.com/sechan9999/rxhcc_risk_adjustment
cd rxhcc_risk_adjustment
pip install -r requirements.txt
export GOOGLE_API_KEY="your_key_here"
streamlit run app.py
```

### Key ADK Patterns Used
- `Agent(model, name, instruction, tools, sub_agents, output_key)` — agent definition
- `Runner(agent, app_name, session_service)` — execution engine
- `InMemorySessionService` — stateful session management across agent turns
- `event.is_final_response()` — streaming event loop for UI integration

### Limitations & Future Work
- Provider database is simulated; production requires CMS PECOS / NPI Registry integration
- ICD-10 reference covers ~15 codes for demo; full CM has 70,000+ codes
- No feedback loop yet — investigator verdicts should retrain the risk model over time
- Multi-claim batch analysis (network-level fraud detection) is the natural next step

---

*Word count: ~1,800 words — well within the 2,500-word limit.*
*This writeup focuses on architecture and decisions, trusting the live demo and video to show execution.*
