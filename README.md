# RxHCC FWA Investigation Agent 🏥

> **Kaggle 5-Day AI Agents Intensive Vibe Coding Capstone — Agents for Good**

An autonomous multi-agent system that investigates Medicare Part D claims for **Fraud, Waste & Abuse (FWA)** *before* payment is released — using the Google Gen AI SDK + Gemini 2.0 Flash.

## Architecture

```
User submits claim
      │
      ▼
fwa_orchestrator  (Gemini 2.0 Flash)
  ├── claim_analyzer   → lookup_icd10_code
  ├── risk_scorer      → calculate_rxhcc_risk_score, check_drug_combination, get_provider_billing_history
  └── report_writer    → generate_fwa_report
```

## Fraud Patterns Detected

- **Gender-Diagnosis Mismatches**: e.g., Male patient billed for female-specific conditions.
- **Controlled Substance Cocktails**: Opioid + benzodiazepine + muscle relaxant "holy trinity" (pill-mill signature).
- **Billing Anomalies**: Provider billing in the 99th percentile versus peers for controlled substances.
- **Claim Cost Mismatches**: Claim amounts 3–10x the provider's own historic average.

## Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/sechan9999/rxhcc_agent.git
   cd rxhcc_agent
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   export GOOGLE_API_KEY="your_api_key_here"
   ```

4. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

## Repository Structure

| File | Description |
| :--- | :--- |
| `tools.py` | 5 ADK tool functions (ICD-10 lookup, provider history, drug combo checker, risk scorer, report generator) |
| `agents.py` | 4 Google ADK agents & Orchestration logic |
| `app.py` | Streamlit demo user interface with 3 pre-loaded FWA investigation scenarios |
| `sample_claims.json` | 4 labeled test claims for validation |
| `requirements.txt` | Python library dependencies |
| `KAGGLE_WRITEUP.md` | Capstone writeup for the Agents for Good track |

## Track
**Agents for Good** — Medicare fraud costs $60–100B/year. This agent system catches potential fraud before the money leaves.
