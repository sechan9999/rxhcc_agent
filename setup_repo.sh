#!/usr/bin/env bash
# =============================================================================
# setup_repo.sh
# Merges the existing rxhcc_risk_adjustment app with the new agent files
# and pushes everything to sechan9999/rxhcc_agent.
#
# Run this from ANY folder on your machine:
#   chmod +x setup_repo.sh && ./setup_repo.sh
# =============================================================================

set -e  # stop on first error

EXISTING_REPO="https://github.com/sechan9999/rxhcc_risk_adjustment.git"
NEW_REPO="https://github.com/sechan9999/rxhcc_agent.git"
WORK_DIR="rxhcc_agent"

# ── Where are the new agent files? ────────────────────────────────────────────
# Default: same folder as this script. Override with: AGENT_FILES=/your/path ./setup_repo.sh
AGENT_FILES="${AGENT_FILES:-$(dirname "$0")}"

echo ""
echo "=========================================="
echo "  RxHCC Agent — Repo Setup Script"
echo "=========================================="
echo "Source repo : $EXISTING_REPO"
echo "Target repo : $NEW_REPO"
echo "Agent files : $AGENT_FILES"
echo ""

# ── 1. Clone existing app ─────────────────────────────────────────────────────
echo "► Cloning existing rxhcc_risk_adjustment..."
if [ -d "$WORK_DIR" ]; then
  echo "  '$WORK_DIR' already exists — pulling latest instead."
  cd "$WORK_DIR"
  git pull origin main 2>/dev/null || git pull origin master 2>/dev/null || true
  cd ..
else
  git clone "$EXISTING_REPO" "$WORK_DIR"
fi

cd "$WORK_DIR"

# ── 2. Copy new agent files into repo ────────────────────────────────────────
echo "► Copying agent files..."

mkdir -p agent

cp "$AGENT_FILES/tools.py"            agent/tools.py
cp "$AGENT_FILES/agents.py"           agent/agents.py
cp "$AGENT_FILES/app.py"              agent/app.py
cp "$AGENT_FILES/sample_claims.json"  agent/sample_claims.json
cp "$AGENT_FILES/requirements.txt"    agent/requirements.txt
cp "$AGENT_FILES/KAGGLE_WRITEUP.md"   KAGGLE_WRITEUP.md

echo "  Copied: agent/tools.py, agents.py, app.py, sample_claims.json, requirements.txt"
echo "  Copied: KAGGLE_WRITEUP.md (repo root)"

# ── 3. Write a top-level README ───────────────────────────────────────────────
echo "► Writing README.md..."
cat > README.md << 'EOF'
# RxHCC FWA Investigation Agent 🏥

> **Kaggle 5-Day AI Agents Intensive Vibe Coding Capstone — Agents for Good**

An autonomous multi-agent system that investigates Medicare Part D claims for
**Fraud, Waste & Abuse (FWA)** *before* payment is released — using
Google ADK + Gemini 2.0 Flash.

## Live Demo
🔗 https://rxhcc-app.vercel.app/

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
- Gender-diagnosis mismatches (male patient billed for female-specific conditions)
- Opioid + benzodiazepine + muscle relaxant "holy trinity" (pill-mill signature)
- Provider billing in 99th percentile vs. peers for controlled substances
- Claim amounts 3–10x the provider's own average

## Quick Start
```bash
cd agent
pip install -r requirements.txt
export GOOGLE_API_KEY="your_key_here"
streamlit run app.py
```

## Files
| File | Description |
|------|-------------|
| `agent/tools.py` | 5 ADK tool functions — ICD-10 lookup, provider history, drug check, risk scorer, report gen |
| `agent/agents.py` | 4 Google ADK agents + Runner factory |
| `agent/app.py` | Streamlit demo UI with 3 pre-loaded FWA scenarios |
| `agent/sample_claims.json` | 4 labeled test claims for evaluation |
| `KAGGLE_WRITEUP.md` | Capstone writeup (Agents for Good track) |

## Original App
The original RxHCC Risk Adjustment + FWA Detection app (Amazon Nova) lives in the repo root.
The new agent system is in the `agent/` subdirectory.

## Track
**Agents for Good** — Medicare fraud costs $60–100B/year.
This system catches it before the money leaves.
EOF

# ── 4. Point remote to new repo ───────────────────────────────────────────────
echo "► Setting remote to $NEW_REPO..."
git remote remove origin 2>/dev/null || true
git remote add origin "$NEW_REPO"

# ── 5. Stage and commit ───────────────────────────────────────────────────────
echo "► Staging all files..."
git add -A

echo "► Committing..."
git commit -m "feat: add multi-agent FWA investigation system (Google ADK + Gemini)

- agent/tools.py: 5 ADK tools (ICD-10 lookup, provider history, drug combo, RxHCC scorer, report gen)
- agent/agents.py: 4-agent topology (claim_analyzer, risk_scorer, report_writer, fwa_orchestrator)
- agent/app.py: Streamlit demo UI with 3 FWA scenario presets
- agent/sample_claims.json: 4 labeled test cases with expected verdicts
- KAGGLE_WRITEUP.md: Capstone writeup for Agents for Good track
- README.md: updated with architecture diagram and quick-start

Kaggle 5-Day AI Agents Intensive Vibe Coding Capstone"

# ── 6. Push ────────────────────────────────────────────────────────────────────
echo "► Pushing to $NEW_REPO ..."
echo "   (GitHub will prompt for your username + personal access token)"
echo ""
git push -u origin main 2>/dev/null || git push -u origin master

echo ""
echo "=========================================="
echo "  ✅  Done! Repo is live at:"
echo "  https://github.com/sechan9999/rxhcc_agent"
echo "=========================================="
