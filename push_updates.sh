#!/usr/bin/env bash
# push_updates.sh — adds the 5 missing files to sechan9999/rxhcc_agent
# Run from inside your local rxhcc_agent clone:
#   cd rxhcc_agent && bash /path/to/push_updates.sh

set -e
SCRIPT_DIR="$(dirname "$0")"

echo "► Copying missing files into repo..."
cp "$SCRIPT_DIR/.gitignore"                    .gitignore
cp "$SCRIPT_DIR/.env.example"                  .env.example
cp "$SCRIPT_DIR/LICENSE"                       LICENSE
cp "$SCRIPT_DIR/rxhcc_fwa_agent_demo.ipynb"   rxhcc_fwa_agent_demo.ipynb

echo "► Staging..."
git add .gitignore .env.example LICENSE rxhcc_fwa_agent_demo.ipynb

echo "► Committing..."
git commit -m "chore: add .gitignore, .env.example, LICENSE, Kaggle demo notebook

- .gitignore: excludes .env, __pycache__, .ipynb_checkpoints
- .env.example: documents GOOGLE_API_KEY setup
- LICENSE: MIT
- rxhcc_fwa_agent_demo.ipynb: self-contained Kaggle notebook with
  all 3 demo scenarios (CLEAR / FLAG / ESCALATE) and batch eval"

echo "► Pushing..."
git push origin main

echo ""
echo "✅ Done! Repo now has all required files for Kaggle submission."
echo "   https://github.com/sechan9999/rxhcc_agent"
echo ""
echo "Last step — add repo description + topics on GitHub:"
echo "  Description : Medicare FWA detection via Google ADK multi-agent system"
echo "  Topics      : google-adk gemini healthcare-ai fraud-detection streamlit kaggle"
