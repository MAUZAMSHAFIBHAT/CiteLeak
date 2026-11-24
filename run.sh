#!/usr/bin/env bash
# One command: environment, tests, full experiment, figures, poster.
# Extra arguments are passed to the experiment, e.g.  ./run.sh --model Qwen/Qwen2.5-0.5B-Instruct
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
if [ ! -d .venv ]; then "$PY" -m venv .venv; fi
source .venv/bin/activate
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements-full.txt
python -m pytest -q tests/test_core.py
python -m citeleak.run all "$@"
echo
echo "Done. Results: results/RESULTS.md   Poster: poster/CiteLeak_poster.pdf"
echo "Demo:  source .venv/bin/activate && streamlit run app.py"
