"""Write requirements.txt for the online demo (Streamlit Community Cloud).

The hosted demo runs in replay mode, so it only needs a few light packages, pinned to the
exact versions in your local environment so the saved attack models load correctly.
Run from the project folder with the venv active:  python tools/pin_demo_requirements.py
"""
from importlib.metadata import version
from pathlib import Path

pkgs = ["streamlit", "numpy", "scipy", "scikit-learn", "rank_bm25"]
lines = [f"{p}=={version(p)}" for p in pkgs]
Path(__file__).resolve().parent.parent.joinpath("requirements.txt").write_text(
    "# Light requirements for the hosted demo (replay mode).\n"
    "# Full experiment dependencies: requirements-full.txt (used by run.sh)\n" + "\n".join(lines) + "\n")
print("wrote requirements.txt:\n" + "\n".join(lines))
