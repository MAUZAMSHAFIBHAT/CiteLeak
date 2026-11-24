"""Fill your GitHub and demo links into README.md.

  python tools/set_links.py                       # repo link from poster/config.json
  python tools/set_links.py https://xyz.streamlit.app   # also set the live-demo link
"""
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
repo = json.loads((root / "poster" / "config.json").read_text())["repo_url"].rstrip("/")
readme = root / "README.md"
s = readme.read_text()
s = s.replace("https://github.com/YOUR-USERNAME/citeleak", repo)
if len(sys.argv) > 1:
    s = s.replace("DEMO_URL_HERE", f"[{sys.argv[1]}]({sys.argv[1]})")
readme.write_text(s)
print("README links set: repo =", repo, "| demo =", sys.argv[1] if len(sys.argv) > 1 else "(not set yet)")
