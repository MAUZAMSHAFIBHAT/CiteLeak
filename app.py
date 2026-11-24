"""CiteLeak interactive demo.

    streamlit run app.py              # replay mode: instant, uses cached experiment results
    streamlit run app.py -- --live    # live mode: loads the LLM and answers new questions

Tabs: (1) use MediAssist, (2) play the attacker, (3) results.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from citeleak import attack as A  # noqa: E402
from citeleak.data import load_world  # noqa: E402
from citeleak.disclosure import EPS_GRID, view  # noqa: E402

LIVE = "--live" in sys.argv
st.set_page_config(page_title="CiteLeak", page_icon="🔎", layout="wide")


@st.cache_resource
def load_all():
    world = load_world(ROOT / "data" / "world.json")
    M = json.loads((ROOT / "results" / "metrics.json").read_text())
    with open(ROOT / "results" / "attack_models.pkl", "rb") as fh:
        AM = pickle.load(fh)
    recs = {}
    for line in (ROOT / "results" / "cache.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("model") == M["model"]:
                recs[r["query"]] = r
    return world, M, AM, recs


@st.cache_resource
def load_assistant(model, docs):
    from citeleak.rag import HFBackend, MediAssist
    return MediAssist(docs, HFBackend(model), k=4, cache_path=ROOT / "results" / "cache.jsonl")


world, M, AM, RECS = load_all()
ASSIST = load_assistant(M["model"], world["docs"]) if LIVE else None


def ask(q: str) -> dict | None:
    if q in RECS:
        return RECS[q]
    if ASSIST is None:
        return None
    r = ASSIST.ask(q)
    RECS[q] = r
    return r


POLICY_LABEL = {"L0": "Answer only (no sources)", "L1": "Citations (ids, no scores)",
                "L2": "Status quo: sources + scores"}
for e in EPS_GRID:
    POLICY_LABEL[f"shield({e})"] = f"🛡️ Privacy Shield ε={e:g}"


def render_view(v: dict):
    st.markdown(f"**MediAssist:** {v['answer'] or '_(empty answer)_'}")
    if not v["sources"]:
        st.caption("No sources panel shown.")
        return
    st.markdown("**Sources**")
    for s in v["sources"]:
        kind = s["kind"].replace("_", " ") if s["kind"] else "type hidden"
        if s["share"] is None:
            st.markdown(f"- `{s['label']}` · {kind}")
        else:
            c1, c2 = st.columns([2, 3])
            c1.markdown(f"`{s['label']}` · {kind}")
            c2.progress(min(max(s["share"], 0.0), 1.0), text=f"{s['share']:.2f}")


st.title("🔎 CiteLeak — when the “Sources” button leaks secrets")
st.caption("A fictional hospital assistant over **synthetic** patient records. "
           + ("Live mode: questions are answered by the local LLM." if LIVE else
              "Replay mode: shows cached answers from the experiment (run with `-- --live` for new questions)."))

tab1, tab2, tab3 = st.tabs(["💬 Use MediAssist", "🕵️ Be the attacker", "📊 Results"])

GUARD = st.sidebar.toggle("Output privacy filter (blocks answers about named people)", value=True)
st.sidebar.caption("Turn it off to see what the model says with only its system-prompt guardrail.")

with tab1:
    pol = st.selectbox("What does the interface show?", ["L0", "L1", "L2", "shield(1.0)", "shield(8.0)"],
                       index=2, format_func=POLICY_LABEL.get, key="pol1")
    if LIVE:
        q = st.text_input("Ask a question", "What is the standard follow-up for type 2 diabetes?")
    else:
        q = st.selectbox("Pick a question that was asked during the experiment", sorted(RECS)[:400])
    if st.button("Ask", type="primary"):
        r = ask(q)
        if r:
            render_view(view(r, pol, guard=GUARD))
        else:
            st.warning("Not in cache. Start the app with `-- --live` to ask new questions.")

with tab2:
    st.markdown("You are an outsider who knows a person's **name, age, sex and city**. "
                "MediAssist will never tell you who is a patient. Can the sources panel?")
    targets = [t for t in world["targets"]
               if all(q in RECS for q in A.membership_queries(t)) or LIVE]
    c1, c2 = st.columns(2)
    names = [f"{t['name']} ({t['age']}, {t['sex']}, {t['city']})" for t in targets]
    i = c1.selectbox("Target person", range(len(targets)), format_func=lambda j: names[j])
    pol2 = c2.select_slider("Defense", options=["L0", "L1", "L2"] + [f"shield({e})" for e in EPS_GRID],
                            value="L2", format_func=POLICY_LABEL.get)
    t = targets[i]
    if st.button("🚀 Run the attack (4 questions)", type="primary"):
        qs = A.membership_queries(t)
        recs = [ask(q) for q in qs]
        views = [view(r, pol2, guard=GUARD) for r in recs]
        x = A.membership_features(t, views, AM["med_vocab"]).reshape(1, -1)
        p = float(AM["models"][pol2].predict_proba(x)[0, 1])
        if not GUARD:
            st.info("The attacker model was trained on filtered outputs; with the filter off, "
                    "just read the answers. The text often gives the person away directly.")
        auc = M["membership"][pol2]["auc"]
        st.metric("Attacker's confidence that this person is a patient", f"{p:.0%}",
                  help=f"Logistic-regression attacker trained on this disclosure policy (AUC {auc:.2f}).")
        st.progress(p)
        for q, v in zip(qs, views):
            with st.expander(f"❓ {q}"):
                render_view(v)
        with st.expander("🔓 Reveal the ground truth"):
            truth = "IS" if t["member"] else "is NOT"
            st.markdown(f"**{t['name']} {truth} in the datastore.**"
                        + (f" Their record says: _{t['condition']}_." if t["member"] else ""))
            verdict = (p >= 0.5) == bool(t["member"])
            st.markdown("✅ The attacker was right." if verdict else "❌ The attacker was wrong.")

with tab3:
    mi = M["membership"]
    a, b, c = st.columns(3)
    a.metric("Attack AUC, answer only", f"{mi['L0']['auc']:.2f}")
    b.metric("Attack AUC, sources + scores", f"{mi['L2']['auc']:.2f}",
             delta=f"{mi['L2']['auc'] - mi['L0']['auc']:+.2f}", delta_color="inverse")
    c.metric("Attack AUC, Privacy Shield ε=1", f"{mi['shield(1.0)']['auc']:.2f}",
             delta=f"{mi['shield(1.0)']['auc'] - mi['L2']['auc']:+.2f}", delta_color="inverse")
    fig = ROOT / "results" / "figures"
    l, r = st.columns(2)
    l.image(str(fig / "fig_leak.png"))
    r.image(str(fig / "fig_frontier.png"))
    l.image(str(fig / "fig_roc.png"))
    r.image(str(fig / "fig_attr.png"))
    with st.expander("All membership-attack numbers"):
        rows = [{"policy": p, "AUC": round(d["auc"], 3),
                 "95% CI": f"{d['auc_ci'][0]:.2f}–{d['auc_ci'][1]:.2f}",
                 "TPR@1%FPR": round(d["tpr@1%fpr"], 3), "TPR@5%FPR": round(d["tpr@5%fpr"], 3),
                 "faithfulness (top-1)": round(M["faithfulness"].get(p, {}).get("top1_agreement", float("nan")), 3)}
                for p, d in mi.items()]
        st.dataframe(rows, width="stretch")
