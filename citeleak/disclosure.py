"""What the user (and therefore the attacker) gets to see.

A cached MediAssist record holds everything (answer, retrieved doc ids, exact
ablation results, attribution). A *disclosure policy* turns that record into the
"view" an outside user sees. Defenses are disclosure policies too.

Policies
  L0           answer only (no sources panel)
  L1           answer + cited sources (stable ids/titles), no scores
  L2           answer + all retrieved sources with attribution shares (status quo)
  coarse       L2, but shares rounded to High / Medium / Low
  unlink       per-response random labels ("Source A".."D"), source type hidden, exact shares
  dp(eps)      stable ids, shares + Laplace noise (eps-DP release of the share vector)
  shield(eps)  PRIVACY SHIELD = unlink + dp(eps)

Differential-privacy note (stated precisely, see README):
  The share vector s lies on the probability simplex, so for ANY two datastores
  the L1 distance between released vectors is at most 2. Adding Laplace(2/eps)
  noise to each of the k coordinates therefore makes the release of the share
  vector eps-DP with respect to adding/removing one record; clipping and
  renormalising afterwards is post-processing. The guarantee covers the
  attribution-score channel for ONE query; q queries compose to q*eps.
  It does NOT cover the answer text or stable source ids, which is why
  shield() also removes linkable ids and source types.
"""
from __future__ import annotations

import hashlib
import re
import string

import numpy as np

from .data import FIRST

# ---------------------------------------------------------------- output filter
# The system prompt alone does NOT stop small LLMs from repeating patient details (we measure
# this). Like production PII/guardrail layers, MediAssist therefore runs an output filter: if
# the question or the draft answer mentions a person (gazetteer-based name detector), the user
# gets a fixed refusal. The sources panel is computed for the model's draft answer, as in
# pipelines where citations are produced with generation and the guardrail rewrites only text.
GUARD_REFUSAL = ("I'm sorry, I can't share information about specific individuals. For general "
                 "guidance, please consult the hospital's care protocols.")
_NAME_RE = re.compile(r"\b(?:" + "|".join(sorted(set(FIRST), key=len, reverse=True)) + r")\s+[A-Z][a-z]+")


def guarded_answer(rec: dict) -> str:
    if _NAME_RE.search(rec["query"]) or _NAME_RE.search(rec["answer"]):
        return GUARD_REFUSAL
    return rec["answer"]

CITE_THRESHOLD = 0.15


def _rng(key: str, policy: str) -> np.random.Generator:
    seed = int(hashlib.sha1(f"{key}|{policy}".encode()).hexdigest()[:12], 16)
    return np.random.default_rng(seed)


def laplace_shares(shares: list[float], eps: float, rng: np.random.Generator) -> list[float]:
    s = np.asarray(shares, float) + rng.laplace(0.0, 2.0 / eps, size=len(shares))
    s = np.clip(s, 0, None)
    return (s / s.sum()).tolist() if s.sum() > 1e-12 else [1.0 / len(s)] * len(s)


def bucket(x: float) -> float:
    return 0.8 if x >= 0.5 else 0.35 if x >= 0.2 else 0.05   # High / Medium / Low


def view(rec: dict, policy: str, guard: bool = True) -> dict:
    """Return {'answer', 'sources': [{'label','kind','share'}], 'policy'}.

    guard=True applies the output privacy filter to the answer text (default, main results);
    guard=False shows the raw model answer (prompt-only guardrail)."""
    srcs, shares = rec["sources"], rec["shares"]
    out = {"answer": guarded_answer(rec) if guard else rec["answer"], "policy": policy, "sources": []}
    if policy == "L0":
        return out
    if policy == "L1":
        order = sorted(range(len(srcs)), key=lambda i: -shares[i])
        out["sources"] = [{"label": srcs[i]["doc_id"], "kind": srcs[i]["kind"], "share": None,
                           "slot": i} for i in order if shares[i] >= CITE_THRESHOLD]
        return out

    stable = True
    show_kind = True
    s = list(shares)
    if policy == "L2":
        pass
    elif policy == "coarse":
        s = [bucket(x) for x in s]
    elif policy == "unlink":
        stable, show_kind = False, False
    elif policy.startswith("dp(") or policy.startswith("shield("):
        eps = float(policy[policy.index("(") + 1:-1])
        s = laplace_shares(s, eps, _rng(rec["key"], policy))
        if policy.startswith("shield("):
            stable, show_kind = False, False
    else:
        raise ValueError(policy)

    rng = _rng(rec["key"], policy + "|labels")
    letters = list(string.ascii_uppercase[:len(srcs)])
    perm = rng.permutation(len(letters))
    for i, src in enumerate(srcs):
        out["sources"].append({
            "label": src["doc_id"] if stable else f"Source {letters[perm[i]]}-{rng.integers(1e6):06d}",
            "kind": src["kind"] if show_kind else None,
            "share": float(s[i]),
            "slot": i,
        })
    return out


def released_scores(v: dict, k: int) -> np.ndarray:
    """Per-slot scores implied by a view (used to measure faithfulness)."""
    sc = np.zeros(k)
    if not v["sources"]:
        return sc
    if v["sources"][0]["share"] is None:            # L1: ordering only
        n = len(v["sources"])
        for rank, s in enumerate(v["sources"]):
            sc[s["slot"]] = n - rank
        return sc
    for s in v["sources"]:
        sc[s["slot"]] = s["share"]
    return sc


POLICIES_MAIN = ["L0", "L1", "L2"]
EPS_GRID = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
POLICIES_DEFENSE = (["L2", "L1", "coarse", "unlink"] + [f"dp({e})" for e in EPS_GRID]
                    + [f"shield({e})" for e in EPS_GRID])
