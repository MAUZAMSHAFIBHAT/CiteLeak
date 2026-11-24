"""Fast unit tests (no LLM needed):  python -m pytest -q"""
import itertools

import numpy as np

from citeleak.data import build_world
from citeleak.disclosure import laplace_shares, view
from citeleak.rag import Retriever, attribution_from_ablations


def test_attribution_recovers_linear_effects():
    true_w, b = np.array([3.0, -1.0, 0.5, 2.0]), -10.0
    f = {m: b + sum(true_w[i] for i in m)
         for r in range(5) for m in itertools.combinations(range(4), r)}
    out = attribution_from_ablations(4, f)
    assert np.allclose(out["weights"], true_w)
    assert np.allclose(out["loo"], true_w)
    assert abs(sum(out["shares"]) - 1) < 1e-9 and out["shares"][1] == 0


def test_laplace_shares_valid_distribution():
    rng = np.random.default_rng(0)
    for eps in (0.5, 4.0):
        s = laplace_shares([0.7, 0.2, 0.1, 0.0], eps, rng)
        assert abs(sum(s) - 1) < 1e-9 and min(s) >= 0


def test_world_membership_split_and_retrieval():
    w = build_world(n_background=200, n_targets=20, seed=1)
    ids = {d["doc_id"] for d in w["docs"]}
    for t in w["targets"]:
        assert (t["doc_id"] in ids) == bool(t["member"])
    r = Retriever(w["docs"])
    t = next(t for t in w["targets"] if t["member"])
    hits = [d["doc_id"] for d, _ in r.search(f"{t['name']} {t['city']}", 4)]
    assert t["doc_id"] in hits


def test_shield_hides_linkable_information():
    rec = {"key": "k", "query": "q", "answer": "x", "shares": [0.6, 0.2, 0.1, 0.1],
           "sources": [{"doc_id": f"R-{i}", "kind": "patient_record", "title": ""} for i in range(4)]}
    v = view(rec, "shield(1.0)")
    assert all(s["kind"] is None and not s["label"].startswith("R-") for s in v["sources"])
    assert view(rec, "L0")["sources"] == []
    assert all(s["share"] is None for s in view(rec, "L1")["sources"])
