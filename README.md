# 🔎 CiteLeak: when the “Sources” button leaks secrets

**Privacy side channels in explainable retrieval-augmented generation (RAG), and a defense.**

AI assistants increasingly show *sources*, often with attribution or confidence scores, so users
can check where an answer came from. That's good explainability practice. But when the assistant
searches a **private** document collection, the sources panel is also a **side channel**: it reveals
what the system found and how heavily the model relied on it.

CiteLeak asks:

> **Can the sources panel of a RAG assistant leak private information even when the text answer is
> guarded? And can we make the panel private without destroying its value as an explanation?**

We build **MediAssist**, a fictional hospital assistant over a *synthetic* patient database, attack it
using only what a normal user sees, and evaluate a defense called **Privacy Shield**.

> ⚠️ **All data is synthetic.** Names are random combinations of common first and last names; clinical
> details are templated. No real patient data is used anywhere.

**🔗 Live demo:** [https://citeleak.streamlit.app](https://citeleak.streamlit.app)  ·  **📄 Poster:** [`poster/CiteLeak_poster.pdf`](poster/CiteLeak_poster.pdf)  ·  **📊 All numbers:** [`results/RESULTS.md`](results/RESULTS.md)

## Results at a glance

Qwen2.5-1.5B-Instruct (local, Apple Silicon), 1,666 synthetic documents, 300 target people (150 truly in the
datastore), BM25 top-4, adaptive logistic-regression attacker with out-of-fold scores.

| What the user sees | Membership attack AUC (95% CI) | Patients caught at 5% false alarms | Sources-panel faithfulness |
|---|---|---|---|
| Answer only (output filter on) | **0.53** (0.46–0.60) | 0% | – |
| + citations (ids, no scores) | 0.95 (0.93–0.97) | 58% | – |
| + attribution scores (**status quo**) | **0.98** (0.97–1.00) | **95%** | 96% |
| Unlinkable ids only | 0.95 | 65% | 96% |
| DP noise only, ε=2 (stable ids) | 0.94 | 67% | – |
| **Privacy Shield ε=1** (unlinkable ids + DP noise) | **0.54** (0.47–0.60) | 4% | **32%** |

* **H1 ✓**: the sources panel adds large leakage beyond the guarded answer.
* **H2 ✓**: leakage grows with granularity: nothing < citations < scores.
* **H3 partly**: Privacy Shield returns leakage to the answer-only floor, but faithfulness of the panel falls
  from 96% to 32%. **Ids and scores each leak on their own**: fixing only one barely helps.
* **Prompt-only guardrails failed**: with only the system prompt, the model repeated the person's full name in
  97% of answers and a patient's diagnosis in 74% (answer-only AUC 0.72), so all main results use an output filter.
* **Attribute inference did not work**: guessing the diagnosis among 5 candidates stayed at chance (15% vs. 20%).
  *Whether* someone is a patient leaked; *what* they have did not, with this attack.

---

## Quick start (macOS / Apple Silicon, also works on Linux with CUDA or CPU)

```bash
git clone https://github.com/MAUZAMSHAFIBHAT/CiteLeak.git.git && cd citeleak
./run.sh                                   # venv + install + tests + full experiment + figures + poster
streamlit run app.py                       # interactive demo (replay mode, instant)
streamlit run app.py -- --live             # demo with the live LLM (ask new questions)
```

> **Two requirement files.** `requirements-full.txt` (used by `run.sh`) has everything for the experiment.
> `requirements.txt` is the light set for the hosted demo, which runs in replay mode from the saved results.

| Preset | Command | Model | Approx. time on an M-series Mac |
|---|---|---|---|
| Full (paper numbers) | `./run.sh` | Qwen2.5-1.5B-Instruct | ~45 min (measured: 1.7 s/query × 1,600 queries) |
| Fast | `./run.sh --model Qwen/Qwen2.5-0.5B-Instruct` | Qwen2.5-0.5B-Instruct | ~15–30 min |
| Tiny smoke test | `python tests/make_tiny_model.py && python -m citeleak.run all --model tests/tiny-model --n-targets 40 --n-background 300 --n-attr 6` | random weights | ~1 min |

The LLM phase is **resumable**: every answer and ablation is cached in `results/cache.jsonl`.
If you stop it, rerun the same command and it continues. Analysis (`python -m citeleak.run analyze`)
needs no LLM and runs in seconds.

**Requirements:** Python 3.10+, ~4 GB of disk for the model (downloaded automatically from Hugging Face
on first run), and ~8 GB of RAM. If you see a `Non-finite log-probs` error, add `--dtype float32`.

**Outputs**

| File | What it is |
|---|---|
| `results/RESULTS.md` | All numbers in tables |
| `results/metrics.json` | Machine-readable metrics |
| `results/figures/*.png` | Figures (leakage, ROC, privacy–faithfulness frontier, attribute inference, histogram) |
| `poster/CiteLeak_poster.pdf` | 36″ × 24″ poster, auto-built from the metrics (edit `poster/config.json` first) |
| `results/attack_models.pkl` | Trained attackers used by the demo |

---

## How it works

### 1. The system: MediAssist
```
question → BM25 retriever (top-k=4 of ~1,650 docs) → guarded LLM → answer
                                                    ↘ attribution → "Sources" panel
```
* **Datastore** (`citeleak/data.py`): 1,500 background patients + target patients + 16 care protocols.
* **Guardrails**: the system prompt forbids revealing patients or confirming whether anyone is a patient.
  Small LLMs often ignore this (we measure how often), so MediAssist also runs an **output privacy filter**:
  if the question or draft answer mentions a person (a first-name gazetteer acts as the name detector), the
  user gets a fixed refusal. The sources panel is computed for the model's draft answer, as in pipelines
  where citations come out of generation and the guardrail rewrites only the text. Main results use the
  filter; `membership_unguarded` in `metrics.json` reports the prompt-only setting.
* **Attribution** (`citeleak/rag.py`): ContextCite-style (Cohen-Wang et al., NeurIPS 2024). The *same*
  answer is re-scored under **all 2^k = 16 subsets** of the retrieved sources (one batched forward pass),
  and a linear surrogate `log p(answer | subset) ≈ w0 + Σ wᵢ·[source i kept]` gives each source's weight.
  With k=4 we can afford the full ablation design instead of random sampling plus LASSO. Positive weights
  are normalized into **shares** shown to the user.

### 2. What the user sees (disclosure policies, `citeleak/disclosure.py`)
| Policy | Sources panel |
|---|---|
| `L0` | none (answer only) |
| `L1` | cited sources (share ≥ 0.15), stable ids, no scores |
| `L2` | **status quo**: all retrieved sources with ids, type and attribution share |
| `coarse` | L2 with High / Medium / Low instead of numbers |
| `unlink` | fresh random labels per answer, source type hidden, exact shares |
| `dp(ε)` | stable ids, shares + Laplace(2/ε) noise |
| `shield(ε)` | **Privacy Shield** = `unlink` + `dp(ε)` |

### 3. The attacker (`citeleak/attack.py`)
* **Knowledge:** the target's name, age, sex and city (a neighbour, employer, or data broker).
* **Access:** black-box; sees only the policy's view.
* **Queries:** three natural questions that mention the person plus one *control* question without the name.
* **Features:** text cues (refusals, name/medication mentions, answer change vs. control) and
  sources-panel cues (does the same record id recur across questions, top patient-record share,
  share entropy, change vs. control).
* **Classifier:** logistic regression trained like a shadow-model attack. All scores are **out-of-fold**
  (5-fold CV), and the attacker is **retrained for every defense** (an adaptive attacker).
* **Attribute inference:** for member targets, ask about each of 5 candidate conditions and pick
  the one whose answer leans hardest on a patient record (chance = 20%).

### 4. Metrics
* **Leakage:** ROC-AUC with 95% bootstrap CI; TPR at 1% and 5% FPR (the confident-hit regime that matters for privacy).
* **Faithfulness of the sources panel**, measured against the model's *true* leave-one-out effects
  computed from the cached ablations: top-1 agreement, Spearman correlation, and removal ratio.

### 5. The differential-privacy claim, stated precisely
The share vector lies on the probability simplex, so between *any* two datastores the released
vectors differ by at most 2 in L1 norm. Adding Laplace(2/ε) noise per coordinate therefore makes
**one release of the share vector ε-DP** with respect to adding or removing a record. Clipping and
renormalising are post-processing. Scope:
* ✅ covers the attribution-score channel for one query; q queries compose to q·ε.
* ❌ does **not** cover the answer text (that is the `L0` "answer-only floor"), nor stable source ids.
  That is why Privacy Shield also makes labels unlinkable and hides source types. The unlinkability
  part is an empirical defense, not a formal guarantee.

---

## Repository layout
```
citeleak/
  data.py         synthetic hospital world (seeded, reproducible)
  rag.py          retriever, HF LLM backend (MPS/CUDA/CPU), exact ablation attribution, cache
  disclosure.py   what the user sees: L0/L1/L2 and the defenses
  attack.py       membership and attribute inference attacks plus metrics
  run.py          experiment runner (query → analyze → figures → poster)
  figures.py      matplotlib figures (color-blind-checked palette)
  poster.py       auto-generated 36"×24" poster PDF; every claim is computed from the metrics
app.py            Streamlit demo (replay or live)
tests/            unit tests + tiny random-weight model for smoke tests
tools/            pin_demo_requirements.py (exact versions for the hosted demo)
results/          cached LLM outputs, metrics, figures and trained attackers from the reported run
data/world.json   the synthetic hospital (seeded, regenerated automatically if missing)
poster/           CiteLeak_poster.pdf, config.json (name, course, repo link for the QR code), Lato fonts (OFL)
```

## Design decisions
* **BM25, not dense retrieval:** deterministic, no extra model download, and keyword retrievers are
  widely used in production hybrid search. Dense retrieval is future work.
* **Exact 2^k ablations:** removes surrogate-fitting noise from the attribution, so leakage is not an
  artifact of a noisy explainer.
* **Synthetic data:** lets us know ground-truth membership and publish everything without privacy risk.
* **Adaptive attacker:** every defense is evaluated against an attacker retrained on that defense's
  outputs. Evaluating defenses against a fixed attacker overstates protection.

## Limitations
* Templated synthetic notes are more regular than real clinical text, which probably makes matching easier.
* One small open LLM and one retriever; results may differ for larger models and dense or hybrid retrieval.
* The attacker knows exact quasi-identifiers; noisy side knowledge would weaken it.
* Faithfulness is measured against leave-one-out effects of the same model; it does not capture
  whether users can still verify answers. A user study is future work.
* The DP guarantee is per query and covers only the score channel (see above).
* The output filter is a simple name gazetteer; real PII detectors differ. Attribution is computed for the
  draft answer, not the refusal text; systems that attribute the final displayed text may behave differently.

## Ethics and responsible disclosure
The attack targets only our own system and only synthetic data. We publish the defense with the
attack. The underlying risk (explanations of private retrieval act as side channels) applies to
real deployments that show source attributions over sensitive collections; deployers should
threat-model their transparency features.

## Related work and what is new here
Prior work shows that (a) RAG datastores leak membership through *answers*, and (b) feature-attribution
explanations of *trained models* leak training-set membership. As far as we found, CiteLeak is the first to
measure the **extra** leakage from the **context-attribution / sources panel** of a RAG system over a
private store, to separate the effect of **linkable source ids** from the effect of the **scores**, and to map the
privacy–faithfulness trade-off against an adaptive attacker.

* Cohen-Wang, Shah, Georgiev, Mądry. *ContextCite: Attributing Model Generation to Context.* NeurIPS 2024.
* Anderson et al. *Is My Data in Your Retrieval Database? Membership Inference Attacks Against RAG.* 2024 (arXiv:2405.20446).
* Naseh, Peng, Suri, Chaudhari, Oprea, Houmansadr. *Riddle Me This! Stealthy Membership Inference for Retrieval-Augmented Generation.* ACM CCS 2025.
* Li, Liu, Wang, Yang. *Generating Is Believing: Membership Inference Attacks against RAG (S²MIA).* 2024 (arXiv:2406.19234).
* Li et al. *BudgetLeak: Membership Inference Attacks on RAG via the Generation Budget Side Channel.* 2025 (arXiv:2511.12043).
* Liu, Wu, Yu, Zhang. *Please Tell Me More: Privacy Impact of Explainability through the Lens of Membership Inference Attack.* IEEE S&P 2024.
* *DeepLeak: Privacy Enhancing Hardening of Model Explanations Against Membership Leakage.* 2026 (arXiv:2601.03429). Hardens image-model explanations with noise, clipping and masking; CiteLeak instead studies context attribution over a private *retrieval* store and the role of id linkability.
* Shokri, Strobel, Zick. *On the Privacy Risks of Model Explanations.* AIES 2021.
* Dwork & Roth. *The Algorithmic Foundations of Differential Privacy.* 2014.

## License
MIT. See `LICENSE`.
