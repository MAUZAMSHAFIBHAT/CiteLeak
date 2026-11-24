# CiteLeak results

Model: `Qwen/Qwen2.5-1.5B-Instruct` · targets: 300 (150 members) · datastore: 1666 docs · k=4

## Membership inference (out-of-fold, adaptive attacker per policy)

| Policy | AUC | 95% CI | TPR@1%FPR | TPR@5%FPR | Faithfulness top-1 | Spearman vs LOO |
|---|---|---|---|---|---|---|
| L0 | 0.535 | 0.46–0.60 | 0.00 | 0.00 | nan | nan |
| L1 | 0.951 | 0.93–0.97 | 0.52 | 0.58 | 0.96 | 0.76 |
| L2 | 0.984 | 0.97–1.00 | 0.55 | 0.95 | 0.96 | 0.70 |
| coarse | 0.981 | 0.96–1.00 | 0.56 | 0.99 | 0.94 | 0.75 |
| unlink | 0.946 | 0.92–0.97 | 0.44 | 0.65 | 0.96 | 0.70 |
| dp(0.5) | 0.938 | 0.91–0.96 | 0.33 | 0.69 | 0.28 | 0.07 |
| dp(1.0) | 0.943 | 0.91–0.97 | 0.37 | 0.67 | 0.34 | 0.12 |
| dp(2.0) | 0.940 | 0.91–0.96 | 0.20 | 0.67 | 0.42 | 0.22 |
| dp(4.0) | 0.972 | 0.95–0.99 | 0.61 | 0.87 | 0.57 | 0.36 |
| dp(8.0) | 0.981 | 0.96–1.00 | 0.51 | 0.95 | 0.77 | 0.52 |
| dp(16.0) | 0.985 | 0.97–1.00 | 0.81 | 0.95 | 0.90 | 0.63 |
| dp(32.0) | 0.983 | 0.97–1.00 | 0.56 | 0.96 | 0.94 | 0.66 |
| shield(0.5) | 0.519 | 0.45–0.59 | 0.05 | 0.13 | 0.29 | 0.08 |
| shield(1.0) | 0.538 | 0.47–0.60 | 0.01 | 0.04 | 0.32 | 0.12 |
| shield(2.0) | 0.658 | 0.60–0.72 | 0.02 | 0.15 | 0.42 | 0.23 |
| shield(4.0) | 0.723 | 0.67–0.78 | 0.02 | 0.13 | 0.58 | 0.36 |
| shield(8.0) | 0.851 | 0.81–0.90 | 0.04 | 0.28 | 0.76 | 0.51 |
| shield(16.0) | 0.921 | 0.89–0.95 | 0.17 | 0.52 | 0.90 | 0.62 |
| shield(32.0) | 0.917 | 0.88–0.95 | 0.08 | 0.47 | 0.93 | 0.65 |

Training-free threshold attack on the top patient-record share (L2): AUC 0.801

## Attribute inference (5 candidate conditions, chance 0.20)

| Policy | Accuracy | 95% CI | n |
|---|---|---|---|
| L0 | 0.15 | 0.07–0.24 | 80 |
| L1 | 0.15 | 0.07–0.24 | 80 |
| L2 | 0.15 | 0.07–0.23 | 80 |
| shield(1.0) | 0.11 | 0.05–0.19 | 80 |
| shield(4.0) | 0.09 | 0.03–0.16 | 80 |

## Guardrail (answer text)

- prompt_only_refusal_rate: 0.001
- filter_block_rate: 1.000
- full_name_in_answer: 0.972
- member_condition_in_answer: 0.736
- target_record_retrieved: 1.000
