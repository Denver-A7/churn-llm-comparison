# Churn LLM Comparison

A small, reproducible study of customer churn prediction using the Telco Customer Churn dataset. It compares a trained logistic-regression baseline with Claude Haiku used zero-shot on natural-language summaries of customer records.

## Problem statement

The project asks whether a zero-shot large language model can substitute for a conventional trained model on a structured prediction task: predict whether a customer will churn (`Yes` or `No`) from the same held-out customer attributes used by the baseline.

The comparison is not just about which model scores higher. It also asks whether the LLM's short, stated rationale is consistent with the profile and whether it holds up when predictions are wrong. Agreement, disagreements, segment-level results, and examples of correct and incorrect reasoning should be examined alongside aggregate metrics. Generated rationales are explanations to inspect, not proof of the model's internal reasoning or causal evidence.

## Dataset

The input is the Telco Customer Churn CSV stored locally at `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`. The inspected file has 7,043 rows and 21 columns. It includes customer attributes for contract, tenure, services, payment method, and charges; `Churn` is the binary target and `customerID` is an identifier, not a predictive feature.

`TotalCharges` is converted to numeric with invalid/blank values coerced to missing values. The baseline preprocessing pipeline median-imputes numeric features using training data. Raw data is excluded from Git; obtain and place the dataset locally before running the notebooks.

## Method

### Trained baseline

- A logistic-regression classifier with `class_weight="balanced"`.
- An 80/20 train/test split stratified by `Churn`, with `random_state=42`.
- Numeric features are median-imputed and standardized; categorical features, including the low-cardinality `SeniorCitizen` field, are imputed and one-hot encoded.
- Evaluation uses accuracy, precision, recall, F1, and a confusion matrix. Baseline probabilities are saved with held-out predictions.

### Zero-shot LLM

- Claude Haiku receives a compact, natural-language customer summary and is asked for a `Yes`/`No` prediction with one short reason. It is not fine-tuned on this dataset.
- Customer IDs, target labels, and baseline outputs are kept out of the LLM prompt and joined locally for evaluation.
- The workflow is staged: validate summaries and response parsing, then run a small live test, review it, and only then opt into the 200-row test-set sample (`random_state=42`). API calls incur usage costs. A separate 20–30-row Sonnet side-sample is available as an optional experiment.
- Compare models on the same valid rows using standard classification metrics, prediction agreement, disagreement examples, and breakdowns by contract type, tenure bucket (`0–12`, `13–24`, `25–48`, and `49+` months), and monthly-charge quartile.
- An optional cost-sensitive score assigns a false negative (missed churner) five times the cost of a false positive. Review error counts as well as the normalized score.

## Reproduce

1. Create/select a Python environment and install the dependencies from `requirements.txt`.
2. Place the source CSV in `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`.
3. Run `notebooks/01_load_and_inspect.ipynb` from top to bottom to inspect and clean the data, explore churn patterns, and train/evaluate the baseline. This writes `data/processed/baseline_predictions.csv`.
4. For the LLM experiment, make sure the local `.env` contains `ANTHROPIC_API_KEY`. `.env` is ignored by Git; never commit it or put the key in a notebook. `ANTHROPIC_MODEL` can select the configured Haiku model.
5. Run `notebooks/02_haiku_churn_pilot.ipynb` cell by cell so you can review API-call cells before executing them. It contains an 8-row JSON-format pilot and a separate three-row natural-language-summary test; both make billable requests. Inspect the outputs, then explicitly enable the 200-row batch only after a pilot succeeds and billing is available. The comparison section reads `data/processed/llm_predictions.csv` when it exists.
6. After predictions are available, rerun the comparison section to create the merged table and segment reports. See `data/processed/` for generated artifacts. Do not treat failed API rows as valid classifications.

The notebook uses sequential API calls and records response errors. API credits and model availability are controlled by Anthropic and may change; a valid key alone does not guarantee available usage balance.

## Findings (fill in after running the experiments)

- **Baseline test metrics:** accuracy `[TBD]`, precision `[TBD]`, recall `[TBD]`, F1 `[TBD]`.
- **LLM sample and completion status:** sample size `[TBD]`, valid predictions `[TBD]`, failures `[TBD]`.
- **LLM test metrics:** accuracy `[TBD]`, precision `[TBD]`, recall `[TBD]`, F1 `[TBD]`.
- **Baseline/LLM agreement:** `[TBD]`.
- **Disagreement rate and segment patterns:** `[TBD]`.
- **Which model was correct more often on disagreements:** `[TBD]`.
- **Cost-weighted comparison (false negative cost = 5, false positive cost = 1):** `[TBD]`.
- **Rationale review:** examples where the stated reason was supported by the customer profile `[TBD]`; examples where it was weak, contradicted, or unhelpful `[TBD]`.
- **Conclusion on substitution:** `[TBD]`.

## Project layout

```text
churn-llm-comparison/
├── data/
│   ├── processed/       # Baseline and experiment outputs
│   └── raw/             # Local source data (gitignored)
├── documentation/       # Project log
├── notebooks/           # Inspection, baseline, and LLM experiment notebooks
├── outputs/             # Additional figures/reports
├── src/                 # Reusable project code
├── .env.example         # Local configuration template
└── requirements.txt
```
