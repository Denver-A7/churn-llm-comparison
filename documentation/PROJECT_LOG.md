# Project Log

## 2026-09-30 — Project setup

This project is being set up to explore and compare approaches to customer churn prediction, including large language model (LLM)-based approaches. The initial scaffold provides dependencies for data analysis, machine learning, visualization, notebooks, and Anthropic API integration. The inspection notebook is intended to load and summarize the Telco Customer Churn dataset before analysis.

Initial folder structure:

```text
churn-llm-comparison/
├── data/
│   ├── processed/
│   └── raw/
├── documentation/
├── notebooks/
├── outputs/
└── src/
```

Project configuration files at the root include `requirements.txt` and `.gitignore`. The raw data directory is excluded from version control; add the source dataset there locally to run the inspection notebook.

## 2026-10-17: Baseline and Haiku comparison

Trained a class-balanced logistic-regression baseline on 5,634 customers and evaluated it on 1,409 held out customers (accuracy 0.737, F1 0.613). Ran Claude Haiku with no training on a 200 row sample of the test set; 194 answers were usable. On those 194 customers the two models agreed on 160. Haiku was right on 20 of the 34 disagreements and the baseline on 14, which is not statistically meaningful difference (p about 0.39). The early 8-row pilot ran before API credits were added, so it recorded billing errors; the results come from the later 3 row test and 200 row batch.

## 2026-10-18: Review and cleanup

Added findings and related work to the README, removed local file paths from saved notebook outputs, reset the paid-run switches to off, and added a note explaining the early failed pilot.