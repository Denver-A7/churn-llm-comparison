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
