# Churn Prediction API

An end-to-end Telco customer churn prediction system — not just a notebook, but
a structured ML package moving toward a deployable API. Built as a portfolio
project to demonstrate production ML engineering practices: leakage-safe
pipelines, reproducible splits, and a design that survives the jump from
"notebook that works" to "code someone else could run."

**Dataset:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle, BlastChar) — 7,043 customers, ~26.5% churn rate.

## Status

- [x] **Phase 1 — Modeling:** EDA, preprocessing pipeline, model training & selection
- [ ] **Phase 2 — Serving:** FastAPI app, Docker, tests, deployment

## Setup