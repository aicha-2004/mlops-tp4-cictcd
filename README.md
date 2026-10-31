# MLOps TP4 : CI/CD, tests, quality gates et gouvernance

Classification de sentiment (TF-IDF + regression logistique) avec tests, quality gates,
MLflow Registry (candidate / champion), pipeline GitHub Actions et model card.

## Utilisation locale

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    set PYTHONPATH=.
    pytest tests/ -v --cov=src --cov-report=term-missing
Pipeline CI/CD avec approbation manuelle avant la production.