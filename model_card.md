# Sentiment Model

## Use case
Classification binaire de sentiment.

## Version
Version du registre : A COMPLETER (ex. 1), alias candidate / champion.
Run MLflow : A COMPLETER (identifiant du run).

## Dataset
reviews.csv, jeu pedagogique (50 avis courts en francais, 25 positifs et 25 negatifs).

## Method
TF-IDF + Logistic Regression.

## Parameters
C = 1.0
max_features = 5000
ngram_range = [1, 2]

## Metrics
Accuracy = A COMPLETER
F1 = A COMPLETER

## Quality Gates
Unit tests = A COMPLETER (passed / failed)
Data tests = A COMPLETER
Model tests = A COMPLETER
Security checks = A COMPLETER (bandit, pip-audit)

## Known limitations
- Petit jeu de donnees pedagogique (jeu de test de seulement quelques exemples).
- Langue principalement francaise.
- Evaluation non representative d'un contexte industriel.

## Governance
validation_status = passed
approval_status = A COMPLETER (approved / non approuve)
alias = candidate / champion

## Deployment decision
A COMPLETER (decision, date, qui a approuve, condition ayant autorise le passage a champion).
