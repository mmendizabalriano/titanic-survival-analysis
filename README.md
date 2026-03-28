# Titanic — Survival Analysis & Predictive Modeling

Exploratory data analysis and classification model built on the
Titanic dataset, as part of my data analysis portfolio.

## Goal

Analyze how passenger characteristics — sex, social class, age,
family size, and fare — influenced survival probability during
the Titanic disaster.

## Key Findings

- **Sex was the strongest predictor**: 74% of women survived vs
  19% of men, confirming the "women and children first" protocol
  was actively enforced.
- **Class mattered**: 63% of 1st-class passengers survived vs
  24% of 3rd-class, likely due to cabin proximity to the deck
  and differential treatment by crew.
- **Family size had a sweet spot**: passengers traveling with
  1–3 family members survived at above-average rates. Traveling
  alone was a disadvantage (no one to help navigate the chaos),
  and large groups (4+) were harder to coordinate during
  evacuation.
- **Embarkation port differences** (Cherbourg 55% vs Southampton
  34%) are likely a proxy for class composition, not a direct
  effect of the port itself.

## Model

A Random Forest classifier trained on 6 features achieves
**~82% cross-validated accuracy** (5-fold stratified CV).
The most important features were sex, fare, and age —
consistent with the EDA findings.

## Limitations

- The dataset covers 891 of the ~2,224 passengers on board —
  survival estimates carry uncertainty.
- Fare and passenger class are highly correlated, making it
  difficult to isolate the independent contribution of each.
- ~20% of age values were missing and imputed with the median
  by sex and class, which introduces a small bias.
- The `deck` column (cabin location) had 77% missing values
  and could not be used, despite likely being informative.

## What I Would Add With More Time

- Analyze survival by deck/cabin zone, if more complete data
  were available.
- Cross family size with class — does traveling in a group
  help equally in 1st and 3rd class?
- Compare Random Forest against XGBoost and evaluate whether
  the gain justifies the added complexity.

## Stack

Python · pandas · seaborn · matplotlib · scikit-learn

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook notebooks/titanic_analysis.ipynb
```
