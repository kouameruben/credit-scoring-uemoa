# Dictionnaire des Donnees -- Telecom Churn Predictor

## Table: telco_final (7,043 lignes)

### Variables IBM originales
| Colonne | Type | Description |
|---------|------|-------------|
| customerID | VARCHAR | Identifiant client |
| gender | VARCHAR | Male / Female |
| SeniorCitizen | INT 0/1 | Senior (>65 ans) |
| Partner | VARCHAR | Yes / No |
| Dependents | VARCHAR | Yes / No |
| tenure | INTEGER | Mois d'anciennete (0-72) |
| PhoneService | VARCHAR | Service telephonique |
| InternetService | VARCHAR | DSL / Fiber optic / No |
| Contract | VARCHAR | Month-to-month / One year / Two year |
| MonthlyCharges | DECIMAL | Charges mensuelles ($) |
| TotalCharges | DECIMAL | Charges totales ($) |
| Churn | VARCHAR | Yes / No (TARGET ~26.5%) |

### Variables enrichies CI
| Colonne | Type | Description |
|---------|------|-------------|
| charges_mensuelles_fcfa | INTEGER | MonthlyCharges * 600 |
| region | VARCHAR | 10 regions CI |
| forfait | VARCHAR | Fibre-10/25/50/100 Mbps |
| satisfaction_score | INT [1-5] | Score satisfaction |
| segment | VARCHAR | Nouveau/Actif/Fidele/VIP |
| churn_proba | DECIMAL | Probabilite de churn [0-1] |
| risque | VARCHAR | Faible/Moyen/Eleve |
| action_retention | VARCHAR | Recommandation ciblee |
