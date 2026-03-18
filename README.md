# Telecom Churn Predictor -- Retention Client Operateur Broadband CI

> **Probleme :** Un operateur telecom perd 2-3% de ses abonnes par mois. Chaque point de churn en moins = **+500M FCFA/an** de revenus preserves pour un operateur de 500K abonnes. Ce modele predit le churn 30 jours a l'avance et recommande des actions de retention ciblees.

[![Streamlit](https://img.shields.io/badge/Live_Dashboard-Streamlit-FF4B4B?style=for-the-badge)](https://telecom-churn-ci.streamlit.app)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)

---

## Impact Business

| Action | Impact financier |
|--------|-----------------|
| Reduire le churn de 26% a 20% | **+~300M FCFA/an** preserves |
| Identifier les clients a risque | Cibler 20% pour sauver 80% du churn |
| Actions de retention ciblees | +15% de retention vs campagne generique |

## Donnees

| Source | Description |
|--------|-------------|
| [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) | 7,043 clients reels, 21 features, dataset de reference |
| Enrichissement CI | Regions ivoiriennes, forfaits FCFA, segments RFM |

> Le dataset est telecharge automatiquement depuis le repo GitHub officiel d'IBM. Sinon : [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Pipeline

```
01_prepare_data.py       02_churn_model.py         03_segmentation.py        dashboard/app.py
+------------------+     +-------------------+     +------------------+      +--------------+
| IBM Telco 7K     |---->| 4 modeles ML      |---->| Segments RFM     |----->| Streamlit    |
| + enrichment CI  |     | LogReg/RF/GB/XGB  |     | Actions retention|     | 4 onglets    |
| Regions, forfaits|     | AUC ~0.84         |     | ROI estimation   |     | Interactif   |
+------------------+     +-------------------+     +------------------+      +--------------+
```

## Quick Start

```bash
pip install -r requirements.txt
python python/pipeline.py
streamlit run dashboard/app.py
```

## Auteur

**Kouame Ruben** -- Senior Data Analyst | 2 ans d'experience en telecom broadband (KPI, commercial, marketing)
- [LinkedIn](https://www.linkedin.com/in/kouameruben/) | [GitHub](https://github.com/kouameruben)

## Licence

MIT License
