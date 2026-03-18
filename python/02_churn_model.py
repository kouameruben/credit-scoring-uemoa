"""
02_churn_model.py -- Churn Prediction Model Training
Author: Kouame Ruben
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, classification_report

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# Features for the model (mix of IBM original + CI enrichment)
FEATURES = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "SeniorCitizen",
    "gender_encoded", "Partner_encoded", "Dependents_encoded",
    "PhoneService_encoded", "PaperlessBilling_encoded",
    "InternetService_Fiber optic", "InternetService_No",
    "Contract_One year", "Contract_Two year",
    "satisfaction_score",
]

TARGET = "churn"


def main():
    Path("models").mkdir(parents=True, exist_ok=True)
    
    df = pd.read_parquet("data/processed/telco_enriched.parquet")
    
    # Keep only features that exist
    features = [f for f in FEATURES if f in df.columns]
    
    X = df[features].copy()
    y = df[TARGET]
    
    # Fill NaN
    X = X.fillna(0)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"   Churn rate train: {y_train.mean():.1%} | test: {y_test.mean():.1%}")
    print(f"   Features: {len(features)}")
    print()
    
    # Scale for logistic regression
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                            random_state=42, use_label_encoder=False,
                                            eval_metric="auc", verbosity=0)
    
    results = []
    trained = {}
    
    for name, model in models.items():
        print(f"     Training {name}...", end=" ", flush=True)
        if "Logistic" in name:
            model.fit(X_train_sc, y_train)
            y_proba = model.predict_proba(X_test_sc)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, y_proba)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ks = max(tpr - fpr)
        acc = accuracy_score(y_test, (y_proba > 0.5).astype(int))
        
        results.append({"model": name, "auc_roc": round(auc, 4),
                         "gini": round(2*auc-1, 4), "ks": round(ks, 4), "accuracy": round(acc, 4)})
        trained[name] = model
        print(f"AUC={auc:.4f} | KS={ks:.4f}")
    
    results_df = pd.DataFrame(results)
    best_name = results_df.loc[results_df["auc_roc"].idxmax(), "model"]
    best_model = trained[best_name]
    
    # Feature importance
    if hasattr(best_model, "feature_importances_"):
        imp = pd.DataFrame({"feature": features, "importance": best_model.feature_importances_})
    else:
        imp = pd.DataFrame({"feature": features, "importance": np.abs(best_model.coef_[0])})
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
    imp["importance_pct"] = (100 * imp["importance"] / imp["importance"].sum()).round(1)
    
    # ROC data
    if "Logistic" in best_name:
        y_best = best_model.predict_proba(scaler.transform(X_test))[:, 1]
    else:
        y_best = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_best)
    
    # Predictions on full dataset for dashboard
    if "Logistic" in best_name:
        df["churn_proba"] = best_model.predict_proba(scaler.transform(df[features].fillna(0)))[:, 1]
    else:
        df["churn_proba"] = best_model.predict_proba(df[features].fillna(0))[:, 1]
    
    df["risque"] = pd.cut(df["churn_proba"], bins=[0, 0.15, 0.40, 1.0],
                           labels=["Faible", "Moyen", "Eleve"])
    
    # Save everything
    results_df.to_parquet("data/processed/model_comparison.parquet", index=False)
    imp.to_parquet("data/processed/feature_importance.parquet", index=False)
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_parquet("data/processed/roc_curve.parquet", index=False)
    df.to_parquet("data/processed/telco_scored.parquet", index=False)
    
    with open("models/best_model.pkl", "wb") as f:
        pickle.dump({"model": best_model, "name": best_name, "scaler": scaler, "features": features}, f)
    
    print()
    print(f"[OK] Churn model training complete:")
    print(f"   Best model:   {best_name} (AUC={results_df.loc[results_df['model']==best_name, 'auc_roc'].values[0]})")
    print(f"   Risque Eleve: {(df['risque']=='Eleve').sum():,} clients ({(df['risque']=='Eleve').mean():.0%})")
    print(f"   Risque Moyen: {(df['risque']=='Moyen').sum():,} clients ({(df['risque']=='Moyen').mean():.0%})")
    print(f"   Top 3 features: {', '.join(imp.head(3)['feature'].tolist())}")


if __name__ == "__main__":
    main()
