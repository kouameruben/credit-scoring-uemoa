"""
03_segmentation.py -- Customer Segmentation & Retention Recommendations
Author: Kouame Ruben
"""

import pandas as pd
import numpy as np
from pathlib import Path


def generate_retention_actions(df):
    """Generate targeted retention recommendations for at-risk customers."""
    df = df.copy()
    
    conditions = [
        (df["churn_proba"] > 0.6) & (df["satisfaction_score"] <= 2),
        (df["churn_proba"] > 0.6) & (df["tenure"] > 24),
        (df["churn_proba"] > 0.4) & (df["Contract_One year"] == 0) & (df["Contract_Two year"] == 0),
        (df["churn_proba"] > 0.4) & (df["MonthlyCharges"] > 70),
        (df["churn_proba"] > 0.3),
    ]
    actions = [
        "Appel retention + upgrade offert",
        "Offre fidelite 3 mois -20%",
        "Proposition engagement 12 mois avec remise",
        "Migration forfait inferieur + bonus data",
        "Email personnalise + bonus data 1 mois",
    ]
    
    df["action_retention"] = np.select(conditions, actions, default="Aucune action")
    return df


def compute_segment_kpis(df):
    """Compute KPIs by segment."""
    if "segment" not in df.columns:
        return pd.DataFrame()
    
    seg = df.groupby("segment", observed=True).agg(
        nb_clients=("customerID", "count"),
        churn_rate=("churn", "mean"),
        arpu_fcfa=("charges_mensuelles_fcfa", "mean"),
        churn_proba_moy=("churn_proba", "mean"),
        tenure_moy=("tenure", "mean"),
    ).round(2).reset_index()
    
    seg["revenu_mensuel_total"] = (seg["nb_clients"] * seg["arpu_fcfa"]).astype(int)
    seg["revenu_a_risque"] = (seg["nb_clients"] * seg["arpu_fcfa"] * seg["churn_rate"]).astype(int)
    
    return seg


def compute_region_kpis(df):
    """Compute KPIs by region."""
    if "region" not in df.columns:
        return pd.DataFrame()
    
    reg = df.groupby("region").agg(
        nb_clients=("customerID", "count"),
        churn_rate=("churn", "mean"),
        arpu_fcfa=("charges_mensuelles_fcfa", "mean"),
    ).round(3).reset_index().sort_values("churn_rate", ascending=False)
    
    return reg


def main():
    df = pd.read_parquet("data/processed/telco_scored.parquet")
    
    # Retention actions
    # Need contract columns for conditions
    for col in ["Contract_One year", "Contract_Two year"]:
        if col not in df.columns:
            df[col] = 0
    
    df = generate_retention_actions(df)
    
    # Segment KPIs
    seg_kpis = compute_segment_kpis(df)
    region_kpis = compute_region_kpis(df)
    
    # Top at-risk customers
    at_risk = df.nlargest(50, "churn_proba")[[
        "customerID", "region", "forfait", "tenure", "charges_mensuelles_fcfa",
        "satisfaction_score", "churn_proba", "risque", "segment", "action_retention"
    ]].reset_index(drop=True)
    
    # Save
    df.to_parquet("data/processed/telco_final.parquet", index=False)
    seg_kpis.to_parquet("data/processed/segment_kpis.parquet", index=False)
    region_kpis.to_parquet("data/processed/region_kpis.parquet", index=False)
    at_risk.to_parquet("data/processed/at_risk_customers.parquet", index=False)
    
    # ROI estimation
    revenue_at_risk = df[df["risque"] == "Eleve"]["charges_mensuelles_fcfa"].sum()
    saved_30pct = int(revenue_at_risk * 0.30 * 12)
    
    print(f"[OK] Segmentation & retention complete:")
    print(f"   Segments:         {len(seg_kpis)}")
    print(f"   Regions:          {len(region_kpis)}")
    print(f"   Clients a risque: {(df['risque']=='Eleve').sum():,}")
    print(f"   Revenu mensuel menace: {revenue_at_risk:,.0f} FCFA")
    print(f"   Gain si 30% retenus:   {saved_30pct:,.0f} FCFA/an")
    
    actions_count = df["action_retention"].value_counts()
    for action, count in actions_count.items():
        if action != "Aucune action":
            print(f"   => {action}: {count:,} clients")


if __name__ == "__main__":
    main()
