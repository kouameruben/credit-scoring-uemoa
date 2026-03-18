"""
01_prepare_data.py -- Telecom Churn Data Preparation
Author: Kouame Ruben
Sources:
  - IBM Telco Customer Churn (7,043 customers, 21 features)
  - Auto-download from GitHub or local file
  - Enrichment: adapted to Cote d'Ivoire broadband context
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

GITHUB_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
LOCAL_FILE = "data/raw/Telco-Customer-Churn.csv"


def download_dataset():
    """Download IBM Telco dataset from GitHub."""
    if not HAS_REQUESTS:
        return None
    try:
        print("   Downloading from GitHub (IBM official repo)...")
        resp = requests.get(GITHUB_URL, verify=False, timeout=30)
        resp.raise_for_status()
        if "customerID" in resp.text[:200]:
            Path("data/raw").mkdir(parents=True, exist_ok=True)
            Path(LOCAL_FILE).write_text(resp.text, encoding="utf-8")
            print(f"   [OK] Downloaded and saved to {LOCAL_FILE}")
            return LOCAL_FILE
    except Exception as e:
        print(f"   [!] Download failed: {e}")
    return None


def load_dataset():
    """Load dataset: local file first, then try download."""
    path = Path(LOCAL_FILE)
    
    if path.exists():
        df = pd.read_csv(path)
        print(f"   [OK] Loaded from local file: {len(df):,} rows")
        return df, "IBM Telco (local file)"
    
    # Try download
    downloaded = download_dataset()
    if downloaded:
        df = pd.read_csv(downloaded)
        return df, "IBM Telco (github.com/IBM)"
    
    print("   [ERROR] Dataset not found. Download manually:")
    print("   => https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
    print(f"   => Place the CSV in {LOCAL_FILE}")
    return None, None


def clean_ibm_data(df):
    """Clean the IBM Telco dataset."""
    df = df.copy()
    
    # TotalCharges has some spaces (should be numeric)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    
    # Encode target
    df["churn"] = (df["Churn"] == "Yes").astype(int)
    
    # Encode binary features
    binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in binary_cols:
        if col in df.columns:
            df[col + "_encoded"] = (df[col].isin(["Yes", "Male"])).astype(int)
    
    # Encode multi-category features
    for col in ["InternetService", "Contract", "PaymentMethod"]:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
    
    return df


def enrich_cote_ivoire(df):
    """Adapt IBM dataset to Cote d'Ivoire broadband context."""
    n = len(df)
    np.random.seed(2026)
    df = df.copy()
    
    # Convert USD to FCFA
    df["charges_mensuelles_fcfa"] = (df["MonthlyCharges"] * 600).astype(int)
    df["charges_totales_fcfa"] = (df["TotalCharges"] * 600).astype(int)
    
    # Map to CI regions based on tenure patterns
    regions = ["Cocody", "Yopougon", "Plateau", "Marcory", "Abidjan-Nord",
               "Abidjan-Sud", "Bouake", "Yamoussoukro", "San-Pedro", "Treichville"]
    df["region"] = np.random.choice(regions, n, p=[0.18, 0.20, 0.08, 0.10, 0.12,
                                                     0.10, 0.08, 0.05, 0.05, 0.04])
    
    # Map to CI broadband plans
    df["forfait"] = pd.cut(
        df["MonthlyCharges"],
        bins=[0, 35, 60, 85, 200],
        labels=["Fibre-10Mbps", "Fibre-25Mbps", "Fibre-50Mbps", "Fibre-100Mbps"]
    )
    
    forfait_prix = {"Fibre-10Mbps": 15000, "Fibre-25Mbps": 25000,
                     "Fibre-50Mbps": 40000, "Fibre-100Mbps": 75000}
    df["prix_forfait_fcfa"] = df["forfait"].map(forfait_prix)
    
    # Satisfaction score (derived from churn indicators)
    df["satisfaction_score"] = np.where(df["churn"] == 1,
        np.random.choice([1, 2, 3], n, p=[0.3, 0.4, 0.3]),
        np.random.choice([3, 4, 5], n, p=[0.3, 0.4, 0.3])
    )
    
    # RFM segment based on tenure
    df["segment"] = pd.cut(
        df["tenure"],
        bins=[0, 6, 18, 36, 100],
        labels=["Nouveau (<6m)", "Actif (6-18m)", "Fidele (18-36m)", "VIP (>36m)"]
    )
    
    return df


def main():
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    # Load
    df, source = load_dataset()
    if df is None:
        return
    
    # Clean
    df = clean_ibm_data(df)
    
    # Enrich
    df = enrich_cote_ivoire(df)
    
    # Save
    df.to_csv("data/raw/telco_enriched.csv", index=False)
    df.to_parquet("data/processed/telco_enriched.parquet", index=False)
    
    # Metadata
    churn_rate = df["churn"].mean()
    with open("data/processed/data_source.txt", "w", encoding="utf-8") as f:
        f.write(f"Source: {source}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Rows: {len(df)}\n")
        f.write(f"Churn rate: {churn_rate:.1%}\n")
    
    print(f"[OK] Telecom data preparation complete:")
    print(f"   Source:      {source}")
    print(f"   Rows:        {len(df):,}")
    print(f"   Columns:     {len(df.columns)}")
    print(f"   Churn rate:  {churn_rate:.1%}")
    print(f"   Regions CI:  {df['region'].nunique()}")
    print(f"   Forfaits:    {df['forfait'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
