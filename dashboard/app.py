"""
app.py -- Telecom Churn Predictor Dashboard
Author: Kouame Ruben
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import subprocess, sys, os
from pathlib import Path

st.set_page_config(page_title="Telecom Churn Predictor", page_icon="", layout="wide")

# -- Auto-run pipeline --
def ensure_data():
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)
    if not (root / "data" / "processed" / "telco_final.parquet").exists():
        with st.spinner("Running churn pipeline (~30s)..."):
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            result = subprocess.run(
                [sys.executable, str(root / "python" / "pipeline.py")],
                capture_output=True, text=True, cwd=str(root),
                env=env, encoding="utf-8", errors="replace"
            )
            if result.returncode != 0:
                st.error(f"Pipeline failed: {result.stderr[-500:]}")
                st.stop()

ensure_data()

@st.cache_data
def load():
    root = Path(__file__).resolve().parent.parent
    base = root / "data" / "processed"
    d = {}
    for f in ["telco_final", "model_comparison", "feature_importance", "roc_curve",
              "segment_kpis", "region_kpis", "at_risk_customers"]:
        p = base / f"{f}.parquet"
        if p.exists():
            d[f] = pd.read_parquet(p)
    source_file = base / "data_source.txt"
    if source_file.exists():
        d["source"] = source_file.read_text(encoding="utf-8").strip()
    return d

data = load()
df = data.get("telco_final", pd.DataFrame())
models = data.get("model_comparison", pd.DataFrame())
feat_imp = data.get("feature_importance", pd.DataFrame())
roc = data.get("roc_curve", pd.DataFrame())
seg_kpis = data.get("segment_kpis", pd.DataFrame())
region_kpis = data.get("region_kpis", pd.DataFrame())
at_risk = data.get("at_risk_customers", pd.DataFrame())

# -- Header --
st.markdown("# Telecom Churn Predictor")
source_info = data.get("source", "")
date_line = [l for l in source_info.split("\n") if l.startswith("Date:")]
if date_line:
    st.markdown(f"*Prediction de resiliation et strategie de retention - Donnees du **{date_line[0].replace('Date: ', '')}***")
else:
    st.markdown("*Prediction de resiliation et strategie de retention - Operateur Broadband CI*")
st.markdown("---")

# -- KPIs --
if len(df) > 0:
    churn_rate = df["churn"].mean()
    high_risk = (df["risque"] == "Eleve").sum() if "risque" in df.columns else 0
    revenue_risk = df[df["risque"] == "Eleve"]["charges_mensuelles_fcfa"].sum() if "risque" in df.columns and "charges_mensuelles_fcfa" in df.columns else 0
    best_auc = models["auc_roc"].max() if len(models) > 0 else 0
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Clients total", f"{len(df):,}")
    c2.metric("Taux de churn", f"{churn_rate:.1%}")
    c3.metric("AUC modele", f"{best_auc:.4f}")
    c4.metric("Clients a risque", f"{high_risk:,}")
    c5.metric("Revenu menace/mois", f"{revenue_risk/1e6:.1f}M FCFA")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["Vue Globale", "Clients a Risque", "Segmentation", "Performance Modele"])

# -- TAB 1: GLOBAL --
with tab1:
    if len(df) > 0 and "region" in df.columns:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Churn par region")
            reg = df.groupby("region")["churn"].mean().sort_values(ascending=False).reset_index()
            reg.columns = ["Region", "Taux Churn"]
            fig = px.bar(reg, x="Taux Churn", y="Region", orientation="h",
                          color="Taux Churn", color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"])
            fig.update_layout(height=350, template="plotly_white", showlegend=False)
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.markdown("### Churn par forfait")
            if "forfait" in df.columns:
                plan = df.groupby("forfait", observed=True).agg(
                    clients=("customerID", "count"), churn_rate=("churn", "mean"),
                    arpu=("charges_mensuelles_fcfa", "mean")
                ).reset_index()
                fig2 = px.scatter(plan, x="arpu", y="churn_rate", size="clients",
                                   color="forfait", size_max=50,
                                   labels={"churn_rate": "Taux Churn", "arpu": "ARPU (FCFA)"})
                fig2.update_layout(height=350, template="plotly_white")
                st.plotly_chart(fig2, width='stretch')
        
        st.markdown("### Churn par type de contrat")
        if "Contract" in df.columns:
            contract = df.groupby("Contract")["churn"].mean().sort_values(ascending=False).reset_index()
            fig3 = px.bar(contract, x="Contract", y="churn", color="churn",
                           color_continuous_scale=["#10B981", "#EF4444"],
                           labels={"churn": "Taux de churn"})
            fig3.update_layout(height=300, template="plotly_white", showlegend=False)
            st.plotly_chart(fig3, width='stretch')

# -- TAB 2: AT RISK --
with tab2:
    st.markdown("### Top 50 clients a risque - Actions de retention recommandees")
    if len(at_risk) > 0:
        st.dataframe(at_risk, width='stretch')
        
        saved = at_risk["charges_mensuelles_fcfa"].sum() * 0.30 * 12 if "charges_mensuelles_fcfa" in at_risk.columns else 0
        st.success(f"Si 30% de ces clients sont retenus : **{saved/1e6:,.1f}M FCFA/an** preserves")
    
    st.markdown("### Repartition des actions de retention")
    if len(df) > 0 and "action_retention" in df.columns:
        actions = df[df["action_retention"] != "Aucune action"]["action_retention"].value_counts().reset_index()
        actions.columns = ["Action", "Nombre de clients"]
        if len(actions) > 0:
            fig4 = px.bar(actions, x="Nombre de clients", y="Action", orientation="h",
                           color="Nombre de clients", color_continuous_scale=["#93C5FD", "#1D4ED8"])
            fig4.update_layout(height=300, template="plotly_white", showlegend=False,
                                yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig4, width='stretch')

# -- TAB 3: SEGMENTATION --
with tab3:
    if len(seg_kpis) > 0:
        st.markdown("### KPIs par segment")
        st.dataframe(seg_kpis, width='stretch')
        
        fig5 = px.bar(seg_kpis, x="segment", y="nb_clients", color="churn_rate",
                       color_continuous_scale=["#10B981", "#EF4444"], text="nb_clients",
                       labels={"nb_clients": "Clients", "churn_rate": "Taux Churn"})
        fig5.update_layout(height=350, template="plotly_white")
        st.plotly_chart(fig5, width='stretch')
    
    if len(region_kpis) > 0:
        st.markdown("### KPIs par region")
        st.dataframe(region_kpis, width='stretch')

# -- TAB 4: MODEL --
with tab4:
    if len(models) > 0:
        st.markdown("### Comparaison des modeles")
        st.dataframe(models.style.highlight_max(subset=["auc_roc", "gini", "ks", "accuracy"],
                                                  color="#d4edda"), width='stretch')
    
    if len(roc) > 0:
        st.markdown("### Courbe ROC")
        auc_val = models["auc_roc"].max() if len(models) > 0 else 0
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines",
                                      name="Best Model", line=dict(color="#0EA5E9", width=2)))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random",
                                      line=dict(color="#94A3B8", dash="dash")))
        fig_roc.update_layout(height=400, template="plotly_white",
                               title=f"Courbe ROC (AUC = {auc_val:.4f})",
                               xaxis_title="FPR", yaxis_title="TPR")
        st.plotly_chart(fig_roc, width='stretch')
    
    if len(feat_imp) > 0:
        st.markdown("### Importance des variables")
        fig_feat = px.bar(feat_imp, x="importance_pct", y="feature", orientation="h",
                           color="importance_pct", color_continuous_scale=["#93C5FD", "#1D4ED8"])
        fig_feat.update_layout(height=400, template="plotly_white",
                                yaxis=dict(autorange="reversed"), showlegend=False)
        st.plotly_chart(fig_feat, width='stretch')

st.markdown("---")
st.caption("Telecom Churn Predictor - Kouame Ruben | [GitHub](https://github.com/kouameruben) | [LinkedIn](https://linkedin.com/in/kouameruben)")
