# absenteeism_dashboard.py
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)
from sklearn.cluster import KMeans

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Absenteeism Analytics Dashboard",
    layout="wide"
)

st.title("📊 Absenteeism Analytics Dashboard")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df

DATA_PATH = r"C:\Users\nseli\Absenteeism_clean_basic.csv"
df_raw = load_data(DATA_PATH)
df = df_raw.copy()

# --------------------------------------------------
# BASIC FEATURE ENGINEERING
# --------------------------------------------------
if "High_absenteeism" not in df.columns:
    df["High_absenteeism"] = (df["Absenteeism time in hours"] >= 8).astype(int)

if "Day of the week" in df.columns:
    dow_map = {2: "Mon", 3: "Tue", 4: "Wed", 5: "Thu", 6: "Fri"}
    df["Day_name"] = df["Day of the week"].map(dow_map)

# pick reason column name once
REASON_COL = "Reason_desc" if "Reason_desc" in df.columns else (
    "Reason for absence" if "Reason for absence" in df.columns else None
)

# --------------------------------------------------
# SIDEBAR FILTERS (ENHANCED)
# --------------------------------------------------
st.sidebar.header("Filters")

# Seasons
if "Seasons" in df.columns:
    seasons = sorted(df["Seasons"].dropna().unique())
    selected_seasons = st.sidebar.multiselect(
        "Season(s)", options=seasons, default=seasons
    )
    df = df[df["Seasons"].isin(selected_seasons)]

# Months
if "Month of absence" in df.columns:
    months = sorted(df["Month of absence"].dropna().unique())
    selected_months = st.sidebar.multiselect(
        "Month(s)", options=months, default=months
    )
    df = df[df["Month of absence"].isin(selected_months)]

# Age range
if "Age" in df.columns:
    min_age, max_age = int(df["Age"].min()), int(df["Age"].max())
    age_range = st.sidebar.slider(
        "Age range", min_age, max_age, (min_age, max_age)
    )
    df = df[(df["Age"] >= age_range[0]) & (df["Age"] <= age_range[1])]

# Service time range
if "Service time" in df.columns:
    min_serv, max_serv = int(df["Service time"].min()), int(df["Service time"].max())
    serv_range = st.sidebar.slider(
        "Service time (years)", min_serv, max_serv, (min_serv, max_serv)
    )
    df = df[(df["Service time"] >= serv_range[0]) & (df["Service time"] <= serv_range[1])]

# High / low absenteeism filter
if "High_absenteeism" in df.columns:
    high_filter = st.sidebar.selectbox(
        "High absenteeism filter",
        ["All", "High only", "Low only"],
        index=0
    )
    if high_filter == "High only":
        df = df[df["High_absenteeism"] == 1]
    elif high_filter == "Low only":
        df = df[df["High_absenteeism"] == 0]

# Reason filter
if REASON_COL is not None:
    reasons = df[REASON_COL].dropna().unique()
    reasons = sorted(reasons)
    selected_reasons = st.sidebar.multiselect(
        "Reason(s) for absence", options=reasons, default=reasons
    )
    df = df[df[REASON_COL].isin(selected_reasons)]

# Day-of-week filter
if "Day_name" in df.columns:
    days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    selected_days = st.sidebar.multiselect(
        "Day(s) of week", options=days, default=days
    )
    df = df[df["Day_name"].isin(selected_days)]

st.sidebar.markdown("---")
st.sidebar.write("Filters apply to all tabs.")

# --------------------------------------------------
# KPIs (ENHANCED)
# --------------------------------------------------
kpi1, kpi3, kpi4, kpi5 = st.columns(4)


total_hours = df["Absenteeism time in hours"].sum()
avg_abs = df["Absenteeism time in hours"].mean()
median_abs = df["Absenteeism time in hours"].median()
high_abs_pct = (df["High_absenteeism"].mean() * 100) if len(df) > 0 else 0

if "ID" in df.columns:
    num_employees = df["ID"].nunique()
else:
    num_employees = None

kpi1.metric("Employees (Filtered)", num_employees if num_employees is not None else "-")
kpi3.metric("Total Hours Missed", f"{total_hours:.0f}")
kpi4.metric("Avg Hours per Record", f"{avg_abs:.2f}")
kpi5.metric("% High Absenteeism", f"{high_abs_pct:.1f}%")

st.markdown("---")

# --------------------------------------------------
# TABS
# --------------------------------------------------
tab_overview, tab_eda, tab_models, tab_clusters, = st.tabs(
    ["Overview", "EDA", "Models", "Clusters"]
)

# --------------------------------------------------
# OVERVIEW TAB
# --------------------------------------------------
with tab_overview:
    st.subheader("Absenteeism by Reason (Top 10)")

    if REASON_COL is not None:
        reason_abs = (
            df.groupby(REASON_COL)["Absenteeism time in hours"]
              .sum()
              .sort_values(ascending=False)
              .head(10)
              .reset_index()
        )

        fig_reason = px.bar(
            reason_abs,
            x=REASON_COL,
            y="Absenteeism time in hours",
            labels={"Absenteeism time in hours": "Total Hours"},
            title="Top Reasons by Total Absenteeism Hours"
        )
        fig_reason.update_layout(xaxis_tickangle=-45, height=450)
        st.plotly_chart(fig_reason, use_container_width=True)
    else:
        st.info("Reason column not found.")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Monthly Absenteeism Trend")
        if "Month of absence" in df.columns:
            month_abs = (
                df.groupby("Month of absence")["Absenteeism time in hours"]
                  .sum()
                  .reset_index()
                  .sort_values("Month of absence")
            )
            fig_month = px.line(
                month_abs,
                x="Month of absence",
                y="Absenteeism time in hours",
                markers=True,
                labels={"Absenteeism time in hours": "Total Hours"},
                title="Total Absenteeism Hours per Month"
            )
            st.plotly_chart(fig_month, use_container_width=True)
        else:
            st.info("Column 'Month of absence' not found.")

    with col_b:
        st.subheader("Absenteeism by Day of the Week")
        if "Day_name" in df.columns:
            dow_abs = (
                df.groupby("Day_name")["Absenteeism time in hours"]
                  .sum()
                  .reindex(["Mon", "Tue", "Wed", "Thu", "Fri"])
                  .reset_index()
            )
            fig_dow = px.bar(
                dow_abs,
                x="Day_name",
                y="Absenteeism time in hours",
                labels={"Day_name": "Day", "Absenteeism time in hours": "Total Hours"},
                title="Total Absenteeism Hours by Day"
            )
            st.plotly_chart(fig_dow, use_container_width=True)
        else:
            st.info("Day-of-week information not found.")

    st.subheader("Seasonal Absenteeism Trend")
    if "Seasons" in df.columns:
        season_abs = (
            df.groupby("Seasons")["Absenteeism time in hours"]
              .sum()
              .reset_index()
              .sort_values("Seasons")
        )
        fig_season = px.bar(
            season_abs,
            x="Seasons",
            y="Absenteeism time in hours",
            labels={"Absenteeism time in hours": "Total Hours"},
            title="Total Absenteeism Hours by Season"
        )
        st.plotly_chart(fig_season, use_container_width=True)
    else:
        st.info("Column 'Seasons' not found.")

# --------------------------------------------------
# EDA TAB
# --------------------------------------------------
with tab_eda:
    st.subheader("Distribution of Absenteeism Hours")

    fig_dist = px.histogram(
        df,
        x="Absenteeism time in hours",
        nbins=20,
        labels={"Absenteeism time in hours": "Hours"},
        title="Distribution of Absenteeism Time"
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    st.subheader("Top 15 Employees by Total Absenteeism")

    if "ID" in df.columns:
        emp_abs = (
            df.groupby("ID")["Absenteeism time in hours"]
              .sum()
              .sort_values(ascending=False)
              .head(15)
              .reset_index()
        )
        fig_emp = px.bar(
            emp_abs,
            x="ID",
            y="Absenteeism time in hours",
            labels={"Absenteeism time in hours": "Total Hours"},
            title="Top 15 Employees by Absenteeism"
        )
        st.plotly_chart(fig_emp, use_container_width=True)
    else:
        st.info("Column 'ID' not found.")

    st.subheader("Correlation Heatmap (numeric only)")
    corr = df.select_dtypes(include=[np.number]).corr()
    fig_corr = px.imshow(
        corr,
        title="Correlation Heatmap",
        height=700
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# --------------------------------------------------
# MODELS TAB
# --------------------------------------------------
with tab_models:
    st.subheader("Predictive Models: Logistic Regression vs Random Forest")

    model_df = df.copy()

    # columns to drop if present
    drop_cols = []
    for col in ["ID", "Reason_desc"]:
        if col in model_df.columns:
            drop_cols.append(col)
    model_df = model_df.drop(columns=drop_cols, errors="ignore")

    y = model_df["High_absenteeism"]
    X = model_df.drop(columns=["High_absenteeism", "Absenteeism time in hours"])

    # one-hot encode categoricals
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    # ----- Logistic Regression -----
    st.markdown("### Logistic Regression")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    log_model = LogisticRegression(max_iter=5000)
    log_model.fit(X_train_scaled, y_train)
    y_pred_log = log_model.predict(X_test_scaled)
    y_prob_log = log_model.predict_proba(X_test_scaled)[:, 1]

    acc_log = accuracy_score(y_test, y_pred_log)
    auc_log = roc_auc_score(y_test, y_prob_log)
    st.write(f"**Accuracy:** {acc_log:.3f} | **AUC:** {auc_log:.3f}")

    # ----- Random Forest -----
    st.markdown("### Random Forest (class_weight='balanced')")
    rf_model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )
    rf_model.fit(X_train_imp, y_train)
    y_pred_rf = rf_model.predict(X_test_imp)
    y_prob_rf = rf_model.predict_proba(X_test_imp)[:, 1]

    acc_rf = accuracy_score(y_test, y_pred_rf)
    auc_rf = roc_auc_score(y_test, y_prob_rf)
    st.write(f"**Accuracy:** {acc_rf:.3f} | **AUC:** {auc_rf:.3f}")

    st.markdown("**Random Forest Classification Report**")
    cr = classification_report(y_test, y_pred_rf, output_dict=True)
    cr_df = pd.DataFrame(cr).T
    st.dataframe(cr_df.style.format("{:.2f}"))

    st.markdown("**Random Forest Confusion Matrix**")
    cm = confusion_matrix(y_test, y_pred_rf)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"]
    )
    st.dataframe(cm_df)

    # ROC curves
    st.subheader("ROC Curve Comparison")
    fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr_log, y=tpr_log, mode="lines",
        name=f"Logistic (AUC={auc_log:.2f})"
    ))
    fig_roc.add_trace(go.Scatter(
        x=fpr_rf, y=tpr_rf, mode="lines",
        name=f"Random Forest (AUC={auc_rf:.2f})"
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines", line=dict(dash="dash"),
        name="Random Guess"
    ))
    fig_roc.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        title="ROC Curve Comparison"
    )
    st.plotly_chart(fig_roc, use_container_width=True)
# --------------------------------------------------
# CLUSTERS TAB
# --------------------------------------------------
with tab_clusters:
    st.subheader("K-Means Clustering of Employees")

    cluster_cols = [
        "Absenteeism time in hours",
        "Work load Average/day ",
        "Distance from Residence to Work",
        "Age",
        "Body mass index",
        "Service time",
        "Hit target"
    ]
    # keep only those that actually exist
    cluster_cols = [c for c in cluster_cols if c in df.columns]

    if len(cluster_cols) < 2:
        st.info("Not enough numeric columns available for clustering.")
    else:
        cluster_df = df[cluster_cols].copy()

        scaler_cl = StandardScaler()
        cluster_scaled = scaler_cl.fit_transform(cluster_df)

        k = 3
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(cluster_scaled)

        df_cl = df.copy()
        df_cl["Cluster"] = labels

        st.markdown("**Cluster Sizes**")
        st.write(df_cl["Cluster"].value_counts())

        st.markdown("**Cluster Profiles (Mean Values)**")
        profile = df_cl.groupby("Cluster")[cluster_cols].mean().round(2)
        st.dataframe(profile)

        st.subheader("Age vs Absenteeism by Cluster")
        if "Age" in df_cl.columns:
            fig_cl = px.scatter(
                df_cl,
                x="Age",
                y="Absenteeism time in hours",
                color="Cluster",
                title="Employee Clusters by Age and Absenteeism",
                labels={"Absenteeism time in hours": "Hours Missed"},
                hover_data=["Service time", "Body mass index"] if "Body mass index" in df_cl.columns else None
            )
            st.plotly_chart(fig_cl, use_container_width=True)
        else:
            st.info("Column 'Age' not available for plotting clusters.")

