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
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

# NOTE: file is in same folder as this script
DATA_PATH = "absenteeism_clean_basic.csv"
df_raw = load_data(DATA_PATH)        # unfiltered data
df = df_raw.copy()                   # filtered view will overwrite this

# --------------------------------------------------
# BASIC FEATURE ENGINEERING
# --------------------------------------------------
# High absenteeism flag if not already present
if "High_absenteeism" not in df.columns:
    df["High_absenteeism"] = (df["Absenteeism time in hours"] >= 8).astype(int)

# Day name
if "Day of the week" in df.columns:
    dow_map = {2: "Mon", 3: "Tue", 4: "Wed", 5: "Thu", 6: "Fri"}
    df["Day_name"] = df["Day of the week"].map(dow_map)

# Which column describes the reason?
REASON_COL = "Reason_desc" if "Reason_desc" in df.columns else (
    "Reason for absence" if "Reason for absence" in df.columns else None
)

# --------------------------------------------------
# SIDEBAR FILTERS (EXTENDED)
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

# BMI range
if "Body mass index" in df.columns:
    min_bmi, max_bmi = int(df["Body mass index"].min()), int(df["Body mass index"].max())
    bmi_range = st.sidebar.slider(
        "Body mass index (BMI)", min_bmi, max_bmi, (min_bmi, max_bmi)
    )
    df = df[(df["Body mass index"] >= bmi_range[0]) & (df["Body mass index"] <= bmi_range[1])]

# Distance range
if "Distance from Residence to Work" in df.columns:
    dmin, dmax = int(df["Distance from Residence to Work"].min()), int(df["Distance from Residence to Work"].max())
    dist_range = st.sidebar.slider(
        "Distance from Residence to Work", dmin, dmax, (dmin, dmax)
    )
    df = df[
        (df["Distance from Residence to Work"] >= dist_range[0])
        & (df["Distance from Residence to Work"] <= dist_range[1])
    ]

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
    reasons = sorted(df[REASON_COL].dropna().unique())
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

# In case filters remove everything
if df.empty:
    st.warning("No records match the current filters. Try relaxing one or more filters.")
    st.stop()

# --------------------------------------------------
# KPIs (EXTENDED)
# --------------------------------------------------

# --- Quick clustering for KPI (cluster sizes) ---
cluster_cols_base = [
    "Absenteeism time in hours",
    "Work load Average/day ",
    "Distance from Residence to Work",
    "Age",
    "Body mass index",
    "Service time",
    "Hit target",
]
cluster_cols_kpi = [c for c in cluster_cols_base if c in df.columns]

cluster_kpi_text = "N/A"
if len(cluster_cols_kpi) >= 2:
    from sklearn.preprocessing import StandardScaler as _Std  # local import for safety
    from sklearn.cluster import KMeans as _KMeans

    cluster_df_kpi = df[cluster_cols_kpi].copy()
    scaler_kpi = _Std()
    cluster_scaled_kpi = scaler_kpi.fit_transform(cluster_df_kpi)

    k_kpi = 3
    kmeans_kpi = _KMeans(n_clusters=k_kpi, random_state=42, n_init="auto")
    labels_kpi = kmeans_kpi.fit_predict(cluster_scaled_kpi)

    df_kpi_clusters = df.copy()
    df_kpi_clusters["Cluster"] = labels_kpi
    sizes = df_kpi_clusters["Cluster"].value_counts().sort_index()

    # Example: "0: 138 | 1: 248 | 2: 53"
    cluster_kpi_text = " | ".join(
        [f"{i}: {int(sizes.get(i, 0))}" for i in range(k_kpi)]
    )

# --- KPI VALUES ---
total_hours = df["Absenteeism time in hours"].sum()
avg_abs = df["Absenteeism time in hours"].mean()
median_abs = df["Absenteeism time in hours"].median()
high_abs_pct = (
    df["High_absenteeism"].mean() * 100
    if "High_absenteeism" in df.columns
    else 0
)

num_employees = df["ID"].nunique() if "ID" in df.columns else None
avg_bmi = df["Body mass index"].mean() if "Body mass index" in df.columns else np.nan
avg_dist = (
    df["Distance from Residence to Work"].mean()
    if "Distance from Residence to Work" in df.columns
    else np.nan
)

# --- LAYOUT: 3 KPIs PER ROW ---
row1 = st.columns(3)
row2 = st.columns(3)

with row1[0]:
    st.metric(
        "Employees (Filtered)",
        num_employees if num_employees is not None else "-"
    )

with row1[1]:
    st.metric("Employees per Cluster (0 | 1 | 2)", cluster_kpi_text)

with row1[2]:
    st.metric("Total Hours Missed", f"{total_hours:.0f}")

with row2[0]:
    st.metric("Avg Hours / Record", f"{avg_abs:.2f}")

with row2[1]:
    st.metric("% High Absenteeism", f"{high_abs_pct:.1f}%")

with row2[2]:
    st.metric(
        "Avg BMI / Distance",
        (
            f"{avg_bmi:.1f} BMI | {avg_dist:.1f} km"
            if not np.isnan(avg_bmi) and not np.isnan(avg_dist)
            else "-"
        ),
    )

st.markdown("---")


# --------------------------------------------------
# TABS
# --------------------------------------------------
tab_overview, tab_eda, tab_models, tab_clusters = st.tabs(
    ["Overview", "EDA", "Models", "Clusters"]
)

# --------------------------------------------------
# HELPER: summary insights
# --------------------------------------------------
def generate_insights(df_view: pd.DataFrame) -> str:
    lines = []

    # Top reason
    if REASON_COL is not None and not df_view.empty:
        reason_summary = (
            df_view.groupby(REASON_COL)["Absenteeism time in hours"]
            .sum()
            .sort_values(ascending=False)
        )
        if len(reason_summary) > 0:
            top_reason = reason_summary.index[0]
            top_reason_hours = reason_summary.iloc[0]
            lines.append(
                f"- **Top reason:** `{top_reason}` with **{top_reason_hours:.0f} hours** missed."
            )

    # Peak month
    if "Month of absence" in df_view.columns:
        month_summary = (
            df_view.groupby("Month of absence")["Absenteeism time in hours"]
            .sum()
            .sort_values(ascending=False)
        )
        if len(month_summary) > 0:
            peak_month = int(month_summary.index[0])
            lines.append(f"- **Peak month:** Month **{peak_month}** has the highest total absence.")

    # Peak day of week
    if "Day_name" in df_view.columns:
        dow_summary = (
            df_view.groupby("Day_name")["Absenteeism time in hours"]
            .sum()
            .reindex(["Mon", "Tue", "Wed", "Thu", "Fri"])
            .dropna()
        )
        if len(dow_summary) > 0:
            peak_day = dow_summary.idxmax()
            lines.append(f"- **Busiest absence day:** **{peak_day}** has the most hours missed.")

    # Peak season
    if "Seasons" in df_view.columns:
        season_summary = (
            df_view.groupby("Seasons")["Absenteeism time in hours"]
            .sum()
            .sort_values(ascending=False)
        )
        if len(season_summary) > 0:
            peak_season = int(season_summary.index[0])
            lines.append(f"- **Highest season:** Season **{peak_season}** shows the most absenteeism.")

    if not lines:
        return "No insights available for the current filters."
    return "\n".join(lines)

# --------------------------------------------------
# OVERVIEW TAB
# --------------------------------------------------
with tab_overview:
    st.markdown("### Absenteeism by Reason (Top 10)")
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

    st.subheader("High vs Low Absenteeism by Reason (Top 10)")
    if REASON_COL is not None and "High_absenteeism" in df.columns:
        reason_hi_lo = (
            df.groupby([REASON_COL, "High_absenteeism"])["Absenteeism time in hours"]
              .sum()
              .reset_index()
        )
        top_reasons = (
            reason_hi_lo.groupby(REASON_COL)["Absenteeism time in hours"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .index
        )
        reason_hi_lo = reason_hi_lo[reason_hi_lo[REASON_COL].isin(top_reasons)]
        reason_hi_lo["High label"] = reason_hi_lo["High_absenteeism"].map({0: "Low", 1: "High"})

        fig_stack = px.bar(
            reason_hi_lo,
            x=REASON_COL,
            y="Absenteeism time in hours",
            color="High label",
            labels={"Absenteeism time in hours": "Total Hours", "High label": "Absenteeism level"},
            barmode="stack",
            title="High vs Low Absenteeism by Reason"
        )
        fig_stack.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_stack, use_container_width=True)

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

    col1, col2 = st.columns(2)

    with col1:
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

with col2:
    st.subheader("Absenteeism by Age Group")
    if "Age" in df.columns:
        age_bins = [20, 30, 40, 50, 60]
        df_age = df.copy()

        # Create interval bins
        df_age["Age group"] = pd.cut(df_age["Age"], bins=age_bins, right=False)

        # Convert intervals to string labels so Plotly can serialize them
        df_age["Age group label"] = df_age["Age group"].astype(str)

        age_group_abs = (
            df_age.groupby("Age group label")["Absenteeism time in hours"]
            .sum()
            .reset_index()
        )

        fig_age = px.bar(
            age_group_abs,
            x="Age group label",
            y="Absenteeism time in hours",
            labels={
                "Age group label": "Age group",
                "Absenteeism time in hours": "Total Hours"
            },
            title="Total Absenteeism by Age Group"
        )
        st.plotly_chart(fig_age, use_container_width=True)
    else:
        st.info("Column 'Age' not found.")

with tab_eda:

    st.subheader("Correlation Heatmap (Numeric Variables)")

    num_df = df.select_dtypes(include=[np.number])

    if not num_df.empty:
        corr = num_df.corr()

        blue_scale = [
            [0.0, "#99CCFF"],
            [1.0, "#0066CC"]
        ]

        # LEFT-ALIGNED HEATMAP
        left_col, right_col = st.columns([4, 1])   # <- widen left side

        with left_col:
            fig_corr = px.imshow(
                corr.values,
                x=corr.columns,
                y=corr.index,
                color_continuous_scale=blue_scale,
                origin="lower",
                title="Correlation Heatmap",
                height=900,
                width=900          # <- reduce width to avoid huge layout
            )

            fig_corr.update_xaxes(tickfont=dict(size=12), tickangle=45)
            fig_corr.update_yaxes(tickfont=dict(size=12))
            fig_corr.update_layout(
                title_font=dict(size=22),
                coloraxis_colorbar=dict(title="Correlation", tickfont=dict(size=10))
            )

            st.plotly_chart(fig_corr, use_container_width=False)

        with right_col:
            st.empty()

    else:
        st.info("No numeric columns available for correlation heatmap.")


  

# --------------------------------------------------
# MODELS TAB (use FULL dataset, not filtered)
# --------------------------------------------------
with tab_models:
    st.subheader("Predictive Models: Logistic Regression vs Random Forest")

    model_df = df_raw.copy()   # << use unfiltered data to match notebook results

    # Drop ID and text columns that are not useful for prediction
    drop_cols = []
    for col in ["ID", "Reason_desc"]:
        if col in model_df.columns:
            drop_cols.append(col)
    model_df = model_df.drop(columns=drop_cols, errors="ignore")

    # Make sure High_absenteeism exists on raw data too
    if "High_absenteeism" not in model_df.columns:
        model_df["High_absenteeism"] = (
            model_df["Absenteeism time in hours"] >= 8
        ).astype(int)

    y = model_df["High_absenteeism"]
    X = model_df.drop(columns=["High_absenteeism", "Absenteeism time in hours"])

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    # Logistic Regression
    st.markdown("### Logistic Regression")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    log_model = LogisticRegression(max_iter=20000)
    log_model.fit(X_train_scaled, y_train)
    y_pred_log = log_model.predict(X_test_scaled)
    y_prob_log = log_model.predict_proba(X_test_scaled)[:, 1]

    acc_log = accuracy_score(y_test, y_pred_log)
    auc_log = roc_auc_score(y_test, y_prob_log)
    st.write(f"**Accuracy:** {acc_log:.3f} | **AUC:** {auc_log:.3f}")

    # Random Forest
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
    # Random Forest Feature Importance
    # --------------------------------------------------
    st.subheader("Key Drivers of High Absenteeism (Random Forest)")

    # rf_model is already trained above; X is the feature DataFrame used for training
    feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)

    # Take top 15 features and sort them for a neat horizontal bar chart
    fi_top = feature_importances.sort_values(ascending=False).head(15)
    fi_top = fi_top.sort_values(ascending=True)  # smallest at bottom, largest at top visually

    # Use blue theme similar to other charts
    fig_imp = px.bar(
        x=fi_top.values,
        y=fi_top.index,
        orientation="h",
        labels={"x": "Importance (Gini)", "y": "Feature"},
        title="Top 15 Features Driving High Absenteeism",
    )

    # Apply solid blue color to match your theme
    fig_imp.update_traces(marker_color="#0066CC")  # dark blue you used elsewhere

    fig_imp.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(showgrid=True),
    )

    st.plotly_chart(fig_imp, use_container_width=True)

# --------------------------------------------------
# CLUSTERS TAB (Improved, data-rich clustering)
# --------------------------------------------------
with tab_clusters:
    st.subheader("K-Means Clustering of Absenteeism Profiles")

    # --- 1. Choose features for clustering (numeric + categorical) ---
    numeric_candidates = [
        "Absenteeism time in hours",
        "High_absenteeism",
        "Work load Average/day ",
        "Distance from Residence to Work",
        "Age",
        "Body mass index",
        "Service time",
        "Hit target",
        "Son",
        "Pet",
    ]

    categorical_candidates = []

    # Behavioral / policy variables
    if "Social drinker" in df.columns:
        categorical_candidates.append("Social drinker")
    if "Social smoker" in df.columns:
        categorical_candidates.append("Social smoker")
    if "Disciplinary failure" in df.columns:
        categorical_candidates.append("Disciplinary failure")

    # Time-related
    if "Month of absence" in df.columns:
        categorical_candidates.append("Month of absence")
    if "Seasons" in df.columns:
        categorical_candidates.append("Seasons")

    # Reason for absence (use whichever column is available)
    if REASON_COL is not None and REASON_COL in df.columns:
        categorical_candidates.append(REASON_COL)

    # Keep only columns that actually exist in df
    numeric_cols = [c for c in numeric_candidates if c in df.columns]
    cat_cols = [c for c in categorical_candidates if c in df.columns]

    if len(numeric_cols) < 2:
        st.info("Not enough numeric columns available for clustering.")
    else:
        # --- 2. Build clustering dataframe ---
        cluster_df = df[numeric_cols + cat_cols].copy()

        # Handle missing values: median for numeric, mode for categorical
        for col in numeric_cols:
            if cluster_df[col].isna().any():
                cluster_df[col] = cluster_df[col].fillna(cluster_df[col].median())

        for col in cat_cols:
            if cluster_df[col].isna().any():
                mode_val = cluster_df[col].mode(dropna=True)
                if not mode_val.empty:
                    cluster_df[col] = cluster_df[col].fillna(mode_val.iloc[0])
                else:
                    cluster_df[col] = cluster_df[col].fillna("Unknown")

        # --- 3. One-hot encode categorical variables ---
        if cat_cols:
            cluster_df = pd.get_dummies(cluster_df, columns=cat_cols, drop_first=True)

        # --- 4. Scale features ---
        scaler_cl = StandardScaler()
        cluster_scaled = scaler_cl.fit_transform(cluster_df)

        # --- 5. Let user choose k ---
        k = st.slider("Number of clusters (k)", min_value=2, max_value=6, value=3, step=1)

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(cluster_scaled)

        # Attach clusters back to filtered df
        df_cl = df.copy()
        df_cl["Cluster"] = labels

        # --- 6. Show cluster sizes ---
        st.markdown("**Cluster Sizes (number of records)**")
        st.write(df_cl["Cluster"].value_counts().sort_index())

        # --- 7. Show numeric profiles for each cluster ---
        st.markdown("**Cluster Profiles (Mean Values)**")

        # Only use numeric columns that still exist in df_cl
        profile_cols = [c for c in numeric_cols if c in df_cl.columns]
        profile = df_cl.groupby("Cluster")[profile_cols].mean().round(2)
        st.dataframe(profile)

        # Also show proportion of high absenteeism by cluster (if available)
        if "High_absenteeism" in df_cl.columns:
            st.markdown("**Share of High Absenteeism by Cluster**")
            high_share = (df_cl.groupby("Cluster")["High_absenteeism"]
                            .mean()
                            .round(3))
            st.write(high_share)

        # --- 8. Visualization: Age vs Absenteeism by Cluster ---
        st.subheader("Age vs Absenteeism by Cluster")
        if "Age" in df_cl.columns:
            fig_cl = px.scatter(
                df_cl,
                x="Age",
                y="Absenteeism time in hours",
                color="Cluster",
                title="Employee Absenteeism Profiles by Cluster",
                labels={"Absenteeism time in hours": "Hours Missed"},
                hover_data=[
                    col for col in ["Service time", "Body mass index", "Work load Average/day "]
                    if col in df_cl.columns
                ]
            )
            st.plotly_chart(fig_cl, use_container_width=True)
        else:
            st.info("Column 'Age' not available for plotting clusters.")
