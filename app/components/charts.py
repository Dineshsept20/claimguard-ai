"""Reusable Plotly chart components for ClaimGuard AI dashboard."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Professional color palette
COLORS = {
    "primary": "#1E3A5F",
    "secondary": "#4A90D9",
    "accent": "#E74C3C",
    "success": "#27AE60",
    "warning": "#F39C12",
    "critical": "#C0392B",
    "high": "#E74C3C",
    "medium": "#F39C12",
    "low": "#27AE60",
    "bg": "#F8F9FA",
}

RISK_COLORS = {
    "Critical": "#C0392B",
    "High": "#E74C3C",
    "Medium": "#F39C12",
    "Low": "#27AE60",
}

ANOMALY_COLORS = {
    "quantity_manipulation": "#E74C3C",
    "prescriber_pharmacy_collusion": "#C0392B",
    "doctor_shopping": "#8E44AD",
    "therapeutic_duplication": "#2980B9",
    "phantom_billing": "#D35400",
    "upcoding": "#F39C12",
    "refill_too_soon": "#16A085",
}

LAYOUT_DEFAULTS = dict(
    font=dict(family="Inter, sans-serif", size=12),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=40, r=40, t=50, b=40),
)


def anomaly_breakdown_pie(df: pd.DataFrame) -> go.Figure:
    """Pie chart of anomaly types among flagged claims."""
    flagged = df[df["is_anomaly"] == 1]
    counts = flagged["anomaly_type"].value_counts().reset_index()
    counts.columns = ["anomaly_type", "count"]

    colors = [ANOMALY_COLORS.get(t, COLORS["secondary"]) for t in counts["anomaly_type"]]

    fig = go.Figure(go.Pie(
        labels=counts["anomaly_type"].str.replace("_", " ").str.title(),
        values=counts["count"],
        marker=dict(colors=colors),
        textinfo="percent+label",
        hole=0.4,
    ))
    fig.update_layout(title="Anomaly Type Distribution", **LAYOUT_DEFAULTS, showlegend=False)
    return fig


def risk_tier_bar(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of risk tier distribution."""
    tier_order = ["Critical", "High", "Medium", "Low"]
    counts = df["risk_tier"].value_counts().reindex(tier_order).fillna(0)

    fig = go.Figure(go.Bar(
        x=counts.values,
        y=counts.index,
        orientation="h",
        marker_color=[RISK_COLORS.get(t, COLORS["secondary"]) for t in counts.index],
        text=[f"{v:,.0f}" for v in counts.values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Claims by Risk Tier",
        xaxis_title="Number of Claims",
        yaxis=dict(autorange="reversed"),
        **LAYOUT_DEFAULTS,
    )
    return fig


def monthly_trend(df: pd.DataFrame) -> go.Figure:
    """Line chart of monthly flagged claims vs total."""
    df = df.copy()
    df["month"] = pd.to_datetime(df["service_date"]).dt.to_period("M").astype(str)

    monthly = df.groupby("month").agg(
        total=("claim_id", "count"),
        flagged=("ensemble_prediction", "sum"),
    ).reset_index()
    monthly["flag_rate"] = monthly["flagged"] / monthly["total"] * 100

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=monthly["month"], y=monthly["total"],
        name="Total Claims", marker_color=COLORS["secondary"], opacity=0.4,
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=monthly["month"], y=monthly["flagged"],
        name="Flagged Claims", mode="lines+markers",
        line=dict(color=COLORS["accent"], width=3),
        marker=dict(size=8),
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=monthly["month"], y=monthly["flag_rate"],
        name="Flag Rate (%)", mode="lines+markers",
        line=dict(color=COLORS["warning"], width=2, dash="dash"),
        marker=dict(size=6),
    ), secondary_y=True)

    fig.update_layout(title="Monthly Claims & Anomaly Trend", **LAYOUT_DEFAULTS,
                      legend=dict(orientation="h", y=1.12))
    fig.update_yaxes(title_text="Claims Count", secondary_y=False)
    fig.update_yaxes(title_text="Flag Rate (%)", secondary_y=True)
    return fig


def top_risky_entities(df: pd.DataFrame, entity_col: str, top_n: int = 10) -> go.Figure:
    """Bar chart of top risky entities by mean ensemble score."""
    entity_label = "Pharmacy" if "pharm" in entity_col else "Prescriber"
    top = (
        df.groupby(entity_col)["ensemble_score"]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=False)
        .head(top_n)
        .reset_index()
    )

    fig = go.Figure(go.Bar(
        x=top["mean"],
        y=top[entity_col],
        orientation="h",
        marker_color=COLORS["accent"],
        text=[f"{v:.3f} ({c:,} claims)" for v, c in zip(top["mean"], top["count"])],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"Top {top_n} Highest-Risk {entity_label}s",
        xaxis_title="Mean Ensemble Score",
        yaxis=dict(autorange="reversed"),
        **LAYOUT_DEFAULTS,
        height=400,
    )
    return fig


def roc_curve_chart(y_true, y_prob_dict: dict) -> go.Figure:
    """ROC curves for multiple models."""
    from sklearn.metrics import roc_curve, auc

    fig = go.Figure()
    colors = [COLORS["primary"], COLORS["accent"], COLORS["warning"], COLORS["success"]]

    for i, (name, y_prob) in enumerate(y_prob_dict.items()):
        if y_prob is None:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{name} (AUC={roc_auc:.3f})",
            line=dict(color=colors[i % len(colors)], width=2),
        ))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random", line=dict(color="gray", dash="dash"),
        showlegend=False,
    ))
    fig.update_layout(
        title="ROC Curves",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        **LAYOUT_DEFAULTS,
    )
    return fig


def pr_curve_chart(y_true, y_prob_dict: dict) -> go.Figure:
    """Precision-Recall curves for multiple models."""
    from sklearn.metrics import precision_recall_curve, average_precision_score

    fig = go.Figure()
    colors = [COLORS["primary"], COLORS["accent"], COLORS["warning"], COLORS["success"]]

    for i, (name, y_prob) in enumerate(y_prob_dict.items()):
        if y_prob is None:
            continue
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        fig.add_trace(go.Scatter(
            x=recall, y=precision, mode="lines",
            name=f"{name} (AP={ap:.3f})",
            line=dict(color=colors[i % len(colors)], width=2),
        ))

    fig.update_layout(
        title="Precision-Recall Curves",
        xaxis_title="Recall",
        yaxis_title="Precision",
        **LAYOUT_DEFAULTS,
    )
    return fig


def confusion_matrix_chart(y_true, y_pred, model_name: str = "Model") -> go.Figure:
    """Interactive confusion matrix heatmap."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    labels = ["Normal", "Anomaly"]

    # Annotate with counts and percentages
    total = cm.sum()
    text = [[f"{v:,}\n({v/total:.1%})" for v in row] for row in cm]

    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        text=text, texttemplate="%{text}",
        colorscale=[[0, "#EBF5FB"], [1, COLORS["primary"]]],
        showscale=False,
    ))
    fig.update_layout(
        title=f"{model_name} — Confusion Matrix",
        xaxis_title="Predicted", yaxis_title="Actual",
        yaxis=dict(autorange="reversed"),
        **LAYOUT_DEFAULTS,
        width=400, height=400,
    )
    return fig


def detection_rate_chart(per_type_df: pd.DataFrame) -> go.Figure:
    """Bar chart of detection rate by anomaly type."""
    per_type_df = per_type_df.sort_values("detection_rate", ascending=True)

    colors = [
        COLORS["success"] if r >= 0.9 else COLORS["warning"] if r >= 0.7 else COLORS["accent"]
        for r in per_type_df["detection_rate"]
    ]

    fig = go.Figure(go.Bar(
        x=per_type_df["detection_rate"] * 100,
        y=per_type_df["anomaly_type"].str.replace("_", " ").str.title(),
        orientation="h",
        marker_color=colors,
        text=[f"{r:.1f}%" for r in per_type_df["detection_rate"] * 100],
        textposition="outside",
    ))
    fig.update_layout(
        title="Detection Rate by Anomaly Type",
        xaxis_title="Detection Rate (%)",
        xaxis=dict(range=[0, 110]),
        **LAYOUT_DEFAULTS,
        height=350,
    )
    return fig


def shap_importance_bar(importance_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Horizontal bar chart of SHAP feature importance."""
    top = importance_df.head(top_n).sort_values("mean_abs_shap", ascending=True)

    fig = go.Figure(go.Bar(
        x=top["mean_abs_shap"],
        y=top["feature"].str.replace("_", " ").str.title(),
        orientation="h",
        marker_color=COLORS["primary"],
        text=[f"{v:.4f}" for v in top["mean_abs_shap"]],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"Top {top_n} Features by SHAP Importance",
        xaxis_title="Mean |SHAP Value|",
        **LAYOUT_DEFAULTS,
        height=max(350, top_n * 28),
    )
    return fig


def shap_waterfall_chart(claim_detail: pd.DataFrame, base_value: float, top_n: int = 10) -> go.Figure:
    """Waterfall chart showing SHAP contributions for a single claim."""
    top = claim_detail.head(top_n).iloc[::-1]  # Reverse for bottom-up

    fig = go.Figure(go.Waterfall(
        y=top["feature"].str.replace("_", " ").str.title(),
        x=top["shap_value"],
        orientation="h",
        connector=dict(line=dict(color="gray", width=1)),
        increasing=dict(marker=dict(color=COLORS["accent"])),
        decreasing=dict(marker=dict(color=COLORS["success"])),
        textposition="outside",
        text=[f"{v:+.3f}" for v in top["shap_value"]],
    ))
    fig.update_layout(
        title=f"SHAP Feature Contributions (Base: {base_value:.3f})",
        xaxis_title="SHAP Value (impact on risk)",
        **LAYOUT_DEFAULTS,
        height=max(350, top_n * 35),
    )
    return fig


def shap_dependence_chart(dep_data: pd.DataFrame, feature: str,
                          interaction: str | None = None) -> go.Figure:
    """Scatter plot showing SHAP dependence for a feature."""
    fig = go.Figure()

    if interaction and "interaction_value" in dep_data.columns:
        fig.add_trace(go.Scatter(
            x=dep_data["feature_value"],
            y=dep_data["shap_value"],
            mode="markers",
            marker=dict(
                color=dep_data["interaction_value"],
                colorscale="RdBu_r",
                size=4,
                opacity=0.6,
                colorbar=dict(title=interaction.replace("_", " ").title()),
            ),
        ))
    else:
        fig.add_trace(go.Scatter(
            x=dep_data["feature_value"],
            y=dep_data["shap_value"],
            mode="markers",
            marker=dict(color=COLORS["primary"], size=4, opacity=0.5),
        ))

    fig.update_layout(
        title=f"SHAP Dependence: {feature.replace('_', ' ').title()}",
        xaxis_title=feature.replace("_", " ").title(),
        yaxis_title="SHAP Value",
        **LAYOUT_DEFAULTS,
    )
    return fig


def prescriber_radar(presc_row: pd.Series, peer_avg: pd.Series) -> go.Figure:
    """Radar chart comparing prescriber to peer averages."""
    metrics = [
        "presc_controlled_ratio", "presc_top_pharmacy_pct",
        "presc_after_hours_ratio", "presc_cost_peer_zscore",
        "presc_claims_per_member",
    ]
    labels = ["Controlled Rx %", "Top Pharmacy %", "After Hours %", "Cost Z-Score", "Claims/Member"]

    available = [m for m in metrics if m in presc_row.index and m in peer_avg.index]
    labels = [labels[metrics.index(m)] for m in available]

    presc_vals = [float(presc_row[m]) for m in available]
    peer_vals = [float(peer_avg[m]) for m in available]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=presc_vals, theta=labels, fill="toself",
        name="This Prescriber", line=dict(color=COLORS["accent"]),
    ))
    fig.add_trace(go.Scatterpolar(
        r=peer_vals, theta=labels, fill="toself",
        name="Peer Average", line=dict(color=COLORS["secondary"]),
        opacity=0.5,
    ))
    fig.update_layout(
        title="Prescriber vs Peer Comparison",
        polar=dict(radialaxis=dict(visible=True)),
        **LAYOUT_DEFAULTS,
    )
    return fig
