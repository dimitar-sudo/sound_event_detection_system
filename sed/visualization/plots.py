from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def visualize_class_distribution(
    Y_train: np.ndarray,
    Y_val: np.ndarray,
    Y_test: np.ndarray,
    class_names: List[str],
    save: bool = False,
    save_path: str = "class_freq_splits.png",
) -> None:
    """Plot class frequency (% of frames) across Train/Val/Test splits."""
    splits = {"Train": Y_train, "Val": Y_val, "Test": Y_test}
    colors = ["#4F81C7", "#E8735A", "#3DBFA8"]

    def _wrap_name(name: str, max_chars: int = 16) -> str:
        name = name.replace("_", " ").title()
        if len(name) <= max_chars:
            return name
        mid = len(name) // 2
        space = name.rfind(" ", 0, mid + 6)
        if space == -1:
            space = name.find(" ", mid)
        if space == -1:
            return name
        return name[:space] + "<br>" + name[space + 1:]

    labels = [_wrap_name(c) for c in class_names]

    fig = go.Figure()
    for (name, Y), color in zip(splits.items(), colors):
        fig.add_trace(
            go.Bar(
                name=name,
                x=labels,
                y=np.round(Y.mean(axis=0) * 100, 2),
                marker_color=color,
                marker=dict(line=dict(color="rgba(255,255,255,0.0)", width=0)),
                opacity=0.92,
            )
        )

    fig.update_layout(
        barmode="group",
        title=dict(
            text="Class Frequency Across Splits (% of Frames)",
            x=0.5,
            xanchor="center",
            y=0.97,
            yanchor="top",
            font=dict(size=18, color="#2C3E50", family="Inter, Arial, sans-serif"),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0)",
            font=dict(size=13, color="#2C3E50"),
        ),
        height=560,
        width=1200,
        margin=dict(t=90, b=120, l=70, r=40),
        plot_bgcolor="#F8F9FB",
        paper_bgcolor="#FFFFFF",
        font=dict(family="Inter, Arial, sans-serif", color="#2C3E50"),
    )
    fig.update_xaxes(
        title_text="Classes",
        title_font=dict(size=13, color="#5A6478"),
        tickangle=-35,
        tickfont=dict(size=11, color="#5A6478"),
        tickmode="array",
        tickvals=list(range(len(labels))),
        ticktext=labels,
        showgrid=False,
        showline=True,
        linecolor="#DDE1E9",
        linewidth=1,
    )
    fig.update_yaxes(
        title_text="% of Frames",
        title_font=dict(size=13, color="#5A6478"),
        tickfont=dict(size=11, color="#5A6478"),
        gridcolor="#EDF0F5",
        gridwidth=1,
        showline=False,
        zeroline=False,
    )
    fig.update_traces(cliponaxis=False)
    fig.show()

    if save:
        html_path = save_path.replace(".png", ".html")
        fig.write_html(html_path)
        print(f"Saved to {html_path}")


def visualize_knn_n_neighbours(
    experiment_log: str = "knn_experiment_log.csv",
    k_features_filter: Optional[int] = None,
    save_path: Optional[str] = "knn_n_neighbours_performance.png",
) -> None:
    """Plot avg validation balanced accuracy and macro F1 vs. n_neighbours."""
    df = pd.read_csv(experiment_log)

    if k_features_filter is not None:
        df = df[df["k_features"] == k_features_filter]
        if df.empty:
            raise ValueError(f"No runs found with k_features={k_features_filter}.")

    grouped = (
        df.groupby("n_neighbours")[["avg_val_balanced_accuracy", "avg_val_macro_f1"]]
        .mean()
        .reset_index()
        .sort_values("n_neighbours")
    )

    k_label = (
        f"k={k_features_filter}" if k_features_filter else f"k={df['k_features'].iloc[0]}"
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=grouped["n_neighbours"],
            y=grouped["avg_val_balanced_accuracy"],
            mode="lines+markers+text",
            name="Balanced Accuracy",
            marker=dict(size=10, symbol="circle"),
            line=dict(width=2.5),
            text=[f"{v:.4f}" for v in grouped["avg_val_balanced_accuracy"]],
            textposition="top center",
            textfont=dict(size=12),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grouped["n_neighbours"],
            y=grouped["avg_val_macro_f1"],
            mode="lines+markers+text",
            name="Macro F1",
            marker=dict(size=10, symbol="diamond"),
            line=dict(width=2.5),
            text=[f"{v:.4f}" for v in grouped["avg_val_macro_f1"]],
            textposition="bottom center",
            textfont=dict(size=12),
        )
    )

    fig.update_layout(
        title=dict(
            text=(
                f"KNN Validation Metrics vs n_neighbours ({k_label} features)<br>"
                "<span style='font-size: 15px; font-weight: normal;'>"
                "Avg balanced accuracy and macro F1 across all classes"
                "</span>"
            )
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="center",
            x=0.5,
        ),
    )
    fig.update_xaxes(
        title_text="n_neighbours",
        tickmode="array",
        tickvals=grouped["n_neighbours"].tolist(),
    )
    fig.update_yaxes(title_text="Score (0-1)", tickformat=".3f")
    fig.update_traces(cliponaxis=False)
    fig.show()

    if save_path:
        fig.write_image(save_path)
        print(f"Chart saved → {save_path}")
