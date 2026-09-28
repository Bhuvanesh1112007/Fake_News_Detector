"""
Linguistic Feature Visualization Module (Phase 9)
Generates exploratory distribution plots and Fake vs. Real comparative visualizations
for POS tag distributions and rule-based linguistic anomaly scores.

Saved to: results/grammar/
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Headless rendering
import matplotlib.pyplot as plt
import seaborn as sns


def generate_linguistic_visualizations(
    feature_csv_path: str = "data/processed/linguistic_features.csv",
    output_dir: str = "results/grammar"
):
    """
    Loads extracted linguistic features and generates publication-grade exploratory figures.

    Args:
        feature_csv_path: Path to linguistic_features.csv.
        output_dir: Output directory for plot images.
    """
    if not os.path.exists(feature_csv_path):
        raise FileNotFoundError(f"Feature file not found at: {feature_csv_path}")

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(feature_csv_path)

    # Set overall aesthetic styling
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11
    })

    palette = {"REAL": "#2ecc71", "FAKE": "#e74c3c"}

    # 1. Distribution of noun_count
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x="noun_count", hue="label", palette=palette, kde=True, bins=10, ax=ax, alpha=0.6)
    ax.set_title("Exploratory Distribution of Noun Count (Fake vs. Real)")
    ax.set_xlabel("Noun Count (spaCy NOUN)")
    ax.set_ylabel("Document Count")
    plt.tight_layout()
    noun_plot_path = os.path.join(output_dir, "distribution_noun_count.png")
    fig.savefig(noun_plot_path, dpi=300)
    plt.close(fig)
    print(f"[Phase 9] Generated: {noun_plot_path}")

    # 2. Distribution of verb_count
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x="verb_count", hue="label", palette=palette, kde=True, bins=10, ax=ax, alpha=0.6)
    ax.set_title("Exploratory Distribution of Verb Count (Fake vs. Real)")
    ax.set_xlabel("Verb Count (spaCy VERB)")
    ax.set_ylabel("Document Count")
    plt.tight_layout()
    verb_plot_path = os.path.join(output_dir, "distribution_verb_count.png")
    fig.savefig(verb_plot_path, dpi=300)
    plt.close(fig)
    print(f"[Phase 9] Generated: {verb_plot_path}")

    # 3. Distribution of adjective_count
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x="adjective_count", hue="label", palette=palette, kde=True, bins=10, ax=ax, alpha=0.6)
    ax.set_title("Exploratory Distribution of Adjective Count (Fake vs. Real)")
    ax.set_xlabel("Adjective Count (spaCy ADJ)")
    ax.set_ylabel("Document Count")
    plt.tight_layout()
    adj_plot_path = os.path.join(output_dir, "distribution_adjective_count.png")
    fig.savefig(adj_plot_path, dpi=300)
    plt.close(fig)
    print(f"[Phase 9] Generated: {adj_plot_path}")

    # 4. Distribution of anomaly_score
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x="anomaly_score", hue="label", palette=palette, kde=True, bins=10, ax=ax, alpha=0.6)
    ax.set_title("Exploratory Distribution of Rule-Based Anomaly Score")
    ax.set_xlabel("Anomaly Score [0.0 - 1.0]")
    ax.set_ylabel("Document Count")
    plt.tight_layout()
    anomaly_plot_path = os.path.join(output_dir, "distribution_anomaly_score.png")
    fig.savefig(anomaly_plot_path, dpi=300)
    plt.close(fig)
    print(f"[Phase 9] Generated: {anomaly_plot_path}")

    # 5. Combined Multi-Panel Exploratory Comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # (a) Noun vs Verb Ratio
    sns.boxplot(data=df, x="label", y="noun_count", hue="label", palette=palette, legend=False, ax=axes[0, 0], width=0.4)
    axes[0, 0].set_title("(a) Noun Count Distribution by Label")
    axes[0, 0].set_xlabel("Article Category")
    axes[0, 0].set_ylabel("Noun Count")

    # (b) Adjective Count by Label
    sns.boxplot(data=df, x="label", y="adjective_count", hue="label", palette=palette, legend=False, ax=axes[0, 1], width=0.4)
    axes[0, 1].set_title("(b) Adjective Count Distribution by Label")
    axes[0, 1].set_xlabel("Article Category")
    axes[0, 1].set_ylabel("Adjective Count")

    # (c) Exclamation Count by Label
    sns.boxplot(data=df, x="label", y="exclamation_count", hue="label", palette=palette, legend=False, ax=axes[1, 0], width=0.4)
    axes[1, 0].set_title("(c) Exclamation Count Distribution by Label")
    axes[1, 0].set_xlabel("Article Category")
    axes[1, 0].set_ylabel("Exclamation Marks (!)")

    # (d) Anomaly Score Boxplot
    sns.boxplot(data=df, x="label", y="anomaly_score", hue="label", palette=palette, legend=False, ax=axes[1, 1], width=0.4)
    axes[1, 1].set_title("(d) Anomaly Score Distribution by Label")
    axes[1, 1].set_xlabel("Article Category")
    axes[1, 1].set_ylabel("Anomaly Score [0.0 - 1.0]")

    plt.suptitle("Exploratory Comparison of Linguistic & Anomaly Features (Fake vs. Real)", fontsize=14, y=0.99)
    plt.tight_layout()
    comparison_plot_path = os.path.join(output_dir, "linguistic_features_comparison_fake_vs_real.png")
    fig.savefig(comparison_plot_path, dpi=300)
    plt.close(fig)
    print(f"[Phase 9] Generated: {comparison_plot_path}")

    print("[Phase 9] All 5 visualization figures saved successfully.")


if __name__ == "__main__":
    generate_linguistic_visualizations()
