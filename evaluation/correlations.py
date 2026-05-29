import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_all_results():
    classic = pd.read_csv("results/classic_metrics_results.csv")
    baseline = pd.read_csv("results/llm_baseline_zero_shot.csv")[["id","llm_score"]].rename(columns={"llm_score":"llama8b_zero"})
    zero = pd.read_csv("results/llm_judge_zero_shot.csv")[["id","llm_score"]].rename(columns={"llm_score":"llama70b_zero"})
    few = pd.read_csv("results/llm_judge_few_shot.csv")[["id","llm_score"]].rename(columns={"llm_score":"llama70b_few"})
    cot = pd.read_csv("results/llm_judge_chain_of_thought.csv")[["id","llm_score"]].rename(columns={"llm_score":"llama70b_cot"})

    df = classic.merge(baseline, on="id").merge(zero, on="id").merge(few, on="id").merge(cot, on="id")
    return df

def compute_correlations(df):
    llm_cols = ["llama8b_zero", "llama70b_zero", "llama70b_few", "llama70b_cot"]
    metric_cols = ["bleu", "rouge1", "rouge2", "rougeL", "bertscore_f1"]

    print("=" * 60)
    print("PEARSON CORRELATION (LLM Score vs Classical Metrics)")
    print("=" * 60)
    pearson_results = []
    for llm in llm_cols:
        row = {"model": llm}
        for metric in metric_cols:
            r, p = stats.pearsonr(df[llm].dropna(), df[metric][df[llm].notna()])
            row[metric] = round(r, 4)
        pearson_results.append(row)
    pearson_df = pd.DataFrame(pearson_results).set_index("model")
    print(pearson_df)

    print("\n" + "=" * 60)
    print("SPEARMAN CORRELATION (LLM Score vs Classical Metrics)")
    print("=" * 60)
    spearman_results = []
    for llm in llm_cols:
        row = {"model": llm}
        for metric in metric_cols:
            r, p = stats.spearmanr(df[llm].dropna(), df[metric][df[llm].notna()])
            row[metric] = round(r, 4)
        spearman_results.append(row)
    spearman_df = pd.DataFrame(spearman_results).set_index("model")
    print(spearman_df)

    pearson_df.to_csv("results/pearson_correlations.csv")
    spearman_df.to_csv("results/spearman_correlations.csv")
    print("\nSaved -> results/pearson_correlations.csv")
    print("Saved -> results/spearman_correlations.csv")
    return pearson_df, spearman_df

def plot_score_distributions(df):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    cols = ["llama8b_zero", "llama70b_zero", "llama70b_few", "llama70b_cot"]
    titles = ["Llama 8B\nZero-shot", "Llama 70B\nZero-shot", "Llama 70B\nFew-shot", "Llama 70B\nCoT"]

    for ax, col, title in zip(axes, cols, titles):
        ax.hist(df[col].dropna(), bins=5, range=(1,6), edgecolor="black", color="steelblue")
        ax.set_title(title)
        ax.set_xlabel("Score (1-5)")
        ax.set_ylabel("Count")
        ax.set_xticks([1,2,3,4,5])

    plt.tight_layout()
    plt.savefig("results/score_distributions.png", dpi=150)
    print("Saved -> results/score_distributions.png")
    plt.close()

def plot_correlation_heatmap(pearson_df):
    plt.figure(figsize=(8, 4))
    sns.heatmap(pearson_df.astype(float), annot=True, fmt=".2f", cmap="RdYlGn",
                vmin=-1, vmax=1, linewidths=0.5)
    plt.title("Pearson Correlation: LLM Scores vs Classical Metrics")
    plt.tight_layout()
    plt.savefig("results/correlation_heatmap.png", dpi=150)
    print("Saved -> results/correlation_heatmap.png")
    plt.close()

if __name__ == "__main__":
    df = load_all_results()
    print(f"Merged dataset: {len(df)} samples\n")
    pearson_df, spearman_df = compute_correlations(df)
    plot_score_distributions(df)
    plot_correlation_heatmap(pearson_df)