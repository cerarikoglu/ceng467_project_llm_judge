import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def load_merged_data():
    classic = pd.read_csv("results/classic_metrics_results.csv")
    zero = pd.read_csv("results/llm_judge_zero_shot.csv")[["id","llm_score"]].rename(columns={"llm_score":"llama70b_zero"})
    cot = pd.read_csv("results/llm_judge_chain_of_thought.csv")[["id","llm_score"]].rename(columns={"llm_score":"llama70b_cot"})
    baseline = pd.read_csv("results/llm_baseline_zero_shot.csv")[["id","llm_score"]].rename(columns={"llm_score":"llama8b_zero"})
    df = classic.merge(zero, on="id").merge(cot, on="id").merge(baseline, on="id")
    return df

def find_disagreement_cases(df, top_n=5):
    """LLM yüksek puan vermiş ama BLEU düşük olan vakalar ve tersi."""
    
    # BLEU normalize et (0-1 arasına)
    df["bleu_norm"] = df["bleu"] / 100.0
    df["disagreement"] = abs(df["llama70b_zero"] / 5.0 - df["bleu_norm"])

    print("=" * 60)
    print("TOP DISAGREEMENT CASES (LLM vs BLEU)")
    print("=" * 60)
    top = df.nlargest(top_n, "disagreement")
    for _, row in top.iterrows():
        print(f"\nID: {int(row['id'])}")
        print(f"  HYP:  {row['hypothesis'][:100]}")
        print(f"  REF:  {row['reference'][:100]}")
        print(f"  BLEU: {row['bleu']:.1f} | LLM Score: {row['llama70b_zero']:.0f}/5 | BERTScore: {row['bertscore_f1']:.3f}")

    print("\n" + "=" * 60)
    print("LLM HIGH / BLEU LOW (LLM sees quality BLEU misses)")
    print("=" * 60)
    llm_high_bleu_low = df[(df["llama70b_zero"] >= 4) & (df["bleu"] < 30)]
    print(f"Count: {len(llm_high_bleu_low)}")
    for _, row in llm_high_bleu_low.head(3).iterrows():
        print(f"\n  HYP: {row['hypothesis'][:100]}")
        print(f"  REF: {row['reference'][:100]}")
        print(f"  BLEU: {row['bleu']:.1f} | LLM: {row['llama70b_zero']:.0f} | BERTScore: {row['bertscore_f1']:.3f}")

    print("\n" + "=" * 60)
    print("LLM LOW / BLEU HIGH (BLEU sees quality LLM misses)")
    print("=" * 60)
    llm_low_bleu_high = df[(df["llama70b_zero"] <= 2) & (df["bleu"] > 50)]
    print(f"Count: {len(llm_low_bleu_high)}")
    for _, row in llm_low_bleu_high.head(3).iterrows():
        print(f"\n  HYP: {row['hypothesis'][:100]}")
        print(f"  REF: {row['reference'][:100]}")
        print(f"  BLEU: {row['bleu']:.1f} | LLM: {row['llama70b_zero']:.0f} | BERTScore: {row['bertscore_f1']:.3f}")

    return llm_high_bleu_low, llm_low_bleu_high

def score_distribution_stats(df):
    print("\n" + "=" * 60)
    print("SCORE DISTRIBUTION ANALYSIS")
    print("=" * 60)
    for col, label in [
        ("llama8b_zero", "Llama 8B Zero-shot"),
        ("llama70b_zero", "Llama 70B Zero-shot"),
        ("llama70b_cot", "Llama 70B CoT"),
    ]:
        counts = df[col].value_counts().sort_index()
        print(f"\n{label}:")
        for score, count in counts.items():
            bar = "#" * count
            print(f"  {int(score)}: {bar} ({count})")

def plot_bleu_vs_llm(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    pairs = [
        ("bleu", "llama8b_zero", "BLEU vs Llama 8B Zero-shot"),
        ("bleu", "llama70b_zero", "BLEU vs Llama 70B Zero-shot"),
        ("bleu", "llama70b_cot", "BLEU vs Llama 70B CoT"),
    ]
    for ax, (x, y, title) in zip(axes, pairs):
        ax.scatter(df[x], df[y], alpha=0.4, color="steelblue", s=20)
        ax.set_xlabel("BLEU Score")
        ax.set_ylabel("LLM Score (1-5)")
        ax.set_title(title)
        ax.set_yticks([1,2,3,4,5])

    plt.tight_layout()
    plt.savefig("results/bleu_vs_llm.png", dpi=150)
    print("\nSaved -> results/bleu_vs_llm.png")
    plt.close()

if __name__ == "__main__":
    df = load_merged_data()
    find_disagreement_cases(df, top_n=5)
    score_distribution_stats(df)
    plot_bleu_vs_llm(df)