import pandas as pd
import sacrebleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def compute_bleu(hypothesis, reference):
    result = sacrebleu.sentence_bleu(hypothesis, [reference])
    return round(result.score, 4)

def compute_rouge(hypothesis, reference):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }

def run_classic_baselines(df):
    print("Computing classical metrics on MT hypotheses...")
    results = []

    hypotheses = df["hypothesis"].tolist()
    references = df["reference"].tolist()

    print("Computing BERTScore...")
    P, R, F1 = bert_score(hypotheses, references, lang="en", verbose=False)

    for i, row in df.iterrows():
        rouge = compute_rouge(row["hypothesis"], row["reference"])
        results.append({
            "id": i,
            "source": row["source"],
            "hypothesis": row["hypothesis"],
            "reference": row["reference"],
            "bleu": compute_bleu(row["hypothesis"], row["reference"]),
            "rouge1": rouge["rouge1"],
            "rouge2": rouge["rouge2"],
            "rougeL": rouge["rougeL"],
            "bertscore_f1": round(F1[i].item(), 4),
        })

    result_df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    result_df.to_csv("results/classic_metrics_results.csv", index=False)
    print("Saved -> results/classic_metrics_results.csv")
    print("\n--- Summary Statistics ---")
    print(result_df[["bleu", "rouge1", "rouge2", "rougeL", "bertscore_f1"]].describe().round(4))
    return result_df

if __name__ == "__main__":
    df = pd.read_csv("results/wmt_samples_with_hypotheses.csv")
    run_classic_baselines(df)