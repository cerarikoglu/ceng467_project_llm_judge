from groq import Groq
import pandas as pd
import re
import time
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GROQ_API_KEY, BASELINE_MODEL, JUDGE_MODEL
from judge.prompts import SYSTEM_PROMPT, PROMPT_CONFIGS

client = Groq(api_key=GROQ_API_KEY)

def parse_score(response_text):
    match = re.search(r"Score:\s*([1-5])", response_text)
    if match:
        return int(match.group(1))
    numbers = re.findall(r"[1-5]", response_text)
    return int(numbers[-1]) if numbers else None

def judge_single(source, hypothesis, reference, prompt_type="zero_shot", use_baseline=False):
    config = PROMPT_CONFIGS[prompt_type]
    template = config["template"]
    prompt = template.format(source=source, hypothesis=hypothesis, reference=reference)
    model_name = BASELINE_MODEL if use_baseline else JUDGE_MODEL

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
    )
    raw = response.choices[0].message.content
    return parse_score(raw), raw

def run_llm_judge(df, prompt_type="zero_shot", use_baseline=False, max_samples=50):
    model_label = f"{BASELINE_MODEL} (baseline)" if use_baseline else f"{JUDGE_MODEL} (judge)"
    print(f"LLM Judge is working: {prompt_type} | {model_label} | {max_samples} sample")
    results = []

    for i, row in df.head(max_samples).iterrows():
        try:
            score, raw_response = judge_single(
                row["source"],
                row["reference"],
                row["reference"],
                prompt_type,
                use_baseline,
            )
            results.append({
                "id": i,
                "source": row["source"],
                "reference": row["reference"],
                "prompt_type": prompt_type,
                "model": BASELINE_MODEL if use_baseline else JUDGE_MODEL,
                "llm_score": score,
                "raw_response": raw_response,
            })
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{max_samples} tamamlandı...")
            time.sleep(0.5)
        except Exception as e:
            print(f" errored example {i}: {e}")
            continue

    result_df = pd.DataFrame(results)
    label = "baseline" if use_baseline else "judge"
    filename = f"results/llm_{label}_{prompt_type}.csv"
    result_df.to_csv(filename, index=False)
    print(f"recorded → {filename}")
    if not result_df.empty:
        print(f"Average Score: {result_df['llm_score'].mean():.2f}")
    return result_df


if __name__ == "__main__":
    df = pd.read_csv("results/wmt_samples.csv")
    
    # Baseline: Llama 8B, zero-shot
    run_llm_judge(df, prompt_type="zero_shot", use_baseline=True, max_samples=50)
    
    # Judge: Llama 70B, zero-shot
    run_llm_judge(df, prompt_type="zero_shot", use_baseline=False, max_samples=50)
    
    # Judge: Llama 70B, few-shot
    run_llm_judge(df, prompt_type="few_shot", use_baseline=False, max_samples=50)
    
    # Judge: Llama 70B, chain-of-thought
    run_llm_judge(df, prompt_type="chain_of_thought", use_baseline=False, max_samples=50)