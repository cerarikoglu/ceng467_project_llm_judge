import openai
import anthropic
import pandas as pd
import re
import time
import sys
sys.path.append("..")
from config import OPENAI_API_KEY, JUDGE_MODEL_OPENAI, MAX_TOKENS, TEMPERATURE
from judge.prompts import SYSTEM_PROMPT, PROMPT_CONFIGS

openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

def parse_score(response_text):
    """1-5 arası skoru yanıttan çıkar."""
    match = re.search(r"Score:\s*([1-5])|^([1-5])$", response_text.strip(), re.MULTILINE)
    if match:
        return int(match.group(1) or match.group(2))
    numbers = re.findall(r"[1-5]", response_text)
    return int(numbers[-1]) if numbers else None

def judge_with_openai(source, hypothesis, reference, prompt_type="zero_shot"):
    template = PROMPT_CONFIGS[prompt_type]["template"]
    prompt = template.format(source=source, hypothesis=hypothesis, reference=reference)
    
    response = openai_client.chat.completions.create(
        model=JUDGE_MODEL_OPENAI,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    raw = response.choices[0].message.content
    return parse_score(raw), raw

def run_llm_judge(df, prompt_type="zero_shot", max_samples=50):
    """Week 11 için 50 örnek yeterli, maliyet kontrolü için."""
    print(f"LLM Judge çalışıyor: {prompt_type} | {max_samples} örnek")
    results = []
    
    for i, row in df.head(max_samples).iterrows():
        try:
            score, raw_response = judge_with_openai(
                row["source"], row["reference"], row["reference"], prompt_type
            )
            results.append({
                "id": i,
                "source": row["source"],
                "reference": row["reference"],
                "prompt_type": prompt_type,
                "llm_score": score,
                "raw_response": raw_response,
            })
            if i % 10 == 0:
                print(f"  {i}/{max_samples} tamamlandı...")
            time.sleep(0.5)   # Rate limit için
        except Exception as e:
            print(f"Hata örnek {i}: {e}")
            continue
    
    result_df = pd.DataFrame(results)
    result_df.to_csv(f"results/llm_judge_{prompt_type}.csv", index=False)
    print(f"Kaydedildi → results/llm_judge_{prompt_type}.csv")
    return result_df

if __name__ == "__main__":
    df = pd.read_csv("results/wmt_samples.csv")
    run_llm_judge(df, prompt_type="zero_shot", max_samples=50)