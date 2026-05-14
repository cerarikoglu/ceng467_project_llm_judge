from transformers import MarianMTModel, MarianTokenizer
import pandas as pd
import os

def generate_translations():
    print("Downloading Helsinki-NLP/opus-mt-de-en model...")
    model_name = "Helsinki-NLP/opus-mt-de-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    df = pd.read_csv("results/wmt_samples.csv")
    hypotheses = []

    print(f"{len(df)} sentence being translated")
    for i, row in df.iterrows():
        inputs = tokenizer([row["source"]], return_tensors="pt", padding=True)
        translated = model.generate(**inputs)
        hypothesis = tokenizer.decode(translated[0], skip_special_tokens=True)
        hypotheses.append(hypothesis)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(df)} tamamlandı...")

    df["hypothesis"] = hypotheses
    df.to_csv("results/wmt_samples_with_hypotheses.csv", index=False)
    print("Recorded → results/wmt_samples_with_hypotheses.csv")
    return df

if __name__ == "__main__":
    df = generate_translations()
    print(df[["source", "hypothesis", "reference"]].head(3))