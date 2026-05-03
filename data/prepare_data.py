from datasets import load_dataset
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASET_NAME, LANGUAGE_PAIR, SAMPLE_SIZE

def load_wmt_data():
  dataset = load_dataset(DATASET_NAME, LANGUAGE_PAIR, split="test")
  samples = []
  for i, item in enumerate(dataset):
    if i >= SAMPLE_SIZE:
      break
    samples.append({
            "source": item["translation"]["de"],
            "reference": item["translation"]["en"],
        })  
  df = pd.DataFrame(samples)
  os.makedirs("results", exist_ok=True)
  df.to_csv("results/wmt_samples.csv",index=False)
  print(f"{len(df)} örnek kaydedildi → results/wmt_samples.csv")
  return df
if __name__ == "__main__":
  df = load_wmt_data()
  print(df.head())