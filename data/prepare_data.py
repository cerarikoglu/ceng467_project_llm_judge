from datasets import load_dataset
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASET_NAME, LANGUAGE_PAIR, SAMPLE_SIZE

def load_wmt_data():
    dataset = load_dataset(DATASET_NAME, LANGUAGE_PAIR, split="validation")
    
    samples = []
    for i, item in enumerate(dataset):
        if i >= SAMPLE_SIZE:
            break
        samples.append({
            "source": item["translation"]["de"],
            "reference": item["translation"]["en"],
        })
    
    df = pd.DataFrame(samples)
    
    current_dir = os.path.dirname(os.path.abspath(__file__)) 
    project_root = os.path.dirname(current_dir)
    output_dir = os.path.join(project_root, "results")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "wmt_samples.csv")

    df.to_csv(output_path, index=False, encoding='utf-8')
    
if __name__ == "__main__":
    df = load_wmt_data()
    print(df.head())