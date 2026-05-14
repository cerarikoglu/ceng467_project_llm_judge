import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Model Configuration
JUDGE_MODEL = "llama-3.3-70b-versatile"    # Ana sistem
BASELINE_MODEL = "llama-3.1-8b-instant"    # Baseline (kucuk/hizli model)
MAX_TOKENS = 512
TEMPERATURE = 0.0

# Dataset Configuration
DATASET_NAME = "wmt19"
LANGUAGE_PAIR = "de-en"
SAMPLE_SIZE = 200

# Evaluation Configuration
METRICS = ["bleu", "rouge", "bertscore"]
CORRELATION_METHODS = ["pearson", "spearman"]