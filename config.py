import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Model Configuration
JUDGE_MODEL = "gemini-1.5-pro"
BASELINE_MODEL = "gemini-1.5-flash"   # Daha ucuz/hızlı, baseline için
MAX_TOKENS = 512
TEMPERATURE = 0.0

# Dataset Configuration
DATASET_NAME = "wmt19"
LANGUAGE_PAIR = "de-en"
SAMPLE_SIZE = 200

# Evaluation Configuration
METRICS = ["bleu", "rouge", "bertscore"]
CORRELATION_METHODS = ["pearson", "spearman"]