import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Model Configuration
JUDGE_MODEL_OPENAI = "gpt-4o"
JUDGE_MODEL_ANTHROPIC = "claude-3-5-sonnet-20241022"
BASELINE_MODEL = "gpt-3.5-turbo"
MAX_TOKENS = 512
TEMPERATURE = 0.0          # Deterministik sonuç için 0

# Dataset Configuration
DATASET_NAME = "wmt19"
LANGUAGE_PAIR = "de-en"
SAMPLE_SIZE = 200          # Test için 200 örnek yeterli

# Evaluation Configuration
METRICS = ["bleu", "rouge", "bertscore"]
CORRELATION_METHODS = ["pearson", "spearman"]