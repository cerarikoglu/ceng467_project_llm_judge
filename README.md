# CENG 467 — LLM-as-a-Judge for Evaluating Generation

## Overview
An evaluation pipeline where a strong LLM (GPT-4o) assesses the quality 
of machine translation outputs and is compared against classical metrics.

## Setup
```bash
git clone https://github.com/cerarikoglu/ceng467_project_llm_judge
cd ceng467_project_llm_judge
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
```

## API Keys
Create a `.env` file in the root:
GROQ_API_KEY=your_key_here

## Reproduce Results
```bash
python data/prepare_data.py          # 1. Download dataset
python baselines/classic_metrics.py  # 2. Run BLEU/ROUGE/BERTScore
python judge/llm_judge.py            # 3. Run LLM Judge (zero-shot)
```

## Prompt Configuration
All prompts and model settings are documented in:
- `judge/prompts.py` — all prompt templates
- `config.py` — model names, temperature, max_tokens