# Bu dosya hocanın "prompts must be documented" şartını karşılar.

SYSTEM_PROMPT = "You are an expert human evaluator for machine translation quality assessment."

ZERO_SHOT_PROMPT = """Rate the quality of the following English translation from German.

Source (German): {source}
Translation (English): {hypothesis}
Reference (English): {reference}

Score the translation on a scale of 1 to 5:
1 = Very poor (wrong meaning, unreadable)
2 = Poor (major errors)
3 = Acceptable (understandable but errors exist)
4 = Good (minor errors only)
5 = Excellent (fluent and accurate)

Respond with only a single integer (1-5)."""

FEW_SHOT_PROMPT = """You are rating machine translations. Here are two examples:

Example 1:
Source: Das Wetter ist heute sehr schön.
Translation: The weather is very nice today.
Score: 5

Example 2:
Source: Die Regierung hat neue Gesetze verabschiedet.
Translation: Government make new law yesterday maybe.
Score: 2

Now rate this translation:
Source (German): {source}
Translation (English): {hypothesis}
Reference (English): {reference}

Respond with only a single integer (1-5)."""

COT_PROMPT = """Evaluate the following machine translation.

Source (German): {source}
Translation (English): {hypothesis}
Reference (English): {reference}

Step 1 - Adequacy: Does the translation preserve the meaning of the source?
Step 2 - Fluency: Is the translation grammatically correct and natural?
Step 3 - Final Score: Give a score from 1 (very poor) to 5 (excellent).

Format your response as:
Adequacy: [your comment]
Fluency: [your comment]
Score: [1-5]"""

PROMPT_CONFIGS = {
    "zero_shot": {
        "template": ZERO_SHOT_PROMPT,
        "description": "Direct scoring without examples",
        "expected_output": "single integer 1-5"
    },
    "few_shot": {
        "template": FEW_SHOT_PROMPT,
        "description": "Two-shot prompting with examples",
        "expected_output": "single integer 1-5"
    },
    "chain_of_thought": {
        "template": COT_PROMPT,
        "description": "Step-by-step reasoning before scoring",
        "expected_output": "reasoning + Score: N"
    }
}