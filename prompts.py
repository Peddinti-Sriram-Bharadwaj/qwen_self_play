"""
prompts.py — Single source of truth for all LLM prompt templates.
Both training (code_self_play.py) and evaluation (evaluate.py) import from here
to guarantee they always use the same format.
"""

SYSTEM_MSG = "You are a helpful AI programming assistant."

FIX_PROMPT = """You are an expert Python developer. 
The following Python code contains a bug that causes it to fail the provided unit tests.
Identify the bug and return the fully corrected Python code.

Problem:
{problem}

Failing Tests:
{tests}

Buggy Code:
{buggy_code}
```
"""

GEN_PROMPT = """\
You are a code bug generator. Here is a correctly working function and its unit tests.
Your task is to introduce a subtle bug into this function such that it fails the unit tests, but remains syntactically valid Python.
First, reason about how to break the logic step-by-step inside a <think> block.
Then, output the buggy Python code block.

Problem: {problem}
Tests: {tests}

Correct Code:
```python
{code}
```
"""

# --- NLP Domain Conflict Prompts (Grammar Correction) ---

NLP_GEN_PROMPT = """You are an adversarial text generator.
Given the following perfectly grammatically correct English sentence, your task is to introduce a subtle grammatical error, typo, or awkward phrasing.
Do not completely change the meaning of the sentence, just introduce an error that needs fixing.

Original Sentence:
{correct_text}

Return only the corrupted sentence. Do not provide explanations or formatting.
"""

NLP_FIX_PROMPT = """You are an expert English copyeditor.
The following sentence contains a grammatical error, typo, or awkward phrasing.
Fix the error and return the perfectly corrected sentence.

Buggy Sentence:
{corrupted_text}

Return only the corrected sentence. Do not provide explanations or formatting.
"""
