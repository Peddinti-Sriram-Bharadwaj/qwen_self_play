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
