"""
prompts.py — Single source of truth for all LLM prompt templates.
Both training (code_self_play.py) and evaluation (evaluate.py) import from here
to guarantee they always use the same format.
"""

SYSTEM_MSG = "You are a helpful AI programming assistant."

FIX_PROMPT = """\
You are an expert Python debugger. Here is a problem description, a suite of unit tests, and a buggy implementation.
Your task is to fix the bug so that all tests pass.
First, reason about the bug and how to fix it step-by-step inside a <think> block.
Then, output the corrected Python code block.

Problem: {problem}
Tests: {tests}

Buggy Code:
```python
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
