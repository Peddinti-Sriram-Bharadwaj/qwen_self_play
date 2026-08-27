"""
parsing.py — Utilities for post-processing raw LLM text output into executable code.
Decoupled from the model so it can be tested independently without GPU.
"""

import re


def extract_python_code(text: str) -> str:
    """
    Extracts the first python code block from a generation.
    Falls back to the raw text if no markdown fenced block is found.
    Also strips hallucinated check() calls that Qwen memorizes from HumanEval,
    which cause NameError crashes when the sandbox appends the real test suite.
    """
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    code = match.group(1).strip() if match else text.strip()

    # Remove top-level check() calls hallucinated by the model
    code = re.sub(r"(?m)^check\s*\(.*?\)", "", code)
    return code
