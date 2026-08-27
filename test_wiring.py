"""
test_wiring.py — Integration wiring test for the self-play pipeline.

Tests the full pipeline (prompts → parsing → sandbox → rewards → eval loop)
WITHOUT loading a real LLM. The agent is replaced by a MockAgent that returns
pre-written code strings, so this test runs in seconds on any machine.

Run with: python test_wiring.py
"""

import json
import os
import sys
import tempfile
import traceback

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"

results = []

def check(name: str, cond: bool, detail: str = ""):
    status = PASS if cond else FAIL
    print(f"  {status}  {name}" + (f" — {detail}" if detail else ""))
    results.append((name, cond))


# ---------------------------------------------------------------------------
# Mock Agent (no real model, no GPU)
# ---------------------------------------------------------------------------

class MockTokenizer:
    pad_token = "<pad>"
    pad_token_id = 0
    padding_side = "left"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        """Returns a flat string representation of the chat messages."""
        return " | ".join(f"{m['role']}: {m['content'][:40]}" for m in messages)


class MockAgent:
    """
    A drop-in replacement for DualLoraCodeAgent that never loads a model.
    Responses are configured per-test via `set_response(...)`.
    """
    device = "cpu"
    tokenizer = MockTokenizer()
    _response = "```python\npass\n```"

    def set_response(self, text: str):
        self._response = text

    def set_active_role(self, role: str):
        pass  # No-op

    def batched_generate(self, prompts: list, **kwargs) -> list:
        return [self._response] * len(prompts)


# ---------------------------------------------------------------------------
# Test 1: parsing.py
# ---------------------------------------------------------------------------

print("\n=== Test 1: parsing.py ===")
try:
    from parsing import extract_python_code

    # 1a. Extracts code from a fenced block
    text_with_fence = "Some explanation.\n```python\ndef foo():\n    return 1\n```\nDone."
    code = extract_python_code(text_with_fence)
    check("extracts fenced python block", "def foo():" in code)

    # 1b. Falls back to raw text when no fence
    text_no_fence = "def bar():\n    return 2"
    code = extract_python_code(text_no_fence)
    check("falls back to raw text", "def bar():" in code)

    # 1c. Strips hallucinated check() calls
    text_with_check = "```python\ndef f():\n    return 1\ncheck(f)\n```"
    code = extract_python_code(text_with_check)
    check("strips hallucinated check() calls", "check(f)" not in code, repr(code))

except Exception:
    print(f"  {FAIL}  parsing module import or test crashed:")
    traceback.print_exc()


# ---------------------------------------------------------------------------
# Test 2: sandbox.py — SandboxResult dataclass
# ---------------------------------------------------------------------------

print("\n=== Test 2: sandbox.py + SandboxResult ===")
try:
    from sandbox import evaluate_code, SandboxResult

    # 2a. Correct code → all_passed
    good_code = "def add(a, b):\n    return a + b"
    res = evaluate_code(good_code, ["assert add(1, 2) == 3", "assert add(0, 0) == 0"])
    check("correct code → all_passed", res.all_passed)
    check("result is SandboxResult", isinstance(res, SandboxResult))

    # 2b. Buggy code → has_assertion_failure
    buggy_code = "def add(a, b):\n    return a - b"
    res = evaluate_code(buggy_code, ["assert add(1, 2) == 3"])
    check("buggy code → has_assertion_failure", res.has_assertion_failure)
    check("buggy code → not all_passed", not res.all_passed)

    # 2c. Syntax error → parseable=False, early exit
    syntax_code = "def add(a, b) return a + b"
    res = evaluate_code(syntax_code, ["assert add(1,2)==3"])
    check("syntax error → not parseable", not res.parseable)
    check("syntax error → not executable", not res.executable)

    # 2d. Multiline humanevalpack-style test block
    hep_test = """


def check(add):
    assert add(1, 2) == 3
    assert add(0, 0) == 0

check(add)"""
    res = evaluate_code(good_code, [hep_test])
    check("multiline check() test block passes", res.all_passed)

except Exception:
    print(f"  {FAIL}  sandbox module crashed:")
    traceback.print_exc()


# ---------------------------------------------------------------------------
# Test 3: rewards.py
# ---------------------------------------------------------------------------

print("\n=== Test 3: rewards.py ===")
try:
    from sandbox import SandboxResult
    from rewards import fixer_reward, generator_reward

    check("all_passed → reward 1.0",    fixer_reward(SandboxResult(all_passed=True)) == 1.0)
    check("assertion_failure → reward 0.0", fixer_reward(SandboxResult(has_assertion_failure=True)) == 0.0)
    check("timed_out → reward -0.5",    fixer_reward(SandboxResult(timed_out=True)) == -0.5)
    check("syntax crash → reward -1.0", fixer_reward(SandboxResult()) == -1.0)

    check("fix_rate 0.5 → gen reward 1.0",  generator_reward(0.5) == 1.0)
    check("fix_rate 0.0 → gen reward -0.5", generator_reward(0.0) == -0.5)
    check("fix_rate 1.0 → gen reward -0.5", generator_reward(1.0) == -0.5)

except Exception:
    print(f"  {FAIL}  rewards module crashed:")
    traceback.print_exc()


# ---------------------------------------------------------------------------
# Test 4: prompts.py
# ---------------------------------------------------------------------------

print("\n=== Test 4: prompts.py ===")
try:
    from prompts import FIX_PROMPT, GEN_PROMPT, SYSTEM_MSG

    fix = FIX_PROMPT.format(problem="Add two numbers", tests="assert add(1,2)==3", buggy_code="def add(a,b): return a-b")
    gen = GEN_PROMPT.format(problem="Add two numbers", tests="assert add(1,2)==3", code="def add(a,b): return a+b")

    check("FIX_PROMPT formats correctly",  "{" not in fix and "buggy_code" not in fix)
    check("GEN_PROMPT formats correctly",  "{" not in gen and "{code}" not in gen)
    check("SYSTEM_MSG is non-empty",       len(SYSTEM_MSG) > 0)

except Exception:
    print(f"  {FAIL}  prompts module crashed:")
    traceback.print_exc()


# ---------------------------------------------------------------------------
# Test 5: evaluate_checkpoint with MockAgent
# ---------------------------------------------------------------------------

print("\n=== Test 5: evaluate.py with MockAgent (no real LLM) ===")
try:
    from evaluate import evaluate_checkpoint

    # Write a tiny synthetic eval dataset to a temp file
    tasks = [
        {
            "task_id": "mock/0",
            "problem": "Write a function add(a, b) that returns a+b.",
            "buggy_code": "def add(a, b):\n    return a - b",  # intentional bug
            "tests": ["assert add(1, 2) == 3", "assert add(0, 0) == 0"]
        },
        {
            "task_id": "mock/1",
            "problem": "Write a function mul(a, b) that returns a*b.",
            "buggy_code": "def mul(a, b):\n    return a + b",  # intentional bug
            "tests": ["assert mul(2, 3) == 6"]
        }
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(tasks, f)
        tmp_path = f.name

    agent = MockAgent()

    # 5a: Mock agent returns a correct fix → expect 100% pass rate
    agent.set_response("```python\ndef add(a, b):\n    return a + b\n\ndef mul(a, b):\n    return a * b\n```")
    rate = evaluate_checkpoint("base", tmp_path, agent=agent)
    check("mock correct fix → pass@1 = 1.0", rate == 1.0, f"got {rate}")

    # 5b: Mock agent returns a broken fix → expect 0% pass rate
    agent.set_response("```python\ndef add(a, b):\n    return a - b\n\ndef mul(a, b):\n    return a - b\n```")
    rate = evaluate_checkpoint("base", tmp_path, agent=agent)
    check("mock broken fix → pass@1 = 0.0", rate == 0.0, f"got {rate}")

    os.unlink(tmp_path)

except Exception:
    print(f"  {FAIL}  evaluate_checkpoint wiring test crashed:")
    traceback.print_exc()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n" + "="*50)
total = len(results)
passed = sum(1 for _, ok in results if ok)
failed = total - passed

print(f"Results: {passed}/{total} passed", end="")
if failed:
    print(f"  ({failed} FAILED)")
    for name, ok in results:
        if not ok:
            print(f"  ✗ {name}")
else:
    print("  — all green!")

sys.exit(0 if failed == 0 else 1)
