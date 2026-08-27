import subprocess
import tempfile
import os
import ast
from dataclasses import dataclass


@dataclass
class SandboxResult:
    """Typed result from evaluate_code(). Replaces the plain dict."""
    parseable: bool = False
    executable: bool = False
    all_passed: bool = False
    has_assertion_failure: bool = False
    timed_out: bool = False
    output: str = ""


def evaluate_code(code: str, test_cases: list, timeout: int = 3) -> SandboxResult:
    """
    Safely evaluates a Python code candidate against a set of unit tests in an isolated subprocess.
    Returns a SandboxResult with a 4-level failure taxonomy.
    """
    result = SandboxResult()

    # 1. Filter: Ensure the code is syntactically valid (parseable)
    try:
        ast.parse(code)
        result.parseable = True
    except SyntaxError as e:
        result.output = f"SyntaxError: {e}"
        return result

    # 2. Build the test harness
    # Append test cases directly — they're multi-line blocks (e.g. def check(...): ...)
    script = code + "\n\n"
    for test in test_cases:
        script += f"{test}\n"
    script += "print('ALL_TESTS_PASSED')\n"

    # 3. Execute in an isolated temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, "sandbox_run.py")
        with open(script_path, "w") as f:
            f.write(script)

        try:
            # (In a production server, prefix with unshare/firejail for full isolation)
            process = subprocess.run(
                ["python", script_path],
                cwd=tmpdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout
            )

            stdout = process.stdout
            stderr = process.stderr

            result.executable = True

            if process.returncode == 0 and "ALL_TESTS_PASSED" in stdout:
                result.all_passed = True
                result.output = stdout.strip()
            elif "AssertionError" in stderr:
                result.has_assertion_failure = True
                result.output = stderr.strip()
            else:
                # NameError, TypeError, RuntimeError, etc.
                result.output = (stderr if stderr else stdout).strip()

        except subprocess.TimeoutExpired:
            result.timed_out = True
            result.output = f"TimeoutExpired: Code execution exceeded {timeout} seconds."

    return result


# Simple validation check
if __name__ == "__main__":
    sample_code = "def multiply(a, b):\n    return a * b"
    sample_tests = ["assert multiply(2, 3) == 6", "assert multiply(0, 5) == 0"]

    print("Testing Valid Code:")
    print(evaluate_code(sample_code, sample_tests))

    print("\nTesting Buggy Code (Assertion Failure):")
    buggy_code = "def multiply(a, b):\n    return a + b"
    print(evaluate_code(buggy_code, sample_tests))

    print("\nTesting Syntax Error Code:")
    syntax_code = "def multiply(a, b) return a * b"
    print(evaluate_code(syntax_code, sample_tests))
