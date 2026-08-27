"""
rewards.py — Reward shaping functions for the self-play RL loop.

Isolated here so reward logic can be extended (e.g., shaped rewards, curriculum)
without modifying the core GRPO training loop in code_self_play.py.
Open/Closed: open for extension, closed for modification of the loop.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sandbox import SandboxResult


def fixer_reward(res: "SandboxResult") -> float:
    """
    Maps a SandboxResult to a scalar reward for the fixer adapter.

    Failure taxonomy (best → worst):
      all_passed          →  1.0   (perfect fix)
      has_assertion_failure → 0.0  (ran OK, logic wrong — neutral, can tune)
      timed_out           → -0.5   (infinite loop, worse than wrong logic)
      SyntaxError / crash → -1.0   (code didn't even execute)
    """
    if res.all_passed:
        return 1.0
    if res.has_assertion_failure:
        return 0.0
    if res.timed_out:
        return -0.5
    return -1.0  # SyntaxError, NameError, RuntimeError before tests ran


def generator_reward(fix_rate: float) -> float:
    """
    Goldilocks reward for the generator adapter.
    The generator is rewarded for producing bugs at the right difficulty:
    hard enough that the fixer sometimes fails, easy enough that it sometimes passes.
    A fix_rate of 0.0 or 1.0 means the bug is either impossible or trivial.
    """
    if 0.25 <= fix_rate <= 0.75:
        return 1.0
    return -0.5


# --- NLP Domain Conflict Rewards ---

def nlp_fixer_reward(fixed_text: str, ground_truth: str) -> float:
    """
    Reward for the Fixer in the NLP grammar correction task.
    """
    import difflib
    fixed_text = fixed_text.strip()
    ground_truth = ground_truth.strip()
    
    if fixed_text == ground_truth:
        return 1.0
        
    similarity = difflib.SequenceMatcher(None, fixed_text, ground_truth).ratio()
    if similarity > 0.95:
        return 0.5
    elif similarity > 0.8:
        return 0.0
    return -1.0

def nlp_generator_reward(corrupted_text: str, ground_truth: str, fixer_success_rate: float) -> float:
    """
    Reward for the Generator. Must corrupt the text but keep it somewhat similar.
    """
    import difflib
    corrupted_text = corrupted_text.strip()
    ground_truth = ground_truth.strip()
    
    similarity = difflib.SequenceMatcher(None, corrupted_text, ground_truth).ratio()
    
    # If the generator completely destroyed the text or outputted nothing
    if similarity < 0.5 or similarity == 1.0:
        return -1.0
        
    # Standard zero-sum
    if fixer_success_rate < 0.2:
        return 1.0
    elif fixer_success_rate < 0.5:
        return 0.5
    return -0.5
