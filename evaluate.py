import json
import argparse
import os

from code_agent import SelfPlayCodeAgent
from parsing import extract_python_code
from sandbox import evaluate_code
from prompts import FIX_PROMPT, SYSTEM_MSG


def evaluate_checkpoint(checkpoint_path, dataset_path, agent=None, model_name=None):
    """
    Evaluates a checkpoint (or the base model) on a dataset and returns pass@1.

    Args:
        checkpoint_path: Path to LoRA checkpoint, full model checkpoint, or "base".
        dataset_path:    Path to a JSON eval dataset (e.g., data/eval_A.json).
        agent:           An already-initialized SelfPlayCodeAgent (preferred).
                         If None, a new agent is created.
        model_name:      Required if agent is None.
    """
    print(f"\nEvaluating Checkpoint: {checkpoint_path}")
    print(f"Dataset: {dataset_path}")

    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    if agent is None:
        if model_name is None:
            raise ValueError("Must provide either 'agent' or 'model_name'")
            
        if checkpoint_path == "base":
            agent = SelfPlayCodeAgent(model_name=model_name, use_lora=True)
        else:
            # Determine if the checkpoint is a PEFT adapter or a Full Model
            is_peft = os.path.exists(os.path.join(checkpoint_path, "adapter_model.safetensors")) or \
                      os.path.exists(os.path.join(checkpoint_path, "adapter_model.bin"))
                      
            if is_peft:
                print(f"Loading LoRA adapter from {checkpoint_path}...")
                agent = SelfPlayCodeAgent(model_name=model_name, use_lora=True)
                agent.model.load_adapter(checkpoint_path, adapter_name="self_play")
            else:
                print(f"Loading Full Model from {checkpoint_path}...")
                # Pass the checkpoint_path as the base model_name, and disable LoRA
                agent = SelfPlayCodeAgent(model_name=checkpoint_path, use_lora=False)

    passed = 0
    total = len(dataset)

    for i, task in enumerate(dataset):
        tests_str = "\n".join(task["tests"])
        raw_prompt = FIX_PROMPT.format(
            problem=task["problem"],
            tests=tests_str,
            buggy_code=task["buggy_code"]
        )

        messages = [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": raw_prompt}
        ]
        prompt = agent.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # pass@1 strict evaluation (temperature ≈ 0 for near-greedy decoding)
        raw_fixes = agent.batched_generate([prompt], max_tokens=512, temperature=0.01)
        parsed_fix = extract_python_code(raw_fixes[0])

        res = evaluate_code(parsed_fix, task["tests"])

        if res.all_passed:
            passed += 1
        else:
            print(f"  [Failure Reason] {res.output}")

        print(f"[{i+1}/{total}] Task {task['task_id']} | Passed: {res.all_passed} | Current pass@1: {(passed/(i+1))*100:.2f}%")

    final_pass_rate = passed / total
    print(f"\nFINAL pass@1 for {dataset_path}: {final_pass_rate*100:.2f}%")
    return final_pass_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="base", help="Path to checkpoint (or 'base')")
    parser.add_argument("--dataset", type=str, required=True, help="Path to evaluation dataset")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct", help="Base model architecture")
    args = parser.parse_args()

    evaluate_checkpoint(args.checkpoint, args.dataset, model_name=args.model_name)
