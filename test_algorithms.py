import torch
from agent import LocalLLMAgent
from trainers.reinforce_strategy import ReinforceStrategy
from trainers.reinforce_plus_strategy import ReinforcePlusStrategy
from trainers.dapo_strategy import DAPOStrategy
from trainers.grpo_strategy import GRPOStrategy
from trainers.ppo_strategy import PPOStrategy

def create_mock_batch(agent, batch_size=2, seq_len=10):
    """Creates a mock batch of data that collect_data() would normally output."""
    batch = []
    for _ in range(batch_size):
        # Create mock tokenized queries and responses
        query = torch.randint(0, agent.tokenizer.vocab_size, (seq_len,))
        response = torch.randint(0, agent.tokenizer.vocab_size, (seq_len,))
        reward = torch.tensor(1.0) # Mock reward
        
        step_data = {
            'query': query,
            'response': response,
            'reward': reward
        }
        batch.append(step_data)
    return batch

def test_algorithm(algo_name, strategy_class, agent, batch):
    print(f"\n==============================================")
    print(f" Testing Algorithm: {algo_name}")
    print(f"==============================================")
    
    try:
        strategy = strategy_class(agent)
        # We only test the update loop (the backprop and shape compatibility)
        # as collect_data just loops through the environment.
        stats = strategy.update(batch)
        print(f"✓ {algo_name} update() completed successfully.")
        print(f"  Stats: {stats}")
        return True
    except Exception as e:
        print(f"✗ {algo_name} update() failed!")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("Initializing mock agent for testing shapes...")
    # Using a tiny model to make local testing fast and avoid OOMs
    try:
        agent = LocalLLMAgent(model_name="sshleifer/tiny-gpt2", device="cpu")
    except Exception as e:
        print("Falling back to gpt2...")
        agent = LocalLLMAgent(model_name="gpt2", device="cpu")
        
    batch = create_mock_batch(agent, batch_size=256, seq_len=15)
    
    algorithms = {
        "REINFORCE": ReinforceStrategy,
        "REINFORCE++": ReinforcePlusStrategy,
        "DAPO": DAPOStrategy,
        "GRPO": GRPOStrategy,
        "PPO": PPOStrategy
    }
    
    all_passed = True
    for name, cls in algorithms.items():
        passed = test_algorithm(name, cls, agent, batch)
        if not passed:
            all_passed = False
            
    print("\n==============================================")
    if all_passed:
        print("All Algorithms passed shape and backprop tests!")
    else:
        print("Some Algorithms failed. Check logs above.")
        
if __name__ == "__main__":
    main()
