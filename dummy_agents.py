import random
import re
import torch
from agent import Agent

class DummyAgent(Agent):
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        
    def act(self, observation: str):
        actions, q, r, c = self.batched_act([observation])
        return actions[0], q[0], r[0], c[0]
        
    def extract_legal_actions(self, obs: str) -> list[str]:
        match = re.search(r"Legal Actions:\s*(.*)", obs)
        if match:
            actions_str = match.group(1).strip()
            return [a.strip() for a in actions_str.split(',')]
        return ["pass", "bet"] # Fallback

class RandomAgent(DummyAgent):
    def batched_act(self, observations: list[str]):
        actions = []
        for obs in observations:
            legal_actions = self.extract_legal_actions(obs)
            action = self.rng.choice(legal_actions)
            actions.append(f"ACTION: {action}")
            
        n = len(observations)
        return actions, [torch.empty(0)]*n, [torch.empty(0)]*n, [torch.empty(0)]*n

class ScriptedKuhnPokerAgent(DummyAgent):
    """
    Deterministic scripted agent for Kuhn Poker.
    Cards in OpenSpiel Kuhn Poker are represented as integers in the information state:
    0 = Jack (J)
    1 = Queen (Q)
    2 = King (K)
    
    The information state string typically starts with the player's card.
    """
    def batched_act(self, observations: list[str]):
        actions = []
        for obs in observations:
            legal_actions = self.extract_legal_actions(obs)
            
            # Parse the information state
            # "Information State:\n0pb\n" -> 0 is the card
            info_state_match = re.search(r"Information State:\n(.*?)\n", obs)
            card = -1
            if info_state_match:
                info_state = info_state_match.group(1).strip()
                if info_state and info_state[0].isdigit():
                    card = int(info_state[0])
                    
            # Logic
            action = "pass"
            if card == 0: # Jack
                # Bluff at a fixed low probability
                if "bet" in legal_actions and self.rng.random() < 0.2:
                    action = "bet"
                else:
                    action = "pass" if "pass" in legal_actions else self.rng.choice(legal_actions)
            elif card == 1: # Queen
                # Mostly pass/check, call/bet moderate probability
                if self.rng.random() < 0.5:
                    action = "bet" if "bet" in legal_actions else self.rng.choice(legal_actions)
                else:
                    action = "pass" if "pass" in legal_actions else self.rng.choice(legal_actions)
            elif card == 2: # King
                # Bet/raise with high probability, call with high probability
                if "bet" in legal_actions and self.rng.random() < 0.9:
                    action = "bet"
                else:
                    action = "pass" if "pass" in legal_actions else self.rng.choice(legal_actions)
            else:
                # Fallback if card parsing failed
                action = self.rng.choice(legal_actions)
                
            actions.append(f"ACTION: {action}")
            
        n = len(observations)
        return actions, [torch.empty(0)]*n, [torch.empty(0)]*n, [torch.empty(0)]*n
