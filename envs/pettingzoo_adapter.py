from envs.base_env import LLMEnvironment
from typing import Tuple, Optional, Any
import re

class PettingZooAdapter(LLMEnvironment):
    """
    Adapter for PettingZoo games (e.g., Tic-Tac-Toe, Connect Four, Simple Spread).
    Translates AEC (Agent Environment Cycle) or Parallel API numerical states into text.
    """
    def __init__(self, game_name: str = "tictactoe_v3"):
        try:
            from pettingzoo.classic import tictactoe_v3
            from pettingzoo.classic import connect_four_v3
        except ImportError:
            raise ImportError("PettingZoo is not installed. Run: pip install pettingzoo")
            
        self.game_name = game_name
        if game_name == "tictactoe_v3":
            self.env = tictactoe_v3.env()
        elif game_name == "connect_four_v3":
            self.env = connect_four_v3.env()
        else:
            raise ValueError(f"PettingZoo environment {game_name} not yet scaffolded in this adapter.")
            
    def reset(self, player_id: int = 1) -> str:
        self.env.reset()
        return self._render_observation()

    def step(self, text_action: str) -> Tuple[str, float, bool, dict]:
        parsed_action = self._parse_action(text_action)
        
        if parsed_action is None:
            return "INVALID", -10.0, True, {"msg": "Invalid action format."}
            
        # AEC step
        agent = self.env.agent_selection
        self.env.step(parsed_action)
        
        # PettingZoo rewards are accessed via env.rewards
        reward = self.env.rewards[agent]
        done = self.env.terminations[agent] or self.env.truncations[agent]
        
        if done:
            return "Game Over.", reward, done, {}
        else:
            return self._render_observation(), 0.0, False, {}

    def _parse_action(self, text_response: str) -> Optional[int]:
        match = re.search(r"ACTION:\s*(.*)", text_response, re.IGNORECASE)
        if not match:
            return None
            
        action_str = match.group(1).strip()
        
        # PettingZoo classic games use discrete integer actions (e.g., 0-8 for TicTacToe)
        if action_str.isdigit():
            action = int(action_str)
            # Ensure it's legal via action_mask
            agent = self.env.agent_selection
            action_mask = self.env.observe(agent)['action_mask']
            if action < len(action_mask) and action_mask[action] == 1:
                return action
        return None

    def _render_observation(self) -> str:
        agent = self.env.agent_selection
        observation = self.env.observe(agent)
        
        # In PettingZoo, the observation is a dict containing 'observation' (the board) and 'action_mask'
        board = observation['observation']
        action_mask = observation['action_mask']
        
        legal_actions = [i for i, valid in enumerate(action_mask) if valid == 1]
        
        prompt = f"You are {agent} in {self.game_name}.\n"
        prompt += f"Board State Array:\n{board[:, :, 0]}\n\n" # Channel 0 usually represents the player's pieces
        prompt += f"Legal Action Integers: {legal_actions}\n"
        prompt += "Think briefly, then end with:\nACTION: <your move>\n"
        return prompt
