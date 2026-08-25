from envs.base_env import LLMEnvironment
from typing import Tuple, Optional, Any
import re

class TextArenaAdapter(LLMEnvironment):
    """
    Adapter for TextArena games (e.g., Tic-Tac-Toe, Kuhn Poker).
    TextArena is natively text-based, so this adapter mainly handles action extraction 
    and enforcing the strict 'ACTION: <move>' grammar.
    """
    def __init__(self, game_name: str = "tic-tac-toe"):
        try:
            import textarena as ta
        except ImportError:
            raise ImportError("TextArena is not installed. Run: pip install textarena")
            
        self.game_name = game_name
        self.env = ta.make(game_name)
        
    def reset(self, player_id: int = 1) -> str:
        # Most 2-player TextArena games require num_players=2
        self.env.reset(num_players=2)
        # Fetch current player and their observation
        current_player_id, text_obs = self.env.get_observation()
        
        prompt = f"You are Player {current_player_id} in {self.game_name}.\n"
        prompt += f"Observation:\n{text_obs}\n\n"
        prompt += "Think briefly, then end with:\nACTION: <your move>\n"
        return prompt

    def step(self, text_action: str) -> Tuple[str, float, bool, dict]:
        parsed_action = self._parse_action(text_action)
        
        if parsed_action is None:
            # Invalid action
            return "INVALID", -10.0, True, {"msg": "Invalid action format."}
            
        # Step the textarena environment with the string
        done, info = self.env.step(parsed_action)
        
        rewards = self.env.state.rewards or {}
        # Assume player 0 for simplicity in this walking skeleton if not set
        player_id = self.env.state.current_player_id if self.env.state.current_player_id is not None else 0
        
        if done:
            reward = rewards.get(player_id, 0.0)
            next_prompt = "Game Over."
        else:
            reward = 0.0
            next_player_id, next_obs = self.env.get_observation()
            next_prompt = f"Observation:\n{next_obs}\n\n"
            next_prompt += "Think briefly, then end with:\nACTION: <your move>\n"
            
        return next_prompt, reward, done, info

    def _parse_action(self, text_response: str) -> Optional[Any]:
        """
        Extracts the action from 'ACTION: <move>'.
        """
        match = re.search(r"ACTION:\s*(.*)", text_response, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _render_observation(self) -> str:
        # Handled dynamically in reset and step for TextArena
        pass
