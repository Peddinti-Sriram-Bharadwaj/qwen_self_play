from envs.base_env import LLMEnvironment
from typing import Tuple, Optional, Any
import re

class OpenSpielAdapter(LLMEnvironment):
    """
    Adapter for OpenSpiel games (e.g., Kuhn Poker, Leduc Poker, Hanabi).
    OpenSpiel requires translating the numerical game state (information state) into text,
    and constraining the LLM's action output back into the discrete OpenSpiel action ID.
    """
    def __init__(self, game_name: str = "kuhn_poker"):
        try:
            import pyspiel
        except ImportError:
            raise ImportError("OpenSpiel is not installed. Run: pip install open_spiel")
            
        self.game_name = game_name
        self.game = pyspiel.load_game(game_name)
        self.state = self.game.new_initial_state()
        
    def reset(self, player_id: int = 1) -> str:
        self.state = self.game.new_initial_state()
        
        # Handle chance nodes (e.g. dealing cards)
        while self.state.is_chance_node():
            outcomes_with_probs = self.state.chance_outcomes()
            action_list, prob_list = zip(*outcomes_with_probs)
            # Just sample deterministically for the skeleton or use a random choice
            import random
            action = random.choices(action_list, weights=prob_list)[0]
            self.state.apply_action(action)
            
        return self._render_observation()

    def step(self, text_action: str) -> Tuple[str, float, bool, dict]:
        parsed_action = self._parse_action(text_action)
        
        if parsed_action is None:
            return "INVALID", -10.0, True, {"msg": "Invalid action format."}
            
        # Apply the discrete action to the OpenSpiel state engine
        try:
            self.state.apply_action(parsed_action)
        except Exception as e:
            return "INVALID", -10.0, True, {"msg": f"Illegal action. {e}"}
            
        # Handle intermediate chance nodes (e.g. dealing next street of cards)
        while self.state.is_chance_node() and not self.state.is_terminal():
            outcomes_with_probs = self.state.chance_outcomes()
            action_list, prob_list = zip(*outcomes_with_probs)
            import random
            action = random.choices(action_list, weights=prob_list)[0]
            self.state.apply_action(action)
            
        done = self.state.is_terminal()
        
        if done:
            rewards = self.state.returns()
            current_player = self.state.current_player()
            # If the game is over, we need the reward for the player who just acted, or all.
            # In OpenSpiel, returns() gives a list of rewards for each player [r0, r1]
            # Since this is a scaffold, we return Player 0's reward
            reward = rewards[0]
            return "Game Over.", reward, done, {"returns": rewards}
        else:
            return self._render_observation(), 0.0, False, {}

    def _parse_action(self, text_response: str) -> Optional[int]:
        """
        Extracts the action from 'ACTION: <move>' and maps it to the discrete OpenSpiel action ID.
        """
        match = re.search(r"ACTION:\s*(.*)", text_response, re.IGNORECASE)
        if not match:
            return None
            
        action_str = match.group(1).strip().lower()
        legal_actions = self.state.legal_actions()
        
        # OpenSpiel provides action_to_string, we map the LLM's text back to the integer ID
        for action_id in legal_actions:
            if action_str == self.state.action_to_string(self.state.current_player(), action_id).lower():
                return action_id
                
        # Fallback if they just output the digit
        if action_str.isdigit() and int(action_str) in legal_actions:
            return int(action_str)
            
        return None

    def _render_observation(self) -> str:
        """
        Renders the OpenSpiel Information State into text.
        """
        player = self.state.current_player()
        info_state_str = self.state.information_state_string(player)
        legal_actions = self.state.legal_actions()
        
        action_strs = [self.state.action_to_string(player, a) for a in legal_actions]
        
        prompt = f"You are Player {player} in {self.game_name}.\n"
        prompt += f"Information State:\n{info_state_str}\n\n"
        prompt += f"Legal Actions: {', '.join(action_strs)}\n"
        prompt += "Think briefly, then end with:\nACTION: <your move>\n"
        
        return prompt
