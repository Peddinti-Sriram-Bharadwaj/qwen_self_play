from abc import ABC, abstractmethod
from typing import Tuple, Optional, Any

class LLMEnvironment(ABC):
    """
    The Base Adapter interface for all environments (OpenSpiel, PettingZoo, TextArena, JaxMARL).
    This strictly enforces the boundary between the LLM (which needs text) and the Game Engine (which needs numbers).
    """
    
    @abstractmethod
    def reset(self, player_id: int = 1) -> str:
        """
        Resets the environment and returns the initial text observation for the given player.
        """
        pass

    @abstractmethod
    def step(self, text_action: str) -> Tuple[str, float, bool, dict]:
        """
        Receives a text response from the LLM.
        1. Parses the text into a numerical/symbolic action.
        2. Steps the underlying game engine.
        3. Serializes the new state into a text observation.
        
        Returns: (text_observation, reward, done, info)
        """
        pass

    @abstractmethod
    def _parse_action(self, text_response: str) -> Optional[Any]:
        """
        Helper method to parse the `ACTION: <move>` grammar.
        Returns None if the action is invalid.
        """
        pass

    @abstractmethod
    def _render_observation(self) -> str:
        """
        Helper method to deterministically convert the numerical game state into a text prompt.
        """
        pass
