from envs.base_env import LLMEnvironment
from typing import Tuple, Optional, Any
import re
import jax
import jax.numpy as jnp

class JaxMARLAdapter(LLMEnvironment):
    """
    Adapter for JaxMARL games (e.g., Overcooked, SMAX).
    JaxMARL runs heavily on XLA/GPU. This adapter ensures we run on a specific GPU
    and translate the JAX arrays into text for the LLM.
    """
    def __init__(self, game_name: str = "overcooked"):
        try:
            import jaxmarl
            from jaxmarl import make
        except ImportError:
            raise ImportError("JaxMARL is not installed. Run: pip install jaxmarl")
            
        self.game_name = game_name
        self.env = make(game_name)
        
        # We JIT compile the step and reset functions for speed
        self.jit_reset = jax.jit(self.env.reset)
        self.jit_step = jax.jit(self.env.step)
        
        # JAX state variables
        self.key = jax.random.PRNGKey(0)
        self.state = None
        self.obs = None

    def reset(self, player_id: int = 1) -> str:
        self.key, subkey = jax.random.split(self.key)
        self.obs, self.state = self.jit_reset(subkey)
        
        # JaxMARL often returns a dict of observations per agent
        # Example agent name: 'agent_0'
        self.agent_name = f"agent_{player_id - 1}"
        
        return self._render_observation()

    def step(self, text_action: str) -> Tuple[str, float, bool, dict]:
        parsed_action = self._parse_action(text_action)
        
        if parsed_action is None:
            return "INVALID", -10.0, True, {"msg": "Invalid action format."}
            
        # JaxMARL step expects a dictionary of actions for all agents
        # Since this is a scaffold, we simulate a dummy action for other agents or run self-play
        # For simplicity, we just pass the parsed action for our agent
        actions = {self.agent_name: jnp.array(parsed_action)}
        
        self.key, subkey = jax.random.split(self.key)
        self.obs, self.state, rewards, dones, infos = self.jit_step(subkey, self.state, actions)
        
        reward = float(rewards.get(self.agent_name, 0.0))
        done = bool(dones.get(self.agent_name, False))
        
        if done:
            return "Game Over.", reward, done, {}
        else:
            return self._render_observation(), reward, False, {}

    def _parse_action(self, text_response: str) -> Optional[int]:
        match = re.search(r"ACTION:\s*(.*)", text_response, re.IGNORECASE)
        if not match:
            return None
            
        action_str = match.group(1).strip()
        if action_str.isdigit():
            return int(action_str)
        return None

    def _render_observation(self) -> str:
        # Extract the JAX array for this specific agent
        agent_obs = self.obs[self.agent_name]
        
        prompt = f"You are {self.agent_name} in JaxMARL {self.game_name}.\n"
        prompt += f"Observation Array:\n{agent_obs}\n\n"
        prompt += "Think briefly, then end with:\nACTION: <your move integer>\n"
        return prompt
