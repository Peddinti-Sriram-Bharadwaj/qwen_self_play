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
    _compiled_envs = {}

    def __init__(self, game_name: str = "overcooked"):
        try:
            import jaxmarl
            from jaxmarl import make
        except ImportError:
            raise ImportError("JaxMARL is not installed. Run: pip install jaxmarl")
            
        self.game_name = game_name
        self.env = make(game_name)
        
        # Cache JIT compilation at the class level so deepcopy doesn't trigger recompiles
        if game_name not in JaxMARLAdapter._compiled_envs:
            JaxMARLAdapter._compiled_envs[game_name] = {
                'reset': jax.jit(self.env.reset),
                'step': jax.jit(self.env.step)
            }
            
        self.jit_reset = JaxMARLAdapter._compiled_envs[game_name]['reset']
        self.jit_step = JaxMARLAdapter._compiled_envs[game_name]['step']
        
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
            
        # JaxMARL step expects a dictionary of actions for all agents (parallel env)
        # We simulate placeholder action (0) for other agents since we only parse for our agent
        actions = {}
        for agent in self.env.agents:
            if agent == self.agent_name:
                actions[agent] = jnp.array(parsed_action)
            else:
                actions[agent] = jnp.array(0)
        
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
