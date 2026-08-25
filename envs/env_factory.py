from envs.base_env import LLMEnvironment
from envs.textarena_adapter import TextArenaAdapter

class EnvFactory:
    """
    Instantiates the correct environment wrapper based on CLI args.
    """
    
    @staticmethod
    def get_env(backend: str, env_name: str) -> LLMEnvironment:
        backend = backend.lower()
        
        if backend == "textarena":
            from envs.textarena_adapter import TextArenaAdapter
            return TextArenaAdapter(game_name=env_name)
        elif backend == "pettingzoo":
            from envs.pettingzoo_adapter import PettingZooAdapter
            return PettingZooAdapter(game_name=env_name)
        elif backend == "openspiel":
            from envs.openspiel_adapter import OpenSpielAdapter
            return OpenSpielAdapter(game_name=env_name)
        elif backend == "jaxmarl":
            from envs.jaxmarl_adapter import JaxMARLAdapter
            return JaxMARLAdapter(game_name=env_name)
        else:
            raise ValueError(f"Unknown backend: {backend}")
