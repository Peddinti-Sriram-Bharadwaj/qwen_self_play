import sys

def check_textarena():
    print("--- Testing TextArena ---")
    try:
        from envs.env_factory import EnvFactory
        env = EnvFactory.get_env("textarena", "TicTacToe-v0")
        obs = env.reset(player_id=0)
        print("✓ TextArena instantiated successfully.")
        n_obs, r, d, info = env.step("ACTION: [4]")
        print("✓ TextArena mock step successful.")
        return True
    except Exception as e:
        print(f"✗ TextArena failed: {e}")
        return False

def check_openspiel():
    print("\n--- Testing OpenSpiel ---")
    try:
        from envs.env_factory import EnvFactory
        env = EnvFactory.get_env("openspiel", "kuhn_poker")
        obs = env.reset(player_id=0)
        print("✓ OpenSpiel instantiated successfully.")
        n_obs, r, d, info = env.step("ACTION: Pass")
        print("✓ OpenSpiel mock step successful.")
        return True
    except Exception as e:
        print(f"✗ OpenSpiel failed: {e}")
        return False

def check_pettingzoo():
    print("\n--- Testing PettingZoo ---")
    try:
        from envs.env_factory import EnvFactory
        env = EnvFactory.get_env("pettingzoo", "tictactoe_v3")
        obs = env.reset(player_id=0)
        print("✓ PettingZoo instantiated successfully.")
        n_obs, r, d, info = env.step("ACTION: 4")
        print("✓ PettingZoo mock step successful.")
        return True
    except Exception as e:
        print(f"✗ PettingZoo failed: {e}")
        return False

def check_jaxmarl():
    print("\n--- Testing JaxMARL ---")
    try:
        from envs.env_factory import EnvFactory
        env = EnvFactory.get_env("jaxmarl", "overcooked")
        obs = env.reset(player_id=1)
        print("✓ JaxMARL instantiated successfully.")
        n_obs, r, d, info = env.step("ACTION: 0")
        print("✓ JaxMARL mock step successful.")
        return True
    except Exception as e:
        print(f"✗ JaxMARL failed: {e}")
        return False

def main():
    print("==============================================")
    print("     MARL Integrations Health Check           ")
    print("==============================================\n")
    
    results = {
        "TextArena": check_textarena(),
        "OpenSpiel": check_openspiel(),
        "PettingZoo": check_pettingzoo(),
        "JaxMARL": check_jaxmarl(),
    }
    
    print("\n==============================================")
    print("                 SUMMARY                      ")
    print("==============================================")
    
    all_passed = True
    for env, passed in results.items():
        if passed:
            print(f"{env:15} | PASS")
        else:
            print(f"{env:15} | FAIL")
            all_passed = False
            
    print("==============================================")
    if all_passed:
        print("All environments are fully operational!")
        sys.exit(0)
    else:
        print("Some environments failed. Please check the logs above and install missing dependencies.")
        print("Common fixes:")
        print(" - PettingZoo Pygame Error: pip install \"pettingzoo[classic]\"")
        print(" - OpenSpiel not found: pip install open_spiel")
        print(" - JaxMARL not found: pip install jaxmarl jax jaxlib")
        print(" - TextArena not found: pip install textarena")
        sys.exit(1)

if __name__ == "__main__":
    main()
