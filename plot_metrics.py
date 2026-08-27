import pandas as pd
import matplotlib.pyplot as plt

def plot_trends():
    try:
        df = pd.read_csv("iclr_trends_history.csv")
    except FileNotFoundError:
        print("Could not find iclr_trends_history.csv. Make sure you've fetched the metrics first!")
        return

    # Filter out empty or un-logged steps
    df = df.dropna(subset=["_step"])

    # 1. Feature Variance Plot
    plt.figure(figsize=(10, 6))
    for run_name in df["Run Name"].unique():
        run_data = df[df["Run Name"] == run_name]
        # Filter for rows that actually have variance logged
        run_data = run_data.dropna(subset=["plasticity/feature_variance"])
        
        plt.plot(run_data["_step"], run_data["plasticity/feature_variance"], label=run_name, linewidth=2)
        
    plt.title("Plasticity Collapse: Feature Variance over Time", fontsize=14, pad=15)
    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel("Feature Variance", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("feature_variance_plot.png", dpi=300)
    print("Saved feature_variance_plot.png")

    # 2. Policy Entropy Plot
    plt.figure(figsize=(10, 6))
    for run_name in df["Run Name"].unique():
        run_data = df[df["Run Name"] == run_name]
        # Filter for rows that actually have entropy logged
        run_data = run_data.dropna(subset=["dapo/policy_entropy"])
        
        plt.plot(run_data["_step"], run_data["dapo/policy_entropy"], label=run_name, linewidth=2)
        
    plt.title("Policy Exploration: Entropy over Time", fontsize=14, pad=15)
    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel("Policy Entropy", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("policy_entropy_plot.png", dpi=300)
    print("Saved policy_entropy_plot.png")

if __name__ == "__main__":
    plot_trends()
