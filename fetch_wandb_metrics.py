import wandb
import pandas as pd
import argparse

def fetch_metrics(entity, project_name):
    print(f"Fetching runs from {entity}/{project_name}...")
    api = wandb.Api()
    
    # Get all runs in the specified project
    runs = api.runs(f"{entity}/{project_name}")
    
    data = []
    
    for run in runs:
        # We only care about runs that have logged plasticity metrics
        if "plasticity/feature_variance" in run.summary:
            run_data = {
                "Run Name": run.name,
                "State": run.state,
                "Final Variance": round(run.summary.get("plasticity/feature_variance", 0), 2),
                "Final Dormant %": round(run.summary.get("plasticity/dormant_neurons_pct", 0), 2),
                "Final Refusal Rate": round(run.summary.get("safety/refusal_rate", 0), 2)
            }
            data.append(run_data)
            
    if not data:
        print("No runs found with plasticity metrics!")
        return
        
    # Convert to DataFrame for a beautiful table print
    df = pd.DataFrame(data)
    
    # Sort by run name for easier reading
    df = df.sort_values(by="Run Name")
    
    print("\n=== FINAL ICLR EXPERIMENTAL METRICS ===")
    print(df.to_markdown(index=False))
    
    # Also save to CSV just in case
    df.to_csv("final_iclr_metrics.csv", index=False)
    print("\nSaved full table to final_iclr_metrics.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch WandB Metrics")
    parser.add_argument("--entity", type=str, required=True, help="Your WandB Username or Team Name")
    parser.add_argument("--project", type=str, default="continual-self-play", help="WandB Project Name")
    args = parser.parse_args()
    
    fetch_metrics(args.entity, args.project)
