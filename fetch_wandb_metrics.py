import wandb
import pandas as pd
import argparse

def fetch_metrics(entity, project_name):
    print(f"Fetching runs from {entity}/{project_name}...")
    api = wandb.Api()
    
    # Get all runs in the specified project, sorted by newest first
    runs = api.runs(f"{entity}/{project_name}", order="-created_at")
    
    data = []
    seen_names = set()
    history_dfs = []
    
    for run in runs:
        # Skip if we already processed a newer run with this exact name
        if run.name in seen_names:
            continue
            
        # We only care about runs that have logged plasticity metrics
        if "plasticity/feature_variance" in run.summary:
            seen_names.add(run.name)
            
            # 1. Grab final summary data
            run_data = {
                "Run Name": run.name,
                "State": run.state,
                "Final Variance": round(run.summary.get("plasticity/feature_variance", 0), 2),
                "Final Entropy": round(run.summary.get("dapo/policy_entropy", 0), 4),
                "Final Refusal Rate": round(run.summary.get("safety/refusal_rate", 0), 2)
            }
            data.append(run_data)
            
            # 2. Grab historical trend data for plotting
            print(f"  -> Downloading history for {run.name}...")
            # Fetch the history dataframe for specific keys
            hist_df = run.history(keys=["_step", "plasticity/feature_variance", "dapo/policy_entropy", "safety/refusal_rate"])
            hist_df["Run Name"] = run.name  # Tag the rows with the run name
            history_dfs.append(hist_df)
            
    if not data:
        print("No runs found with plasticity metrics!")
        return
        
    # Convert to DataFrame for a beautiful table print
    df = pd.DataFrame(data)
    df = df.sort_values(by="Run Name")
    
    print("\n=== FINAL ICLR EXPERIMENTAL METRICS ===")
    print(df.to_markdown(index=False))
    
    # Save the summary table
    df.to_csv("final_iclr_metrics.csv", index=False)
    print("\nSaved summary table to final_iclr_metrics.csv")
    
    # Combine all historical trends and save
    if history_dfs:
        full_history = pd.concat(history_dfs, ignore_index=True)
        # Sort by run name and step
        full_history = full_history.sort_values(by=["Run Name", "_step"])
        full_history.to_csv("iclr_trends_history.csv", index=False)
        print("Saved full historical trends (Variance, Entropy, Safety) to iclr_trends_history.csv")
        print("You can use this CSV to plot your graphs for the paper!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch WandB Metrics")
    parser.add_argument("--entity", type=str, required=True, help="Your WandB Username or Team Name")
    parser.add_argument("--project", type=str, default="continual-self-play", help="WandB Project Name")
    args = parser.parse_args()
    
    fetch_metrics(args.entity, args.project)
