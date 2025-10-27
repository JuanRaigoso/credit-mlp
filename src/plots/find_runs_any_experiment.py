# src/plots/find_runs_any_experiment.py
import mlflow
import pandas as pd
import sys

pattern = sys.argv[1] if len(sys.argv) > 1 else "phase2_"

exps = mlflow.search_experiments()
found = False
for e in exps:
    df = mlflow.search_runs([e.experiment_id], output_format="pandas",
                            order_by=["attributes.start_time DESC"])
    if df.empty: 
        continue
    df = df[["run_id","tags.mlflow.runName"]].rename(columns={"tags.mlflow.runName":"run_name"})
    mask = df["run_name"].fillna("").str.contains(pattern, case=False, na=False)
    if mask.any():
        found = True
        print(f"\n=== EXPERIMENTO: {e.name} (id {e.experiment_id}) ===")
        print(df[mask].head(50).to_string(index=False))

if not found:
    print(f"No se encontraron runs cuyo nombre contenga '{pattern}'.")
