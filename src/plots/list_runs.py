import mlflow
import pandas as pd

exp = mlflow.get_experiment_by_name("credit_risk_mlp")
if exp is None:
    print("No existe experimento 'credit_risk_mlp'. Revisa el nombre.")
else:
    df = mlflow.search_runs(
        [exp.experiment_id],
        output_format="pandas",
        order_by=["attributes.start_time DESC"]
    )
    cols = ["run_id", "tags.mlflow.runName"]
    print(df[cols].head(30).to_string(index=False))
