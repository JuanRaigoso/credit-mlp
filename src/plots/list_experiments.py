# src/plots/list_experiments.py
import mlflow

exps = mlflow.search_experiments()
for e in exps:
    print(f"- ID={e.experiment_id}  NAME={e.name}  ARTIFACT_LOC={e.artifact_location}")
