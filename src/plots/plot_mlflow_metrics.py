# src/plots/plot_mlflow_metrics.py
import os
import argparse
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def get_run_id_by_name(experiment_name: str, run_name: str) -> str:
    mlflow.set_experiment(experiment_name)
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Experimento '{experiment_name}' no existe.")
    # Busca el run por su nombre (mlflow.runName)
    df = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        output_format="pandas",
        max_results=50,
        order_by=["attributes.start_time DESC"],
    )
    if df.empty:
        raise ValueError(f"No se encontró run con run_name='{run_name}' en experimento '{experiment_name}'.")
    return df.iloc[0]["run_id"]

def fetch_metric_series(client: MlflowClient, run_id: str, metric_key: str):
    """
    Devuelve dos listas: steps y values para la métrica dada.
    Si la métrica no existe, devuelve listas vacías.
    """
    hist = client.get_metric_history(run_id, metric_key)
    if not hist:
        return [], []
    steps = [m.step for m in hist]
    values = [m.value for m in hist]
    # Asegurar orden por step
    steps, values = zip(*sorted(zip(steps, values)))
    return list(steps), list(values)

def plot_two_series(x1, y1, x2, y2, title, y_label, labels, out_path: Path):
    plt.figure()
    if x1 and y1:
        plt.plot(x1, y1, label=labels[0])
    if x2 and y2:
        plt.plot(x2, y2, label=labels[1])
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Genera gráficos train vs val desde MLflow.")
    parser.add_argument("--experiment", type=str, default="credit_risk_mlp", help="Nombre del experimento de MLflow")
    parser.add_argument("--run-name", type=str, required=True, help="tags.mlflow.runName del run (ej: phase2_run_09)")
    parser.add_argument("--outdir", type=str, default="reports", help="Carpeta de salida para los PNG")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    # Busca el run y trae métricas
    run_id = get_run_id_by_name(args.experiment, args.run_name)
    client = MlflowClient()

    # Métricas que graficaremos
    keys = {
        "loss": ("train_loss", "val_loss"),
        "auc": ("train_roc_auc", "val_roc_auc"),
    }

    # Loss
    tr_steps, tr_loss = fetch_metric_series(client, run_id, keys["loss"][0])
    va_steps, va_loss = fetch_metric_series(client, run_id, keys["loss"][1])
    loss_path = outdir / f"{args.run_name}_loss.png"
    plot_two_series(
        tr_steps, tr_loss, va_steps, va_loss,
        title=f"Loss - {args.run_name}",
        y_label="Binary Cross-Entropy (logits)",
        labels=("train_loss", "val_loss"),
        out_path=loss_path
    )

    # AUC
    tr_steps, tr_auc = fetch_metric_series(client, run_id, keys["auc"][0])
    va_steps, va_auc = fetch_metric_series(client, run_id, keys["auc"][1])
    auc_path = outdir / f"{args.run_name}_auc.png"
    plot_two_series(
        tr_steps, tr_auc, va_steps, va_auc,
        title=f"ROC AUC - {args.run_name}",
        y_label="AUC",
        labels=("train_roc_auc", "val_roc_auc"),
        out_path=auc_path
    )

    print(f"✅ Guardado: {loss_path}")
    print(f"✅ Guardado: {auc_path}")

if __name__ == "__main__":
    main()
