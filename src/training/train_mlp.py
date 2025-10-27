import os
from pathlib import Path
import time
import json
import numpy as np
import pandas as pd
from typing import List, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score
)

import mlflow
import mlflow.pytorch

# ----------------------------
# Paths y dispositivo
# ----------------------------
DATA_DIR = Path("data/scaled")
X_TRAIN = DATA_DIR / "X_train_scaled.csv"
X_VALID = DATA_DIR / "X_valid_scaled.csv"
Y_TRAIN = DATA_DIR / "y_train.csv"
Y_VALID = DATA_DIR / "y_valid.csv"

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# ----------------------------
# Utilidades
# ----------------------------
def load_data():
    X_tr = pd.read_csv(X_TRAIN).values.astype(np.float32)
    X_va = pd.read_csv(X_VALID).values.astype(np.float32)
    y_tr = pd.read_csv(Y_TRAIN).values.reshape(-1).astype(np.float32)
    y_va = pd.read_csv(Y_VALID).values.reshape(-1).astype(np.float32)

    return (
        torch.tensor(X_tr, dtype=torch.float32).to(device),
        torch.tensor(y_tr, dtype=torch.float32).view(-1, 1).to(device),
        torch.tensor(X_va, dtype=torch.float32).to(device),
        torch.tensor(y_va, dtype=torch.float32).view(-1, 1).to(device),
    )

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], activation: str = "ReLU",
                 dropout: float = 0.4, batch_norm: bool = True):
        super().__init__()
        layers = []
        last = in_dim
        act_layer = {"ReLU": nn.ReLU, "LeakyReLU": nn.LeakyReLU, "GELU": nn.GELU}[activation]

        for h in hidden:
            layers.append(nn.Linear(last, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act_layer())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def compute_metrics(y_true_np: np.ndarray, y_prob_np: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob_np >= thr).astype(int)
    acc = accuracy_score(y_true_np, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true_np, y_pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true_np, y_prob_np)
    except ValueError:
        auc = float("nan")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc}

def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, criterion) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = criterion(logits, y).item()
        prob = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        y_true = y.detach().cpu().numpy().reshape(-1)
        m = compute_metrics(y_true, prob, thr=0.5)
        m["loss"] = loss
    return m

def train_one_run(cfg: Dict):
    mlflow.set_experiment(cfg.get("mlflow_experiment", "credit_risk_mlp"))
    run_name = cfg.get("run_name", f"mlp_{int(time.time())}")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "hidden_layers": json.dumps(cfg["hidden_layers"]),
            "activation": cfg["activation"],
            "dropout": cfg["dropout"],
            "batch_norm": cfg["batch_norm"],
            "optimizer": cfg["optimizer"],
            "learning_rate": cfg["lr"],
            "weight_decay": cfg["weight_decay"],
            "batch_size": cfg["batch_size"],
            "epochs": cfg["epochs"],
            "early_stopping_patience": cfg["early_stopping_patience"]
        })

        X_tr, y_tr, X_va, y_va = load_data()
        in_dim = X_tr.shape[1]

        model = MLP(
            in_dim=in_dim,
            hidden=cfg["hidden_layers"],
            activation=cfg["activation"],
            dropout=cfg["dropout"],
            batch_norm=cfg["batch_norm"],
        ).to(device)

        # Pérdida balanceada
        y_tr_cpu = y_tr.detach().cpu().numpy().reshape(-1)
        pos = float((y_tr_cpu == 1).sum())
        neg = float((y_tr_cpu == 0).sum())
        pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Optimizador
        if cfg["optimizer"].lower() == "adam":
            optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        elif cfg["optimizer"].lower() == "adamw":
            optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        else:
            optimizer = optim.SGD(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"], momentum=0.9)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )

        best_val_loss = float("inf")
        patience = 0
        best_model_path = MODELS_DIR / f"best_{run_name}.pt"

        n = X_tr.shape[0]
        bs = cfg["batch_size"]
        epochs = cfg["epochs"]

        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            idx = torch.randperm(n)
            for i in range(0, n, bs):
                batch_idx = idx[i:i+bs]
                xb = X_tr[batch_idx]
                yb = y_tr[batch_idx]
                logits = model(xb)
                loss = criterion(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            train_metrics = evaluate(model, X_tr, y_tr, criterion)
            val_metrics = evaluate(model, X_va, y_va, criterion)
            scheduler.step(val_metrics["loss"])

            mlflow.log_metrics({
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "val_roc_auc": val_metrics["roc_auc"],
                "val_f1": val_metrics["f1"],
                "lr": optimizer.param_groups[0]["lr"]
            }, step=epoch)

            print(
                f"[{run_name}] Epoch {epoch}/{epochs} | "
                f"Train Loss {train_metrics['loss']:.4f} | "
                f"Val Loss {val_metrics['loss']:.4f} | "
                f"AUC {val_metrics['roc_auc']:.4f} | F1 {val_metrics['f1']:.4f}"
            )

            if val_metrics["loss"] < best_val_loss - 1e-4:
                best_val_loss = val_metrics["loss"]
                patience = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience += 1
                if patience >= cfg["early_stopping_patience"]:
                    print(f"[{run_name}] Early stopping en epoch {epoch}")
                    break

        model.load_state_dict(torch.load(best_model_path, map_location=device))
        final_val = evaluate(model, X_va, y_va, criterion)
        mlflow.log_metrics({f"final_val_{k}": v for k, v in final_val.items()})
        mlflow.pytorch.log_model(model, artifact_path="model")
        print(f"[{run_name}] Final Val: AUC={final_val['roc_auc']:.4f} Acc={final_val['accuracy']:.4f}")
        return final_val

def main():
    grid = [
        {"hidden_layers":[128,64,32], "activation":"ReLU", "dropout":0.5, "batch_norm":True,
         "optimizer":"adam", "lr":1e-3, "weight_decay":1e-4, "batch_size":128, "epochs":70, "early_stopping_patience":15},
        {"hidden_layers":[128,64,32], "activation":"ReLU", "dropout":0.4, "batch_norm":True,
         "optimizer":"adamw", "lr":1e-3, "weight_decay":5e-4, "batch_size":128, "epochs":70, "early_stopping_patience":15},
        {"hidden_layers":[64,64,32], "activation":"ReLU", "dropout":0.5, "batch_norm":True,
         "optimizer":"adam", "lr":2e-3, "weight_decay":1e-4, "batch_size":128, "epochs":70, "early_stopping_patience":15},
        {"hidden_layers":[128,64], "activation":"ReLU", "dropout":0.5, "batch_norm":True,
         "optimizer":"adamw", "lr":1e-3, "weight_decay":1e-3, "batch_size":128, "epochs":80, "early_stopping_patience":15},
        {"hidden_layers":[256,128,64], "activation":"ReLU", "dropout":0.5, "batch_norm":True,
         "optimizer":"adam", "lr":1e-3, "weight_decay":5e-4, "batch_size":128, "epochs":80, "early_stopping_patience":15},
    ]

    common = {
        "mlflow_experiment": "credit_risk_mlp_v2",
        "use_pos_weight": True,
    }

    results = []
    for i, g in enumerate(grid, start=1):
        cfg = {**g, **common, "run_name": f"phase2_run_{i:02d}"}
        print(f"\n=== Iniciando {cfg['run_name']} ===")
        res = train_one_run(cfg)
        res["run_name"] = cfg["run_name"]
        results.append(res)

    print("\n=== RESUMEN (val) ===")
    for r in results:
        print(f"{r['run_name']}: AUC={r['roc_auc']:.4f} Acc={r['accuracy']:.4f} F1={r['f1']:.4f}")

if __name__ == "__main__":
    main()
