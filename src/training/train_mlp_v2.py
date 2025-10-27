import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

import mlflow
import mlflow.pytorch

# ----------------------------
# Rutas / dispositivo
# ----------------------------
DATA_CLEAN = Path("data") / "train_clean.csv"   # usamos el limpio base
MODELS_DIR = Path("models"); MODELS_DIR.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# ----------------------------
# Preprocesamiento (winsor + log + scale)
# ----------------------------
WINSOR_COLS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]
LOG_COLS = [
    "MonthlyIncome",
    "RevolvingUtilizationOfUnsecuredLines",
    "DebtRatio",
]

FEATURES_ALL = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
    "Sex_num",   # ⬅️ añade esta
]


def _winsorize_df(df: pd.DataFrame, caps: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    out = df.copy()
    for c, (low, high) in caps.items():
        out[c] = out[c].clip(lower=low, upper=high)
    return out

def _log1p_df(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        # asegurar no-negativos (ratios ya lo son; por si acaso números enteros también)
        out[c] = np.log1p(out[c].clip(lower=0))
    return out

def load_and_preprocess(random_state: int = 42, test_size: float = 0.2):
    df = pd.read_csv(DATA_CLEAN)

    y = df["SeriousDlqin2yrs"].astype(int).values
    X = df[FEATURES_ALL].copy()

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # --- Winsor (percentiles del TRAIN) ---
    caps = {}
    for c in WINSOR_COLS:
        if c in X_tr.columns:
            p1 = np.percentile(X_tr[c], 1)
            p99 = np.percentile(X_tr[c], 99)
            # manejar casos patológicos (p99 < p1 por ties)
            if p99 < p1:
                p1, p99 = min(p1, p99), max(p1, p99)
            caps[c] = (p1, p99)

    X_tr_w = _winsorize_df(X_tr, caps)
    X_va_w = _winsorize_df(X_va, caps)

    # --- Log-transform en columnas sesgadas ---
    X_tr_t = _log1p_df(X_tr_w, LOG_COLS)
    X_va_t = _log1p_df(X_va_w, LOG_COLS)

    # --- Escalado estándar (fit en train) ---
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_t.values.astype(np.float32))
    X_va_s = scaler.transform(X_va_t.values.astype(np.float32))

    meta = {
        "winsor_caps": {k: {"p1": float(v[0]), "p99": float(v[1])} for k, v in caps.items()},
        "log_cols": LOG_COLS,
        "features": FEATURES_ALL,
        "random_state": random_state,
        "test_size": test_size,
    }
    Path("reports").mkdir(exist_ok=True, parents=True)
    with open("reports/preprocess_v2_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Tensores a dispositivo
    Xt = torch.tensor(X_tr_s, dtype=torch.float32, device=device)
    Xv = torch.tensor(X_va_s, dtype=torch.float32, device=device)
    yt = torch.tensor(y_tr.reshape(-1, 1), dtype=torch.float32, device=device)
    yv = torch.tensor(y_va.reshape(-1, 1), dtype=torch.float32, device=device)
    return Xt, yt, Xv, yv

# ----------------------------
# Modelo
# ----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], activation: str = "ReLU",
                 dropout: float = 0.5, batch_norm: bool = True):
        super().__init__()
        layers = []
        last = in_dim
        act = {"ReLU": nn.ReLU, "LeakyReLU": nn.LeakyReLU, "GELU": nn.GELU}[activation]
        for h in hidden:
            layers += [nn.Linear(last, h)]
            if batch_norm:
                layers += [nn.BatchNorm1d(h)]
            layers += [act()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            last = h
        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # logits
        return self.net(x)

# ----------------------------
# Métricas / evaluación
# ----------------------------
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

# ----------------------------
# Entrenamiento de un run
# ----------------------------
def train_one_run(cfg: Dict):
    mlflow.set_experiment("credit_risk_mlp_v3")
    run_name = cfg.get("run_name", f"mlp_{int(time.time())}")

    with mlflow.start_run(run_name=run_name):
        # Log de hiperparámetros
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
            "early_stopping_patience": cfg["early_stopping_patience"],
            "use_pos_weight": True,
            "preprocess": "winsor(p1-p99)+log1p+standardize"
        })

        # --- PREPROCESAMIENTO AQUÍ ---
        X_tr, y_tr, X_va, y_va = load_and_preprocess()

        in_dim = X_tr.shape[1]
        model = MLP(
            in_dim=in_dim,
            hidden=cfg["hidden_layers"],
            activation=cfg["activation"],
            dropout=cfg["dropout"],
            batch_norm=cfg["batch_norm"],
        ).to(device)

        # pos_weight
        y_tr_cpu = y_tr.detach().cpu().numpy().reshape(-1)
        pos = float((y_tr_cpu == 1).sum())
        neg = float((y_tr_cpu == 0).sum())
        pw = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
        print(f"[INFO] Clase 0 (neg): {int(neg)} | Clase 1 (pos): {int(pos)} | pos_weight={pw.item():.4f}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

        # Optimizador / scheduler
        if cfg["optimizer"].lower() == "adam":
            optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        elif cfg["optimizer"].lower() == "adamw":
            optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        else:
            optimizer = optim.SGD(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"], momentum=0.9)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

        # Entrenamiento
        best_val = float("inf"); patience = 0
        best_path = MODELS_DIR / f"best_{run_name}.pt"
        n, bs, epochs = X_tr.shape[0], cfg["batch_size"], cfg["epochs"]

        for epoch in range(1, epochs + 1):
            model.train()
            idx = torch.randperm(n)
            epoch_loss = 0.0

            for i in range(0, n, bs):
                b = idx[i:i+bs]
                xb, yb = X_tr[b], y_tr[b]
                logits = model(xb)
                loss = criterion(logits, yb)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            tr_m = evaluate(model, X_tr, y_tr, criterion)
            va_m = evaluate(model, X_va, y_va, criterion)
            scheduler.step(va_m["loss"])

            mlflow.log_metrics({
                "train_loss": tr_m["loss"], "val_loss": va_m["loss"],
                "train_accuracy": tr_m["accuracy"], "val_accuracy": va_m["accuracy"],
                "train_precision": tr_m["precision"], "val_precision": va_m["precision"],
                "train_recall": tr_m["recall"], "val_recall": va_m["recall"],
                "train_f1": tr_m["f1"], "val_f1": va_m["f1"],
                "train_roc_auc": tr_m["roc_auc"], "val_roc_auc": va_m["roc_auc"],
                "lr": optimizer.param_groups[0]["lr"],
            }, step=epoch)

            print(
                f"[{run_name}] Epoch {epoch}/{epochs} | "
                f"Train Loss {tr_m['loss']:.4f} AUC {tr_m['roc_auc']:.4f} | "
                f"Val Loss {va_m['loss']:.4f} AUC {va_m['roc_auc']:.4f}"
            )

            if va_m["loss"] < best_val - 1e-4:
                best_val = va_m["loss"]; patience = 0
                torch.save(model.state_dict(), best_path)
            else:
                patience += 1
                if patience >= cfg["early_stopping_patience"]:
                    print(f"[{run_name}] Early stopping en epoch {epoch}")
                    break

        # Final
        model.load_state_dict(torch.load(best_path, map_location=device))
        model.eval()
        final_val = evaluate(model, X_va, y_va, criterion)
        mlflow.log_metrics({f"final_val_{k}": v for k, v in final_val.items()})

        # Log del modelo
        try:
            # incluir input_example para evitar warning
            example = X_va[:5].detach().cpu().numpy()
            mlflow.pytorch.log_model(model, artifact_path="model",
                                     input_example=example)
        except Exception:
            mlflow.pytorch.log_model(model, artifact_path="model")

        mlflow.log_artifact(str(best_path), artifact_path="checkpoints")
        print(f"[{run_name}] Final Val: AUC={final_val['roc_auc']:.4f} Acc={final_val['accuracy']:.4f}")
        return final_val

# ----------------------------
# Grid de la Fase 2 (5 modelos)
# ----------------------------
def main():
    grid = [
        {"run_name":"phase2_run_01","hidden_layers":[128,64,32],"activation":"ReLU","dropout":0.50,"batch_norm":True,
         "optimizer":"adam","lr":1e-3,"weight_decay":1e-4,"batch_size":256,"epochs":70,"early_stopping_patience":15},
        {"run_name":"phase2_run_02","hidden_layers":[128,64,32],"activation":"ReLU","dropout":0.40,"batch_norm":True,
         "optimizer":"adamw","lr":1e-3,"weight_decay":5e-4,"batch_size":256,"epochs":70,"early_stopping_patience":15},
        {"run_name":"phase2_run_03","hidden_layers":[64,64,32],"activation":"ReLU","dropout":0.50,"batch_norm":True,
         "optimizer":"adam","lr":8e-4,"weight_decay":1e-4,"batch_size":256,"epochs":70,"early_stopping_patience":15},
        {"run_name":"phase2_run_04","hidden_layers":[128,64],"activation":"ReLU","dropout":0.50,"batch_norm":True,
         "optimizer":"adamw","lr":8e-4,"weight_decay":1e-3,"batch_size":256,"epochs":80,"early_stopping_patience":15},
        {"run_name":"phase2_run_05","hidden_layers":[256,128,64],"activation":"ReLU","dropout":0.50,"batch_norm":True,
         "optimizer":"adam","lr":1e-3,"weight_decay":5e-4,"batch_size":256,"epochs":80,"early_stopping_patience":15},
        {"run_name":"phase2_run_06","hidden_layers":[256,128,64],"activation":"ReLU","dropout":0.50,"batch_norm":False,
         "optimizer":"adam","lr":1e-1,"weight_decay":5e-4,"batch_size":64,"epochs":80,"early_stopping_patience":15},
        {"run_name":"phase2_run_07","hidden_layers":[256,128,64],"activation":"ReLU","dropout":0.50,"batch_norm":True,
         "optimizer":"adam","lr":1e-1,"weight_decay":5e-4,"batch_size":64,"epochs":80,"early_stopping_patience":15},
        {"run_name":"phase2_run_08","hidden_layers":[256,128,64],"activation":"ReLU","dropout":0.50,"batch_norm":False,
         "optimizer":"adam","lr":1e-1,"weight_decay":5e-4,"batch_size":32,"epochs":80,"early_stopping_patience":15},
        {"run_name":"phase2_run_09","hidden_layers":[256,128,64],"activation":"ReLU","dropout":0.50,"batch_norm":False,
         "optimizer":"adam","lr":1e-1,"weight_decay":5e-4,"batch_size":24,"epochs":80,"early_stopping_patience":15},
    ]
    results = []
    for cfg in grid:
        print(f"\n=== Iniciando {cfg['run_name']} ===")
        res = train_one_run(cfg)
        res["run_name"] = cfg["run_name"]
        results.append(res)

    print("\n=== RESUMEN (val) ===")
    for r in results:
        print(f"{r['run_name']}: AUC={r['roc_auc']:.4f} Acc={r['accuracy']:.4f} F1={r['f1']:.4f}")

if __name__ == "__main__":
    main()
