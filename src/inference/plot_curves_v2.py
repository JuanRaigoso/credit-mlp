# src/inference/plot_curves_v2.py
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    accuracy_score, precision_recall_fscore_support
)
import matplotlib.pyplot as plt

DATA_CLEAN = Path("data") / "train_clean.csv"
META_PATH  = Path("reports") / "preprocess_v2_meta.json"
OUT_DIR    = Path("reports") / "inference"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def winsorize_df(df, caps):
    out = df.copy()
    for c, v in caps.items():
        lo, hi = float(v["p1"]), float(v["p99"])
        out[c] = out[c].clip(lower=lo, upper=hi)
    return out

def log1p_df(df, cols):
    out = df.copy()
    for c in cols:
        out[c] = np.log1p(out[c].clip(lower=0))
    return out

@torch.no_grad()
def predict_proba(model, X_np):
    model.eval()
    X_t = torch.tensor(X_np.astype(np.float32))
    logits = model(X_t)
    prob = torch.sigmoid(logits).cpu().numpy().reshape(-1)
    return prob

def compute_metrics_at_thr(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    auc = roc_auc_score(y_true, y_prob)
    return {"accuracy":acc, "precision":prec, "recall":rec, "f1":f1, "roc_auc":auc, "threshold":thr}

def plot_and_save_curves(y_true, y_prob, prefix):
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve"); plt.legend()
    plt.tight_layout()
    roc_png = OUT_DIR / f"{prefix}_roc_curve.png"
    plt.savefig(roc_png, dpi=150)
    plt.close()

    roc_df = pd.DataFrame({"fpr":fpr, "tpr":tpr})
    roc_csv = OUT_DIR / f"{prefix}_roc_points.csv"
    roc_df.to_csv(roc_csv, index=False)

    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve")
    plt.tight_layout()
    pr_png = OUT_DIR / f"{prefix}_pr_curve.png"
    plt.savefig(pr_png, dpi=150)
    plt.close()

    pr_df = pd.DataFrame({"recall":rec, "precision":prec})
    pr_csv = OUT_DIR / f"{prefix}_pr_points.csv"
    pr_df.to_csv(pr_csv, index=False)

    return str(roc_png), str(pr_png)

def load_model_input_dim(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            return m.in_features
    # fallback si no encuentra (muy raro):
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True, help="MLflow run_id del modelo (phase2_run_09)")
    ap.add_argument("--prefix", default="phase2_run_09_v2", help="Prefijo para archivos de salida")
    args = ap.parse_args()

    # 1) Cargar meta
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    features = meta["features"]            # incluye Sex_num si lo agregaste
    winsor_caps = meta["winsor_caps"]
    log_cols = meta["log_cols"]
    random_state = meta["random_state"]
    test_size = meta["test_size"]

    # 2) Cargar datos base y reconstruir split
    df = pd.read_csv(DATA_CLEAN)
    y = df["SeriousDlqin2yrs"].astype(int).values
    X = df[features].copy()

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 3) Aplicar winsor + log (según caps y cols de meta) y scaler (fit en train)
    X_tr_w = winsorize_df(X_tr, winsor_caps)
    X_va_w = winsorize_df(X_va, winsor_caps)

    X_tr_t = log1p_df(X_tr_w, log_cols)
    X_va_t = log1p_df(X_va_w, log_cols)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_t.values.astype(np.float32))
    X_va_s = scaler.transform(X_va_t.values.astype(np.float32))

    # 4) Cargar modelo desde MLflow
    print(f"[INFO] Cargando modelo run_id={args.run_id} ...")
    model = mlflow.pytorch.load_model(f"runs:/{args.run_id}/model")
    in_dim = load_model_input_dim(model)
    print(f"[INFO] El modelo espera in_features = {in_dim}")

    if in_dim is not None and X_va_s.shape[1] != in_dim:
        raise ValueError(f"Dimensión de entrada no coincide: VALID tiene {X_va_s.shape[1]} columnas y el modelo espera {in_dim}.")

    # 5) Predicción y métricas
    yv_prob = predict_proba(model, X_va_s)
    base_metrics = compute_metrics_at_thr(y_va, yv_prob, thr=0.5)

    # umbral óptimo por F1
    prec, rec, thr_list = precision_recall_curve(y_va, yv_prob)
    f1_list = 2*prec*rec/(prec+rec+1e-9)
    # thr_list tiene longitud len(prec)-1; alineamos:
    best_idx = np.nanargmax(f1_list[:-1])
    best_thr = float(thr_list[best_idx])
    best_metrics = compute_metrics_at_thr(y_va, yv_prob, thr=best_thr)

    # 6) Guardar métricas y curvas
    prefix = args.prefix
    metrics_path = OUT_DIR / f"{prefix}_valid_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"thr_0_5": base_metrics, "thr_best_f1": best_metrics}, f, indent=2)

    roc_png, pr_png = plot_and_save_curves(y_va, yv_prob, prefix)

    # 7) Exportar predicciones VALID (con ambos umbrales)
    df_out = pd.DataFrame({
        "y_true": y_va,
        "y_prob": yv_prob,
        "y_pred_thr_0_5": (yv_prob >= 0.5).astype(int),
        "y_pred_best_f1": (yv_prob >= best_thr).astype(int),
    })
    df_out.to_csv(OUT_DIR / f"{prefix}_valid_predictions.csv", index=False)

    print("[OK] Métricas guardadas en:", metrics_path)
    print("[OK] Curva ROC:", roc_png)
    print("[OK] Curva PR:", pr_png)
    print(f"[OK] Umbral que maximiza F1: {best_thr:.4f}")
    print("[OK] Predicciones VALID:", OUT_DIR / f"{prefix}_valid_predictions.csv")

if __name__ == "__main__":
    main()
