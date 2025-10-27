# src/inference/predict_and_export_v2.py
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import mlflow.pytorch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


# ----------------------------
# Utilidades de preproceso
# ----------------------------
def winsorize_df(df: pd.DataFrame, caps: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Aplicar recorte p1-p99 con los límites ya guardados en meta."""
    out = df.copy()
    for c, lims in caps.items():
        low, high = lims["p1"], lims["p99"]
        if c in out.columns:
            out[c] = out[c].clip(lower=low, upper=high)
    return out

def log1p_df(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """log1p asegurando no-negativos."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = np.log1p(out[c].clip(lower=0))
    return out

def compute_metrics(y_true_np: np.ndarray, y_prob_np: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    """Métricas clasicas + ROC-AUC."""
    y_pred = (y_prob_np >= thr).astype(int)
    acc = accuracy_score(y_true_np, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true_np, y_pred, average="binary", zero_division=0
    )
    try:
        auc = roc_auc_score(y_true_np, y_prob_np)
    except ValueError:
        auc = float("nan")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc, "threshold": thr}

@torch.no_grad()
def predict_proba(model: nn.Module, X_np: np.ndarray, device: torch.device) -> np.ndarray:
    """Inferir probabilidades."""
    model.eval()
    X_t = torch.tensor(X_np, dtype=torch.float32, device=device)
    logits = model(X_t)
    prob = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
    return prob


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, help="Run ID de MLflow (modelo ganador)")
    parser.add_argument("--prefix", required=True, help="Prefijo para los archivos de salida")
    parser.add_argument("--threshold", type=float, default=0.5, help="Umbral para y_pred (default=0.5)")
    parser.add_argument("--experiment-name", default=None, help="(opcional) nombre del experimento, solo informativo")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Rutas base
    DATA_TRAIN_CLEAN = Path("data") / "train_clean.csv"
    DATA_TEST_CLEAN  = Path("data") / "test_clean.csv"     # opcional
    META_PATH        = Path("reports") / "preprocess_v2_meta.json"
    OUT_DIR          = Path("reports") / "inference"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Cargar modelo
    print(f"[INFO] Cargando modelo run_id={args.run_id} ...")
    model = mlflow.pytorch.load_model(f"runs:/{args.run_id}/model").to(device)

    # 2) Cargar meta del preproceso (guardado por train_mlp_v2.py)
    if not META_PATH.exists():
        raise FileNotFoundError(
            f"No encuentro {META_PATH}. Ejecuta el entrenamiento v2 para generar este meta."
        )

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    winsor_caps: Dict[str, Dict[str, float]] = meta["winsor_caps"]         # {col: {"p1":..., "p99":...}}
    log_cols: List[str] = meta["log_cols"]                                  # columnas a log1p
    features: List[str] = meta["features"]                                  # orden de features usadas
    random_state: int = meta.get("random_state", 42)
    test_size: float = meta.get("test_size", 0.2)

    # 3) Reconstruir VALID con el mismo split
    if not DATA_TRAIN_CLEAN.exists():
        raise FileNotFoundError(f"No encuentro {DATA_TRAIN_CLEAN}")

    df_train = pd.read_csv(DATA_TRAIN_CLEAN)
    if "SeriousDlqin2yrs" not in df_train.columns:
        raise ValueError("train_clean.csv no tiene la columna 'SeriousDlqin2yrs'.")

    y_all = df_train["SeriousDlqin2yrs"].astype(int).values
    X_all = df_train[features].copy()

    # Split reproducible
    X_tr_raw, X_va_raw, y_tr, y_va = train_test_split(
        X_all, y_all, test_size=test_size, random_state=random_state, stratify=y_all
    )

    # 4) Aplicar preproceso EXACTO:
    #    - winsor con límites guardados (calculados en el train original)
    #    - log1p en log_cols
    X_tr_w = winsorize_df(X_tr_raw, winsor_caps)
    X_va_w = winsorize_df(X_va_raw, winsor_caps)

    X_tr_t = log1p_df(X_tr_w, log_cols)
    X_va_t = log1p_df(X_va_w, log_cols)

    #    - escalar (ajustar scaler SOLO con TRAIN y aplicar a VALID/TEST)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_t.values.astype(np.float32))
    X_va_s = scaler.transform(X_va_t.values.astype(np.float32))

    # 5) Predicciones VALID
    print("[INFO] Cargando y alineando VALID ...")
    in_features = X_tr_s.shape[1]
    # Chequeo forma esperada por el modelo (no siempre es posible leerla; si falla, continuamos)
    try:
        first_linear = [m for m in model.modules() if isinstance(m, nn.Linear)][0]
        expected_in = first_linear.in_features
        print(f"[INFO] El modelo espera in_features = {expected_in}")
        if expected_in != in_features:
            raise ValueError(f"Mismatch de columnas: modelo espera {expected_in}, data tiene {in_features}")
    except Exception:
        pass

    yv_prob = predict_proba(model, X_va_s, device=device)
    val_metrics = compute_metrics(y_va.reshape(-1), yv_prob, thr=args.threshold)

    # Guardar VALID predicciones
    valid_df = pd.DataFrame({
        "y_true": y_va.reshape(-1),
        "y_prob": yv_prob,
        "y_pred": (yv_prob >= args.threshold).astype(int)
    })
    valid_csv = OUT_DIR / f"{args.prefix}_valid_predictions.csv"
    valid_df.to_csv(valid_csv, index=False)
    # Guardar métricas VALID
    valid_json = OUT_DIR / f"{args.prefix}_valid_metrics.json"
    with open(valid_json, "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=2)
    print(f"[OK] Guardado VALID: {valid_csv}")
    print(f"[OK] Guardado VALID metrics: {valid_json}")

    # 6) Predicciones TEST (si existe test_clean.csv)
    if DATA_TEST_CLEAN.exists():
        print("[INFO] Cargando y alineando TEST ...")
        df_test = pd.read_csv(DATA_TEST_CLEAN)

        # Algunos test no traen etiqueta
        y_test = df_test["SeriousDlqin2yrs"].values if "SeriousDlqin2yrs" in df_test.columns else None
        X_test_raw = df_test[features].copy()

        # aplicar los mismos caps / log / scaler (scaler ajustado SOLO con TRAIN)
        X_te_w = winsorize_df(X_test_raw, winsor_caps)
        X_te_t = log1p_df(X_te_w, log_cols)
        X_te_s = scaler.transform(X_te_t.values.astype(np.float32))

        yt_prob = predict_proba(model, X_te_s, device=device)

        test_out = pd.DataFrame({"y_prob": yt_prob, "y_pred": (yt_prob >= args.threshold).astype(int)})
        if y_test is not None and not pd.isna(y_test).all():
            # si tenemos etiquetas reales, incluimos y_true y métricas
            y_test_int = pd.Series(y_test).fillna(-1).astype(int).values
            test_out.insert(0, "y_true", y_test_int)
            if (y_test_int >= 0).any():
                # solo calcula métricas si hay etiquetas válidas (>=0)
                mask = y_test_int >= 0
                test_metrics = compute_metrics(y_test_int[mask], yt_prob[mask], thr=args.threshold)
                test_json = OUT_DIR / f"{args.prefix}_test_metrics.json"
                with open(test_json, "w", encoding="utf-8") as f:
                    json.dump(test_metrics, f, indent=2)
                print(f"[OK] Guardado TEST metrics: {test_json}")
        else:
            print("[INFO] TEST sin etiquetas; solo se guardan probabilidades y predicciones.")

        test_csv = OUT_DIR / f"{args.prefix}_test_predictions.csv"
        test_out.to_csv(test_csv, index=False)
        print(f"[OK] Guardado TEST: {test_csv}")
    else:
        print("[WARN] No encontré data/test_clean.csv; me salto el TEST.")

    # 7) Guardar columnas usadas (por trazabilidad)
    cols_json = OUT_DIR / f"{args.prefix}_columns_used.json"
    with open(cols_json, "w", encoding="utf-8") as f:
        json.dump({"features": features}, f, indent=2)
    print(f"[OK] Guardado orden de columnas: {cols_json}")


if __name__ == "__main__":
    main()
