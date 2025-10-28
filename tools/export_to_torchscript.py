# tools/export_to_torchscript.py
import argparse, json
from pathlib import Path
import numpy as np
import torch
import mlflow.pytorch

DEFAULT_FEATURES = [
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
    "Sex_num",
]

def load_n_features(models_dir: Path) -> int:
    cfg = models_dir / "columns_used.json"
    if cfg.exists():
        data = json.loads(cfg.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            feats = data.get("features") or data.get("columns")
        else:
            feats = data
        if isinstance(feats, list) and len(feats) > 0:
            return len(feats)
    return len(DEFAULT_FEATURES)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/mlflow_model", help="Ruta al export de MLflow (contiene MLmodel)")
    ap.add_argument("--out", default="models/mlflow_model/artifacts/checkpoints/best_phase2_run_09.pt",
                    help="Ruta de salida TorchScript (.pt)")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Cargar nn.Module desde export de MLflow (no PyFunc)
    print(f"[INFO] Cargando nn.Module desde: {model_dir}")
    model = mlflow.pytorch.load_model(str(model_dir))
    model.eval()
    model.to("cpu")

    # 2) Crear input de ejemplo para trace/script
    n_features = load_n_features(model_dir.parent)
    example = torch.zeros((1, n_features), dtype=torch.float32)

    # 3) Primero intentamos SCRIPT (mejor si hay control flow)
    ts = None
    try:
        print("[INFO] Intentando torch.jit.script(model)...")
        ts = torch.jit.script(model)
    except Exception as e_script:
        print(f"[WARN] script falló: {e_script}")
        print("[INFO] Intentando torch.jit.trace(model, example)...")
        with torch.no_grad():
            ts = torch.jit.trace(model, example, check_trace=False)

    # 4) Guardar TorchScript
    torch.jit.save(ts, str(out_path))
    print(f"[OK] TorchScript guardado en: {out_path}")

if __name__ == "__main__":
    main()
