import os
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

import mlflow
import mlflow.pytorch
import mlflow.pyfunc
from mlflow.pyfunc import PyFuncModel

import torch
from torch import nn
from sklearn.preprocessing import StandardScaler

try:
    import joblib
except Exception:
    joblib = None


# ==========================================================
# App Credit Risk MLP (versión en ESPAÑOL + UX + robustez)
# ==========================================================
# - Carga de modelo robusta: PyFunc o Torch (offline primero)
# - Scaler a prueba de fallos (scaler.pkl / train_clean.csv / Identity)
# - Sanitización de datos
# - Botón de plantilla CSV
# - Registro simple de inferencias a CSV

# ----------------------------
# Rutas/artefactos fijos
# ----------------------------
BASE_DIR = Path(".")
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
INFERENCE_DIR = REPORTS_DIR / "inference"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = REPORTS_DIR / "logs"

RUN_ID_PATH = MODELS_DIR / "run_id.txt"
THRESHOLD_PATH = MODELS_DIR / "threshold.txt"
COLUMNS_USED_PATH = MODELS_DIR / "columns_used.json"
TRAIN_CLEAN_PATH = DATA_DIR / "train_clean.csv"  # opcional: para ajustar scaler

# Columnas por defecto (si no existe columns_used.json)
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

# ----------------------------
# Utilidades básicas
# ----------------------------
def read_text(p: Path, default=None, to_float=False):
    if not p.exists():
        return default
    txt = p.read_text(encoding="utf-8").strip()
    return float(txt) if to_float else txt


def load_columns_used():
    if COLUMNS_USED_PATH.exists():
        data = json.loads(COLUMNS_USED_PATH.read_text(encoding="utf-8"))
        # Acepta {"features":[...]}, {"columns":[...]} o lista directa
        if isinstance(data, dict):
            if "features" in data:
                data = data["features"]
            elif "columns" in data:
                data = data["columns"]
        return data
    return DEFAULT_FEATURES


# ----------------------------
# Carga de modelo (robusta)
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_model(run_id: str):
    """
    Carga el modelo evitando el forward pickled de __main__.MLP.
    Prioridad:
    A) TorchScript exacto: models/mlflow_model/artifacts/checkpoints/best_phase2_run_09.pt
    B) Cualquier TorchScript .pt/.pth dentro de models/mlflow_model/artifacts/checkpoint(s)
    C) Cualquier TorchScript en todo models/mlflow_model
    D) mlflow.pytorch.load_model(...) (último recurso)
    """
    local_dir = MODELS_DIR / "mlflow_model"

    # A) ruta exacta que nos diste
    exact_ts = local_dir / "artifacts" / "checkpoints" / "best_phase2_run_09.pt"
    candidates = []
    if exact_ts.exists():
        candidates.append(exact_ts)

    # B) buscar en checkpoint y checkpoints
    for sub in ["checkpoint", "checkpoints"]:
        p = local_dir / "artifacts" / sub
        if p.exists():
            candidates += list(p.rglob("*.pt")) + list(p.rglob("*.pth"))

    # C) si aún nada, buscar en todo mlflow_model
    if local_dir.exists():
        candidates += [p for p in local_dir.rglob("*.pt")] + [p for p in local_dir.rglob("*.pth")]

    # Intentar TorchScript primero
    for p in candidates:
        try:
            m = torch.jit.load(str(p), map_location="cpu")
            m.eval()
            st.caption(f"🧠 Cargado TorchScript: `{p.as_posix()}`")
            return m
        except Exception:
            continue

    # D) Último recurso: export de MLflow como PyTorch (NO PyFunc)
    try:
        if (local_dir / "MLmodel").exists():
            m = mlflow.pytorch.load_model(str(local_dir))
            m.eval()
            m.to("cpu")
            st.warning("Se cargó el export PyTorch de MLflow (no TorchScript). "
                       "Si ves errores de `MLP + Tensor`, sube el TorchScript .pt.")
            return m
    except Exception:
        pass

    if run_id:
        try:
            m = mlflow.pytorch.load_model(f"runs:/{run_id}/model")
            m.eval()
            m.to("cpu")
            st.warning("Se cargó runs:/<run_id>/model (no TorchScript). "
                       "Si ves errores de `MLP + Tensor`, sube el TorchScript .pt.")
            return m
        except Exception:
            pass

    st.error(
        "No encontré un TorchScript (.pt/.pth) válido. "
        "Asegúrate de versionar `models/mlflow_model/artifacts/checkpoints/best_phase2_run_09.pt`."
    )
    raise RuntimeError("Modelo no disponible")


@st.cache_resource(show_spinner=False)
def fit_scaler_on_train(columns_order):
    """
    1) Carga models/scaler.pkl si existe.
    2) Si no, ajusta StandardScaler con data/train_clean.csv (si existe).
    3) Si no hay nada, usa IdentityScaler (no transforma).
    """
    # 1) scaler.pkl
    scaler_path = MODELS_DIR / "scaler.pkl"
    if scaler_path.exists() and joblib is not None:
        try:
            scaler = joblib.load(scaler_path)
            return scaler
        except Exception as e:
            st.warning(f"No se pudo cargar scaler.pkl: {e}. Intentando con train_clean.csv...")

    # 2) train_clean.csv
    if TRAIN_CLEAN_PATH.exists():
        df = pd.read_csv(TRAIN_CLEAN_PATH)
        X = df[[c for c in columns_order if c in df.columns]].copy()
        for c in columns_order:
            if c not in X.columns:
                X[c] = 0.0
        X = X[columns_order].fillna(0.0).replace([np.inf, -np.inf], 0.0)
        scaler = StandardScaler()
        scaler.fit(X.values.astype(np.float32))
        return scaler

    # 3) IdentityScaler
    st.warning("No se encontró 'models/scaler.pkl' ni 'data/train_clean.csv'. Usando IdentityScaler (sin estandarizar).")
    class IdentityScaler:
        def fit(self, X: Any): return self
        def transform(self, X: Any): return X
    return IdentityScaler()


def ensure_columns(df: pd.DataFrame, columns_order: list):
    """Reordena/crea columnas faltantes con 0.0 para que coincida con el entrenamiento."""
    out = df.copy()
    for c in columns_order:
        if c not in out.columns:
            out[c] = 0.0
    out = out[columns_order]
    return out


def to_sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _torch_try_forward(mod, x_t: "torch.Tensor"):
    """
    Intenta varias formas de llamar a forward, por si el modelo espera
    tensor directo, tupla, o dict con distintos nombres.
    Devuelve el output (tensor o similar) o relanza excepción si todas fallan.
    """
    with torch.no_grad():
        # 1) forward(x)
        try:
            out = mod(x_t)
            return out
        except Exception:
            pass

        # 2) forward((x,))
        try:
            out = mod((x_t,))
            return out
        except Exception:
            pass

        # 3) forward({'x': x})
        try:
            out = mod({"x": x_t})
            return out
        except Exception:
            pass

        # 4) forward({'inputs': x})
        try:
            out = mod({"inputs": x_t})
            return out
        except Exception:
            pass

        # 5) forward({'input': x})
        try:
            out = mod({"input": x_t})
            return out
        except Exception as e:
            raise e  # re-lanzamos la última para ver el motivo real


def predict_scores(model, X_np: np.ndarray, columns_order: list) -> np.ndarray:
    """
    Predicción SOLO con Torch (TorchScript o nn.Module).
    """
    x = torch.tensor(X_np.astype(np.float32), device="cpu")

    def to_probs(y: np.ndarray) -> np.ndarray:
        y = y.reshape(-1)
        if y.min() < 0.0 or y.max() > 1.0:
            y = 1.0 / (1.0 + np.exp(-y))
        return y

    try:
        model.eval()
    except Exception:
        pass
    try:
        model.to("cpu")
    except Exception:
        pass

    with torch.no_grad():
        # 1) forward(x)
        try:
            out = model(x)
            if isinstance(out, (list, tuple)):
                out = out[0]
            return to_probs(out.detach().cpu().numpy())
        except Exception as e1:
            err1 = e1
        # 2) forward((x,))
        try:
            out = model((x,))
            if isinstance(out, (list, tuple)):
                out = out[0]
            return to_probs(out.detach().cpu().numpy())
        except Exception as e2:
            err2 = e2
        # 3) forward({'x': x})
        try:
            out = model({"x": x})
            if isinstance(out, (list, tuple)):
                out = out[0]
            return to_probs(out.detach().cpu().numpy())
        except Exception as e3:
            err3 = e3
        # 4) forward({'inputs': x})
        try:
            out = model({"inputs": x})
            if isinstance(out, (list, tuple)):
                out = out[0]
            return to_probs(out.detach().cpu().numpy())
        except Exception as e4:
            err4 = e4

    raise TypeError(
        "No se pudo inferir con Torch. Es muy probable que el objeto cargado NO sea TorchScript. "
        "Sube y usa el archivo `.pt` TorchScript (ej: `models/mlflow_model/artifacts/checkpoints/best_phase2_run_09.pt`).\n\n"
        f"Últimos intentos:\n - {type(err1).__name__}: {err1}\n - {type(err2).__name__}: {err2}\n"
        f" - {type(err3).__name__}: {err3}\n - {type(err4).__name__}: {err4}"
    )

def log_inference(rows_df: pd.DataFrame, probs: np.ndarray, preds: np.ndarray, threshold: float):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out = rows_df.copy()
    out["prob_default"] = probs
    out["prediction"] = preds
    out["threshold"] = threshold
    out["timestamp"] = ts
    log_path = LOGS_DIR / "log_inference.csv"
    header = not log_path.exists()
    out.to_csv(log_path, mode="a", index=False, header=header)


# ----------------------------
# UI
# ----------------------------
def main():
    st.set_page_config(page_title="Riesgo Crediticio (MLP)", page_icon="💳", layout="centered")

    # Estilos mínimos
    st.markdown(
        """
        <style>
        .small { font-size: 0.85rem; color: #888; }
        .oktag { background:#1f6feb; color:white; padding:2px 6px; border-radius:6px; }
        .pill { background:#30363d; color:#c9d1d9; padding:2px 6px; border-radius:999px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("💳 Credit Risk MLP — Demo (ES)")

    # Cargar artefactos base
    run_id = read_text(RUN_ID_PATH, default=None)
    threshold = read_text(THRESHOLD_PATH, default=0.5, to_float=True)
    columns_order = load_columns_used()

    c0, c1, c2 = st.columns([1.2, 1, 1])
    with c0:
        st.markdown(f"**Run ID (prod):** <span class='pill'>{run_id or 'NO DEFINIDO'}</span>", unsafe_allow_html=True)
    with c1:
        st.markdown(f"**Umbral operativo:** <span class='pill'>{threshold:.4f}</span>", unsafe_allow_html=True)
    with c2:
        if st.button("🔄 Recargar artefactos"):
            # Limpiar cachés y recargar
            load_model.clear()
            fit_scaler_on_train.clear()
            st.rerun()

    # Nota: en modo offline podemos no requerir run_id, pero lo mostramos si existe.
    model = load_model(run_id or "")
    scaler = fit_scaler_on_train(columns_order)

    st.markdown("---")
    st.subheader("🔹 Ingreso manual (un registro)")
    st.caption("Completa los campos y presiona *Predecir riesgo*. Campos numéricos sin símbolo $.")

    # ----------------------------
    # 🔽 🔽 🔽 SECCIÓN: PREDICCIÓN MANUAL
    # ----------------------------
    with st.form("manual_form"):
        c1, c2 = st.columns(2)
        with c1:
            RevolvingUtilizationOfUnsecuredLines = st.number_input(
                "Utilización de líneas no garantizadas (0–1)", min_value=0.0, step=0.01, format="%0.4f", value=0.01
            )
            age = st.number_input("Edad (años)", min_value=0, step=1, value=35)
            NumberOfTime30_59 = st.number_input("Núm. veces atraso 30–59 días", min_value=0, step=1, value=2)
            DebtRatio = st.number_input("Relación de deuda (0–1)", min_value=0.0, step=0.01, format="%0.4f", value=0.01)
            MonthlyIncome = st.number_input("Ingreso mensual (USD)", min_value=0.0, step=100.0, value=25000.0)
            NumberOfOpenCreditLinesAndLoans = st.number_input("Líneas/préstamos abiertos", min_value=0, step=1, value=1)
        with c2:
            NumberOfTimes90DaysLate = st.number_input("Núm. veces atraso ≥90 días", min_value=0, step=1, value=1)
            NumberRealEstateLoansOrLines = st.number_input("Núm. hipotecas/líneas inmobiliarias", min_value=0, step=1, value=1)
            NumberOfTime60_89 = st.number_input("Núm. veces atraso 60–89 días", min_value=0, step=1, value=1)
            NumberOfDependents = st.number_input("Núm. de dependientes", min_value=0, step=1, value=2)
            sex = st.selectbox("Sexo", ["male", "female"], index=1)  # se mapeará a Sex_num
            Sex_num = 1.0 if sex == "male" else 0.0

        submitted = st.form_submit_button("Predecir riesgo")
        if submitted:
            row = {
                "RevolvingUtilizationOfUnsecuredLines": RevolvingUtilizationOfUnsecuredLines,
                "age": age,
                "NumberOfTime30-59DaysPastDueNotWorse": NumberOfTime30_59,
                "DebtRatio": DebtRatio,
                "MonthlyIncome": MonthlyIncome,
                "NumberOfOpenCreditLinesAndLoans": NumberOfOpenCreditLinesAndLoans,
                "NumberOfTimes90DaysLate": NumberOfTimes90DaysLate,
                "NumberRealEstateLoansOrLines": NumberRealEstateLoansOrLines,
                "NumberOfTime60-89DaysPastDueNotWorse": NumberOfTime60_89,
                "NumberOfDependents": NumberOfDependents,
                "Sex_num": Sex_num,
            }
            X = pd.DataFrame([row])
            # Sanitización + orden de columnas
            X = ensure_columns(X, columns_order)
            X = X.fillna(0.0).replace([np.inf, -np.inf], 0.0)
            # Escalado
            Xs = scaler.transform(X.values.astype(np.float32))
            # Predicción robusta
            probs = predict_scores(model, Xs, columns_order)
            prob = float(probs[0])
            yhat = int(prob >= threshold)

            st.success(f"**Probabilidad de morosidad (≥90 días):** {prob:.4f}")
            st.info(f"**Decisión (umbral {threshold:.4f}):** {'RIESGO (1)' if yhat==1 else 'NO RIESGO (0)'}")

            # Log de inferencia unitaria
            try:
                log_inference(pd.DataFrame([row]), np.array([prob]), np.array([yhat]), threshold)
            except Exception:
                pass

    st.markdown("---")
    st.subheader("🔹 Scoring por archivo (CSV)")
    st.write("Formato esperado (columnas, sin objetivo):")
    st.code(",".join(columns_order), language="text")

    # Botón de plantilla CSV
    tpl = pd.DataFrame(columns=columns_order)
    st.download_button(
        "Descargar plantilla CSV",
        data=tpl.to_csv(index=False).encode("utf-8"),
        file_name="plantilla_credit_mlp.csv",
        mime="text/csv"
    )

    file = st.file_uploader("Sube un CSV con las columnas de entrada", type=["csv"])
    if file is not None:
        try:
            df_in = pd.read_csv(file)
            # mapear sex si viniera como 'Sex'
            if "Sex" in df_in.columns and "Sex_num" not in df_in.columns:
                df_in["Sex_num"] = df_in["Sex"].map({"male": 1, "female": 0}).fillna(0).astype(float)

            # Sanitización + orden
            df_in = ensure_columns(df_in, columns_order)
            df_in = df_in.fillna(0.0).replace([np.inf, -np.inf], 0.0)
            # Escalado
            Xs = scaler.transform(df_in.values.astype(np.float32))
            # Predicción robusta
            probs = predict_scores(model, Xs, columns_order)
            preds = (probs >= threshold).astype(int)

            out = df_in.copy()
            out["prob_default"] = probs
            out["prediction"] = preds

            st.write("Vista previa (primeras 20 filas):")
            st.dataframe(out.head(20), use_container_width=True)

            INFERENCE_DIR.mkdir(parents=True, exist_ok=True)
            out_path = INFERENCE_DIR / "batch_predictions.csv"
            out.to_csv(out_path, index=False)
            st.success(f"Predicciones guardadas en: {out_path.as_posix()}")

            # Log de inferencias por lote
            try:
                log_inference(df_in, probs, preds, threshold)
            except Exception:
                pass

        except Exception as e:
            st.error(f"Error procesando el archivo: {e}")

    # Sidebar de ayuda
    with st.sidebar:
        st.header("ℹ️ Acerca de esta demo")
        st.markdown(
            """
            Demo académica para ilustrar un flujo de **riesgo crediticio** con MLP (PyTorch + MLflow).
            Esta aplicación no constituye recomendación financiera.
            """
        )
        st.markdown("**Soporte:** Verifica que existan:\n- `models/mlflow_model/MLmodel`\n- `models/run_id.txt`\n- `models/threshold.txt`\n- `models/columns_used.json`")


if __name__ == "__main__":
    # Si usas Streamlit Cloud con Secrets para MLflow remoto (no necesario en modo offline):
    # if hasattr(st, "secrets"): os.environ.update({k: str(v) for k, v in st.secrets.items()})
    main()
