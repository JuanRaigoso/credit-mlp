import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import mlflow
import mlflow.pytorch
import torch
from sklearn.preprocessing import StandardScaler

# ==========================================================
# App Credit Risk MLP (versión en ESPAÑOL + mejoras de UX)
# ==========================================================
# - Botón de recarga de artefactos
# - Textos y etiquetas en español
# - Registro simple de inferencias a CSV
# - Funciones cacheadas y main() para facilitar despliegue

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
TRAIN_CLEAN_PATH = DATA_DIR / "train_clean.csv"  # usaremos esto para ajustar el scaler

# Columnas por defecto (por si no existe columns_used.json)
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
# Utilidades
# ----------------------------

def read_text(p: Path, default=None, to_float=False):
    if not p.exists():
        return default
    txt = p.read_text(encoding="utf-8").strip()
    return float(txt) if to_float else txt


def load_columns_used():
    if COLUMNS_USED_PATH.exists():
        data = json.loads(COLUMNS_USED_PATH.read_text(encoding="utf-8"))
        # Acepta diferentes formatos: {"features": [...]}, {"columns": [...]}, o lista directa
        if isinstance(data, dict):
            if "features" in data:
                data = data["features"]
            elif "columns" in data:
                data = data["columns"]
        return data
    return DEFAULT_FEATURES


@st.cache_resource(show_spinner=False)
def load_model(run_id: str):
    """
    Intenta primero cargar un modelo MLflow empaquetado en 'models/mlflow_model'
    (ideal para despliegues sin 'mlruns'). Si no existe, carga por runs:/<run_id>/model.
    """
    local_mlflow_model = MODELS_DIR / "mlflow_model" / "MLmodel"
    try:
        if local_mlflow_model.exists():
            # Carga modelo desde la carpeta local que copiaste
            model = mlflow.pytorch.load_model(str(MODELS_DIR / "mlflow_model"))
        else:
            # Si no existe carpeta local, intenta cargar desde MLflow
            uri = f"runs:/{run_id}/model"
            model = mlflow.pytorch.load_model(uri)
        model.eval()
        return model
    except Exception as e:
        st.error(f"No se pudo cargar el modelo. Detalle: {e}")
        raise

@st.cache_resource(show_spinner=False)
def fit_scaler_on_train(columns_order):
    """
    Ajusta el StandardScaler sobre train_clean, aplicando el mismo orden de columnas.
    Nota: para mantener la app simple, hacemos un escalado estándar (sin winsor/log).
    """
    df = pd.read_csv(TRAIN_CLEAN_PATH)
    X = df[columns_order].copy()
    scaler = StandardScaler()
    scaler.fit(X.values.astype(np.float32))
    return scaler


def ensure_columns(df: pd.DataFrame, columns_order: list):
    """Reordena/crea columnas faltantes con 0.0 para que coincida con el entrenamiento."""
    out = df.copy()
    for c in columns_order:
        if c not in out.columns:
            out[c] = 0.0
    return out[columns_order]


def predict_proba_torch(model, X_np: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        x_t = torch.tensor(X_np.astype(np.float32))
        logits = model(x_t)
        prob = torch.sigmoid(logits).cpu().numpy().reshape(-1)
    return prob


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
            st.experimental_rerun()

    if run_id is None:
        st.error("No se encontró models/run_id.txt. Asegúrate de haber guardado el run ganador.")
        st.stop()

    model = load_model(run_id)
    scaler = fit_scaler_on_train(columns_order)

    st.markdown("---")
    st.subheader("🔹 Ingreso manual (un registro)")
    st.caption("Completa los campos y presiona *Predecir riesgo*. Campos numéricos sin símbolo $.")

    with st.form("manual_form"):
        c1, c2 = st.columns(2)
        with c1:
            RevolvingUtilizationOfUnsecuredLines = st.number_input(
                "Utilización de líneas no garantizadas (0–1)", min_value=0.0, step=0.01, format="%0.4f"
            )
            age = st.number_input("Edad (años)", min_value=0, step=1, value=35)
            NumberOfTime30_59 = st.number_input("Núm. veces atraso 30–59 días", min_value=0, step=1, value=0)
            DebtRatio = st.number_input("Relación de deuda (0–1)", min_value=0.0, step=0.01, format="%0.4f")
            MonthlyIncome = st.number_input("Ingreso mensual (USD)", min_value=0.0, step=100.0)
            NumberOfOpenCreditLinesAndLoans = st.number_input("Líneas/préstamos abiertos", min_value=0, step=1, value=3)
        with c2:
            NumberOfTimes90DaysLate = st.number_input("Núm. veces atraso ≥90 días", min_value=0, step=1, value=0)
            NumberRealEstateLoansOrLines = st.number_input("Núm. hipotecas/líneas inmobiliarias", min_value=0, step=1, value=0)
            NumberOfTime60_89 = st.number_input("Núm. veces atraso 60–89 días", min_value=0, step=1, value=0)
            NumberOfDependents = st.number_input("Núm. de dependientes", min_value=0, step=1, value=0)
            sex = st.selectbox("Sexo", ["male", "female"])  # se mapeará a Sex_num
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
            X = ensure_columns(X, columns_order)
            Xs = scaler.transform(X.values.astype(np.float32))
            prob = float(predict_proba_torch(model, Xs)[0])
            yhat = int(prob >= threshold)

            st.success(f"**Probabilidad de morosidad (≥90 días):** {prob:.4f}")
            st.info(f"**Decisión (umbral {threshold:.4f}):** {'RIESGO (1)' if yhat==1 else 'NO RIESGO (0)'}")

            # Log de inferencia unitaria
            try:
                log_inference(pd.DataFrame([row]), np.array([prob]), np.array([yhat]), threshold)
            except Exception as _:
                pass

    st.markdown("---")
    st.subheader("🔹 Scoring por archivo (CSV)")
    st.write("Formato esperado (columnas, sin objetivo):")
    st.code(",".join(columns_order), language="text")

    file = st.file_uploader("Sube un CSV con las columnas de entrada", type=["csv"])
    if file is not None:
        try:
            df_in = pd.read_csv(file)
            # mapear sex si viniera como 'Sex' (opcional)
            if "Sex" in df_in.columns and "Sex_num" not in df_in.columns:
                df_in["Sex_num"] = df_in["Sex"].map({"male": 1, "female": 0}).fillna(0).astype(float)

            df_in = ensure_columns(df_in, columns_order)
            Xs = scaler.transform(df_in.values.astype(np.float32))
            probs = predict_proba_torch(model, Xs)
            preds = (probs >= threshold).astype(int)

            out = df_in.copy()
            out["prob_default"] = probs
            out["prediction"] = preds

            st.write("Vista previa (primeras 20 filas):")
            st.dataframe(out.head(20))

            INFERENCE_DIR.mkdir(parents=True, exist_ok=True)
            out_path = INFERENCE_DIR / "batch_predictions.csv"
            out.to_csv(out_path, index=False)
            st.success(f"Predicciones guardadas en: {out_path.as_posix()}")

            # Log de inferencias por lote
            try:
                log_inference(df_in, probs, preds, threshold)
            except Exception as _:
                pass

        except Exception as e:
            st.error(f"Error procesando el archivo: {e}")

    # Sidebar de ayuda
    with st.sidebar:
        st.header("ℹ️ Ayuda")
        st.markdown(
            """
            **¿Qué significa que un despliegue *se duerma*?**
            
            Algunos servicios gratuitos apagan la app cuando no hay tráfico para ahorrar recursos. 
            Cuando vuelves a abrir la URL, tarda unos segundos en *despertar* (arranque en frío) y luego funciona normal. 
            
            Ventajas: es gratis o muy barato. 
            Contras: la primera carga puede tardar.
            """
        )
        st.markdown("**Soporte:** Si algo falla, revisa que existan:**\n- `models/run_id.txt`\n- `models/threshold.txt`\n- `models/columns_used.json`\n- `data/train_clean.csv`")


if __name__ == "__main__":
    main()
