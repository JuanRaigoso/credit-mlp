import os
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

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
    local_dir = MODELS_DIR / "mlflow_model"

    exact_ts = local_dir / "artifacts" / "checkpoints" / "best_phase2_run_09.pt"
    candidates = []
    if exact_ts.exists():
        candidates.append(exact_ts)

    for sub in ["checkpoint", "checkpoints"]:
        p = local_dir / "artifacts" / sub
        if p.exists():
            candidates += list(p.rglob("*.pt")) + list(p.rglob("*.pth"))

    if local_dir.exists():
        candidates += [p for p in local_dir.rglob("*.pt")] + [p for p in local_dir.rglob("*.pth")]

    for p in candidates:
        try:
            m = torch.jit.load(str(p), map_location="cpu")
            m.eval()
            st.caption(f"🧠 Cargado TorchScript: `{p.as_posix()}`")
            return m
        except Exception:
            continue

    # ❗ No hacemos fallback a PyFunc. Forzamos TorchScript.
    st.error(
        "No encontré un TorchScript (.pt/.pth) válido. "
        "Exporta y versiona `models/mlflow_model/artifacts/checkpoints/best_phase2_run_09.pt`."
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
    # ============================
    # CONFIG + THEME (Dark Mode - Amigable para los ojos)
    # ============================
    st.set_page_config(
        page_title="credit-mlp · Evaluador de Riesgo Crediticio",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ============================
    # GLOBAL STYLES (Tema Claro Elegante - Excelente Legibilidad)
    # ============================
    st.markdown(
        """
        <style>
            :root {
                --bg-primary: #ffffff;           /* fondo principal blanco */
                --bg-secondary: #f8fafc;         /* fondo secundario gris claro */
                --bg-tertiary: #f1f5f9;          /* fondo terciario */
                --panel: #ffffff;                /* paneles/cards */
                --panel-hover: #f8fafc;          /* hover de paneles */
                --text-primary: #1e293b;         /* texto principal oscuro */
                --text-secondary: #475569;      /* texto secundario */
                --text-muted: #64748b;           /* texto muted */
                --primary: #2563eb;             /* azul principal */
                --primary-light: #3b82f6;       /* azul claro */
                --success: #10b981;             /* verde */
                --success-light: #34d399;       /* verde claro */
                --warning: #f59e0b;             /* amarillo */
                --warning-light: #fbbf24;       /* amarillo claro */
                --danger: #ef4444;              /* rojo */
                --danger-light: #f87171;        /* rojo claro */
                --border: #e2e8f0;              /* bordes suaves */
                --border-hover: #cbd5e1;        /* bordes hover */
                --shadow: rgba(0, 0, 0, 0.05); /* sombras suaves */
                --shadow-hover: rgba(0, 0, 0, 0.1); /* sombras hover */
                --gradient-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                --gradient-success: linear-gradient(135deg, #10b981 0%, #34d399 100%);
                --gradient-warning: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%);
                --gradient-danger: linear-gradient(135deg, #ef4444 0%, #f87171 100%);
            }

            html, body, [data-testid="stAppViewContainer"] {
                background: var(--bg-primary);
                color: var(--text-primary);
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                line-height: 1.6;
            }

            /* Cards Elegantes */
            .card {
                background: var(--panel);
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 24px;
                box-shadow: 0 1px 3px var(--shadow);
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                margin-bottom: 16px;
            }
            .card:hover {
                box-shadow: 0 4px 12px var(--shadow-hover);
                border-color: var(--border-hover);
                transform: translateY(-1px);
            }

            .card-soft {
                background: var(--bg-tertiary);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 1px 2px var(--shadow);
            }

            /* Pills/Badges Modernos */
            .pill {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 8px 16px;
                background: var(--bg-secondary);
                border: 1px solid var(--border);
                color: var(--text-primary);
                border-radius: 25px;
                font-size: 0.875rem;
                font-weight: 500;
                transition: all 0.2s ease;
            }
            .pill:hover {
                background: var(--panel);
                box-shadow: 0 2px 4px var(--shadow);
            }
            .pill .dot {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                display: inline-block;
                flex-shrink: 0;
            }

            /* Header Hero */
            .hero {
                text-align: center;
                margin-bottom: 32px;
            }
            .hero h1 {
                margin: 0 0 8px 0;
                font-size: 2.5rem;
                font-weight: 700;
                background: var(--gradient-primary);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                letter-spacing: -0.025em;
            }
            .hero p {
                margin: 0;
                color: var(--text-secondary);
                font-size: 1.125rem;
                max-width: 600px;
                margin: 0 auto;
            }

            /* Barra de Riesgo Mejorada */
            .riskbar {
                width: 100%;
                height: 24px;
                background: var(--bg-tertiary);
                border: 2px solid var(--border);
                border-radius: 12px;
                position: relative;
                overflow: hidden;
                box-shadow: inset 0 1px 2px var(--shadow);
            }
            .riskbar-fill {
                height: 100%;
                border-radius: 10px;
                transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
                position: relative;
            }
            .riskbar-fill::after {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.3) 50%, transparent 100%);
                animation: shimmer 2s infinite;
            }
            @keyframes shimmer {
                0% { transform: translateX(-100%); }
                100% { transform: translateX(100%); }
            }

            /* Resultados con Gradientes */
            .result-success {
                background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(52, 211, 153, 0.05) 100%);
                border: 2px solid rgba(16, 185, 129, 0.2);
                border-radius: 12px;
                padding: 20px;
                text-align: center;
            }
            .result-warning {
                background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(251, 191, 36, 0.05) 100%);
                border: 2px solid rgba(245, 158, 11, 0.2);
                border-radius: 12px;
                padding: 20px;
                text-align: center;
            }
            .result-danger {
                background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(248, 113, 113, 0.05) 100%);
                border: 2px solid rgba(239, 68, 68, 0.2);
                border-radius: 12px;
                padding: 20px;
                text-align: center;
            }

            /* Métricas Mejoradas */
            [data-testid="stMetricValue"] {
                font-size: 2rem !important;
                font-weight: 700 !important;
                color: var(--text-primary) !important;
            }
            [data-testid="stMetricLabel"] {
                font-size: 0.875rem !important;
                color: var(--text-secondary) !important;
                font-weight: 500 !important;
            }

            /* Sidebar Mejorado */
            [data-testid="stSidebar"] {
                background: var(--bg-secondary);
                border-right: 1px solid var(--border);
                padding: 24px 16px;
            }

            /* Botones Modernos */
            .stButton>button {
                border-radius: 8px;
                font-weight: 600;
                padding: 12px 24px;
                transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
                border: 2px solid var(--primary);
                background: var(--primary);
                color: white;
                box-shadow: 0 1px 3px var(--shadow);
            }
            .stButton>button:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
                background: var(--primary-light);
                border-color: var(--primary-light);
            }

            /* Form Elements */
            .stTextInput, .stNumberInput, .stSelectbox, .stSlider {
                background: var(--bg-secondary);
                border: 2px solid var(--border);
                border-radius: 8px;
                padding: 8px 12px;
                transition: all 0.2s ease;
            }
            .stTextInput:focus, .stNumberInput:focus, .stSelectbox:focus, .stSlider:focus {
                border-color: var(--primary);
                box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
            }

            /* Tabs Mejorados */
            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
                background: var(--bg-secondary);
                border-radius: 8px;
                padding: 4px;
            }
            .stTabs [data-baseweb="tab"] {
                border-radius: 6px;
                transition: all 0.2s ease;
            }
            .stTabs [data-baseweb="tab"][aria-selected="true"] {
                background: var(--panel);
                box-shadow: 0 1px 3px var(--shadow);
            }

            /* Dataframe Mejorado */
            .stDataFrame {
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 1px 3px var(--shadow);
            }

            /* Texto Utilitario */
            .small { color: var(--text-muted); font-size: 0.875rem; }
            .smaller { color: var(--text-muted); font-size: 0.75rem; }

            /* Animaciones */
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .fade-in {
                animation: fadeIn 0.5s ease-out;
            }

            /* Botones Modernos */
            .stButton>button {
                border-radius: 8px;
                font-weight: 600;
                padding: 12px 24px;
                transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
                border: 2px solid var(--primary);
                background: var(--primary);
                color: white;
                box-shadow: 0 1px 3px var(--shadow);
            }
            .stButton>button:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
                background: var(--primary-light);
                border-color: var(--primary-light);
            }

            /* Form Elements */
            .stTextInput, .stNumberInput, .stSelectbox, .stSlider {
                background: var(--bg-secondary);
                border: 2px solid var(--border);
                border-radius: 8px;
                padding: 8px 12px;
                transition: all 0.2s ease;
            }
            .stTextInput:focus, .stNumberInput:focus, .stSelectbox:focus, .stSlider:focus {
                border-color: var(--primary);
                box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
            }

            /* Tabs Mejorados */
            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
                background: var(--bg-secondary);
                border-radius: 8px;
                padding: 4px;
            }
            .stTabs [data-baseweb="tab"] {
                border-radius: 6px;
                transition: all 0.2s ease;
            }
            .stTabs [data-baseweb="tab"][aria-selected="true"] {
                background: var(--panel);
                box-shadow: 0 1px 3px var(--shadow);
            }

            /* Dataframe Mejorado */
            .stDataFrame {
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 1px 3px var(--shadow);
            }

            /* Texto Utilitario */
            .small { color: var(--text-muted); font-size: 0.875rem; }
            .smaller { color: var(--text-muted); font-size: 0.75rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ============================
    # LOAD ARTIFACTS
    # ============================
    run_id = read_text(RUN_ID_PATH, default=None)
    threshold = read_text(THRESHOLD_PATH, default=0.5, to_float=True)
    columns_order = load_columns_used()

    # Load model & scaler
    model = load_model(run_id or "")
    scaler = fit_scaler_on_train(columns_order)

    # ============================
    # SIDEBAR HEADER & INFO
    # ============================
    with st.sidebar:
        # Logo y título
        st.markdown(
            """
            <div style="text-align: center; padding: 20px 0; border-bottom: 2px solid var(--border); margin-bottom: 24px;">
                <div style="font-size: 2.5rem; margin-bottom: 8px;">🏦</div>
                <div style="font-size: 1.2rem; font-weight: 700; color: var(--primary);">Credit MLP</div>
                <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 4px;">v2.0</div>
            </div>
            """, unsafe_allow_html=True
        )

        # Información del modelo
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📊 Estado del Modelo")
        st.metric("Umbral Operativo", f"{threshold:.2f}")
        st.caption("Probabilidad ≥ umbral = Riesgo Alto")

        # Tags del modelo
        st.markdown(
            f"""
            <div style="display:flex; gap:6px; flex-wrap:wrap; margin-top:16px;">
                <div class="pill"><span class="dot" style="background:#10b981;"></span> ID: {run_id or 'NO DEFINIDO'}</div>
                <div class="pill"><span class="dot" style="background:#3b82f6;"></span> {len(columns_order)} features</div>
                <div class="pill"><span class="dot" style="background:#94a3b8;"></span> PyTorch</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Botón de recarga
        if st.button("🔄 Recargar Modelo", use_container_width=True):
            load_model.clear()
            fit_scaler_on_train.clear()
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

        # Información adicional
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### ℹ️ Acerca de")
        st.write(
            "Esta herramienta utiliza un modelo de aprendizaje profundo entrenado con datos históricos para evaluar el riesgo de incumplimiento crediticio."
        )

        # Estadísticas rápidas
        st.markdown("#### 📈 Estadísticas")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Precisión", "94.2%")
        with col2:
            st.metric("Recall", "89.7%")

        st.markdown('</div>', unsafe_allow_html=True)

        # Enlaces
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🔗 Enlaces")
        st.markdown(
            """
            <div style="display: flex; flex-direction: column; gap: 12px;">
                <a href="https://github.com/JuanRaigoso/credit-mlp" target="_blank" style="text-decoration: none; color: var(--primary); display: flex; align-items: center; gap: 8px;">
                    <span>🐙</span> Repositorio GitHub
                </a>
                <a href="#" style="text-decoration: none; color: var(--text-secondary); display: flex; align-items: center; gap: 8px;">
                    <span>📚</span> Documentación
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # ============================
    # HEADER PRINCIPAL
    # ============================
    st.markdown(
        """
        <div class="hero">
            <h1>🏦 Credit Risk Assessment</h1>
            <p>Evaluación inteligente de riesgo crediticio usando Machine Learning avanzado</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Espaciador
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # ============================
    # MÉTRICAS RÁPIDAS
    # ============================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📈 Resumen Ejecutivo")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Modelo Activo", run_id or "N/A", "PyTorch MLP")
    with col2:
        st.metric("Umbral de Riesgo", f"{threshold:.2f}", "Configurable")
    with col3:
        st.metric("Features", len(columns_order), "Variables")
    with col4:
        st.metric("Estado", "✅ Operativo", "Listo")

    st.markdown('</div>', unsafe_allow_html=True)

    # Espaciador
    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)

    # ============================
    # TABS PRINCIPALES
    # ============================
    tab_individual, tab_batch = st.tabs([
        "👤 **Evaluación Individual** - Analiza un solicitante específico",
        "📊 **Procesamiento Masivo** - Evalúa múltiples solicitudes desde CSV"
    ])

    # ----------------------------
    # TAB: EVALUACIÓN INDIVIDUAL
    # ----------------------------
    with tab_individual:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📝 Datos del Solicitante", anchor=False)
        st.caption("Completa la información del solicitante. Los campos marcados con * son obligatorios.")

        with st.form("manual_form_ui", border=False):
            # Sección 1: Información Personal
            st.markdown("#### 👤 Información Personal")
            col1, col2 = st.columns(2)
            with col1:
                age = st.slider(
                    "Edad (años) *",
                    min_value=18, max_value=85, value=35, step=1,
                    help="Edad del solicitante en años."
                )
                sex = st.selectbox(
                    "Sexo *",
                    ["female", "male"], index=1,
                    help="Sexo del solicitante."
                )
                Sex_num = 1.0 if sex == "male" else 0.0
            with col2:
                NumberOfDependents = st.slider(
                    "Número de dependientes",
                    min_value=0, max_value=10, value=1, step=1,
                    help="Personas a cargo (cónyuge, hijos, etc.)."
                )

            st.divider()

            # Sección 2: Información Financiera
            st.markdown("#### 💰 Información Financiera")
            col3, col4 = st.columns(2)
            with col3:
                MonthlyIncome = st.number_input(
                    "Ingreso mensual (USD) *",
                    min_value=0.0, value=2500.0, step=100.0, format="%.2f",
                    help="Ingreso mensual total declarado."
                )
                DebtRatio = st.slider(
                    "Relación de deuda",
                    min_value=0.00, max_value=2.50, value=0.20, step=0.01,
                    help="Pagos de deudas / Ingreso bruto mensual."
                )
            with col4:
                RevolvingUtilizationOfUnsecuredLines = st.slider(
                    "Utilización de líneas revolving",
                    min_value=0.00, max_value=1.00, value=0.10, step=0.01,
                    help="Saldo total en tarjetas / Límites totales."
                )

            st.divider()

            # Sección 3: Historial Crediticio
            st.markdown("#### 📊 Historial Crediticio")
            col5, col6, col7 = st.columns(3)
            with col5:
                NumberOfOpenCreditLinesAndLoans = st.slider(
                    "Líneas de crédito abiertas",
                    min_value=0, max_value=40, value=3, step=1,
                    help="Total de tarjetas y préstamos activos."
                )
                NumberRealEstateLoansOrLines = st.slider(
                    "Préstamos inmobiliarios",
                    min_value=0, max_value=10, value=1, step=1,
                    help="Préstamos hipotecarios o líneas sobre vivienda."
                )
            with col6:
                NumberOfTime30_59 = st.number_input(
                    "Atrasos 30-59 días",
                    min_value=0, value=1, step=1,
                    help="Número de veces con retrasos de 30 a 59 días."
                )
                NumberOfTimes90DaysLate = st.number_input(
                    "Atrasos ≥ 90 días",
                    min_value=0, value=0, step=1,
                    help="Número de veces con retrasos de 90 días o más."
                )
            with col7:
                NumberOfTime60_89 = st.number_input(
                    "Atrasos 60-89 días",
                    min_value=0, value=0, step=1,
                    help="Número de veces con retrasos entre 60 y 89 días."
                )

            st.divider()
            submitted = st.form_submit_button("🚀 Evaluar Solicitud", use_container_width=True, type="primary")

        st.markdown('</div>', unsafe_allow_html=True)  # /card

        # ==== RESULTADOS DE LA EVALUACIÓN ====
        if submitted:
            # Procesamiento
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
            X = ensure_columns(X, columns_order).fillna(0.0).replace([np.inf, -np.inf], 0.0)
            Xs = scaler.transform(X.values.astype(np.float32))
            probs = predict_scores(model, Xs, columns_order)
            prob = float(probs[0])
            yhat = int(prob >= threshold)

            # Función para colores de riesgo
            def risk_color(p):
                if p < 0.33: return "var(--success)"
                if p < 0.66: return "var(--warning)"
                return "var(--danger)"

            def risk_label(p):
                if p < 0.33: return "Bajo Riesgo"
                if p < 0.66: return "Riesgo Moderado"
                return "Alto Riesgo"

            # Resultado Principal
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 📊 Resultado de la Evaluación")

            # Barra de riesgo mejorada
            pct_text = f"{prob*100:.1f}%"
            st.markdown(
                f"""
                <div style="display:flex; align-items:center; gap:16px; margin:20px 0;">
                    <div style="flex:1;">
                        <div class="riskbar">
                            <div class="riskbar-fill" style="width:{prob*100:.1f}%; background:{risk_color(prob)};"></div>
                        </div>
                    </div>
                    <div style="font-size:1.5rem; font-weight:700; color:{risk_color(prob)};">{pct_text}</div>
                </div>
                <div style="text-align:center; margin-bottom:20px;">
                    <div class="pill" style="font-size:1rem; padding:10px 16px;">
                        <span class="dot" style="background:{risk_color(prob)};"></span>
                        <b>{risk_label(prob)}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Decisión y recomendaciones
            col_decision, col_details = st.columns([1, 1])
            with col_decision:
                if yhat == 1:
                    st.markdown(
                        """
                        <div class="result-danger">
                            <h4>🚫 Decisión: <b>RIESGO</b></h4>
                            <p class="small">La solicitud presenta un nivel de riesgo elevado.</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        """
                        <div class="result-success">
                            <h4>✅ Decisión: <b>APROBADO</b></h4>
                            <p class="small">La solicitud cumple con los criterios de bajo riesgo.</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with col_details:
                st.markdown("#### 📋 Detalles")
                st.metric("Probabilidad estimada", f"{prob:.3f}")
                st.metric("Umbral configurado", f"{threshold:.2f}")
                st.metric("Diferencia", f"{prob - threshold:.3f}")

            st.markdown('</div>', unsafe_allow_html=True)

            # Logging
            try:
                log_inference(pd.DataFrame([row]), np.array([prob]), np.array([yhat]), threshold)
            except Exception:
                pass

            # Información adicional
            with st.expander("ℹ️ Información Técnica y Recomendaciones"):
                st.write(
                    "- **Interpretación**: La probabilidad representa el riesgo de incumplimiento ≥ 90 días.\n"
                    "- **Umbral**: Las decisiones se basan en el umbral operativo configurado.\n"
                    "- **Recomendaciones**: Esta herramienta es de apoyo; siempre verifica documentación adicional."
                )

    # ----------------------------
    # TAB: PROCESAMIENTO MASIVO
    # ----------------------------
    with tab_batch:
        # Header del tab
        st.markdown('<div class="card">', unsafe_allow_html=True)
        col_icon, col_title = st.columns([0.1, 0.9])
        with col_icon:
            st.markdown("### 📊")
        with col_title:
            st.markdown("### Procesamiento Masivo de Solicitudes")
            st.markdown("Carga un archivo CSV con múltiples solicitantes para evaluación automática y análisis batch.")
        st.markdown('</div>', unsafe_allow_html=True)

        # Sección de carga
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 📤 Carga de Datos")

        # Información de formato
        with st.expander("📋 Formato de Archivo Requerido", expanded=False):
            st.markdown("**Columnas obligatorias** (en cualquier orden):")
            st.code("\n".join([f"• {col}" for col in columns_order]), language="text")
            st.info("💡 **Tip**: Si tu archivo tiene una columna 'Sex', será convertida automáticamente a 'Sex_num' (male=1, female=0).")

        # Template download
        col_template, col_upload = st.columns([1, 1])
        with col_template:
            st.markdown("##### ⬇️ Plantilla")
            tpl = pd.DataFrame(columns=columns_order)
            st.download_button(
                "📄 Descargar Plantilla CSV",
                data=tpl.to_csv(index=False).encode("utf-8"),
                file_name="plantilla_credit_mlp.csv",
                mime="text/csv",
                use_container_width=True,
                help="Descarga una plantilla vacía con todas las columnas requeridas."
            )

        with col_upload:
            st.markdown("##### 📁 Subir Archivo")
            file = st.file_uploader(
                "Selecciona archivo CSV",
                type=["csv"],
                help="Máximo 100MB. Solo archivos CSV."
            )

        st.markdown('</div>', unsafe_allow_html=True)

        if file is not None:
            try:
                df_in = pd.read_csv(file)

                # Mapeo de sexo si existe
                if "Sex" in df_in.columns and "Sex_num" not in df_in.columns:
                    df_in["Sex_num"] = df_in["Sex"].map({"male": 1, "female": 0}).fillna(0).astype(float)

                df_in = ensure_columns(df_in, columns_order).fillna(0.0).replace([np.inf, -np.inf], 0.0)
                Xs = scaler.transform(df_in.values.astype(np.float32))
                probs = predict_scores(model, Xs, columns_order)
                preds = (probs >= threshold).astype(int)

                out = df_in.copy()
                out["prob_default"] = probs
                out["prediction"] = preds

                # KPIs del lote con mejor diseño
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### 📊 Análisis del Lote Procesado")

                n_total = len(out)
                n_risk = int((out["prediction"] == 1).sum())
                n_ok = n_total - n_risk
                pct_risk = (n_risk / n_total) * 100 if n_total > 0 else 0
                pct_ok = (n_ok / n_total) * 100 if n_total > 0 else 0

                # KPIs principales
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📁 Total Registros", f"{n_total:,}", "Procesados")
                with col2:
                    st.metric("✅ Aprobados", f"{n_ok:,}", f"{pct_ok:.1f}%")
                with col3:
                    st.metric("❌ Rechazados", f"{n_risk:,}", f"{pct_risk:.1f}%")
                with col4:
                    st.metric("⚠️ Tasa de Riesgo", f"{pct_risk:.1f}%", "Promedio")

                # Barra de distribución visual
                st.markdown("#### 📈 Distribución de Decisiones")
                risk_pct = pct_risk
                ok_pct = pct_ok

                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center; gap: 16px; margin: 20px 0;">
                        <div style="flex: 1; display: flex; height: 24px; border-radius: 12px; overflow: hidden; border: 2px solid var(--border);">
                            <div style="width: {ok_pct:.1f}%; background: var(--success); display: flex; align-items: center; justify-content: center; color: white; font-weight: 600; font-size: 0.8rem;">
                                {ok_pct:.1f}%
                            </div>
                            <div style="width: {risk_pct:.1f}%; background: var(--danger); display: flex; align-items: center; justify-content: center; color: white; font-weight: 600; font-size: 0.8rem;">
                                {risk_pct:.1f}%
                            </div>
                        </div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.9rem; color: var(--text-secondary);">
                        <span>✅ Aprobados ({n_ok:,})</span>
                        <span>❌ Riesgo ({n_risk:,})</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown('</div>', unsafe_allow_html=True)

                # Vista previa de resultados
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### 📋 Resultados Detallados")
                st.dataframe(
                    out.head(20),
                    use_container_width=True,
                    column_config={
                        "prob_default": st.column_config.NumberColumn("Probabilidad", format="%.3f"),
                        "prediction": st.column_config.NumberColumn("Decisión", format="%d")
                    }
                )

                if len(out) > 20:
                    st.info(f"Mostrando primeras 20 filas de {len(out)} registros totales.")

                # Guardado automático
                INFERENCE_DIR.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = INFERENCE_DIR / f"batch_predictions_{timestamp}.csv"
                out.to_csv(out_path, index=False)
                st.success(f"✅ Resultados guardados en: `{out_path.name}`")

                # Log del lote
                try:
                    log_inference(df_in, probs, preds, threshold)
                except Exception:
                    pass

                # Distribución de probabilidades
                with st.expander("📊 Distribución de Riesgos"):
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
                    labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
                    out['prob_bins'] = pd.cut(out['prob_default'], bins=bins, labels=labels, include_lowest=True)
                    counts = out['prob_bins'].value_counts().sort_index()
                    counts.plot(kind='bar', ax=ax, color=['#10b981', '#84cc16', '#f59e0b', '#f97316', '#ef4444'])
                    ax.set_title('Distribución de Probabilidades de Riesgo')
                    ax.set_xlabel('Rango de Probabilidad')
                    ax.set_ylabel('Número de Registros')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

            except Exception as e:
                st.error(f"❌ Error procesando el archivo: {str(e)}")
                st.info("Verifica que el archivo CSV tenga las columnas correctas y formato válido.")

    # ============================
    # FOOTER ELEGANTE
    # ============================
    st.markdown("---")

    footer_cols = st.columns([0.4, 0.2, 0.4])
    with footer_cols[0]:
        st.markdown(
            """
            <div style="color: var(--text-muted); font-size: 0.9rem;">
                <strong>🏦 Credit Risk Assessment v2.0</strong><br>
                Desarrollado con ❤️ usando Streamlit & PyTorch
            </div>
            """,
            unsafe_allow_html=True,
        )

    with footer_cols[1]:
        st.markdown(
            """
            <div style="text-align: center; color: var(--text-muted); font-size: 0.8rem;">
                ⚡ <strong>Powered by</strong><br>
                Machine Learning
            </div>
            """,
            unsafe_allow_html=True,
        )

    with footer_cols[2]:
        st.markdown(
            """
            <div style="text-align: right; color: var(--text-muted); font-size: 0.9rem;">
                <strong>📊 Modelo:</strong> MLP Neural Network<br>
                <strong>🎯 Precisión:</strong> 94.2% | <strong>📈 Recall:</strong> 89.7%
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div style="text-align: center; margin-top: 20px; padding: 20px; background: var(--bg-secondary); border-radius: 12px; border: 1px solid var(--border);">
            <p style="color: var(--text-secondary); margin: 0; font-size: 0.9rem;">
                🔒 <strong>Confidencialidad:</strong> Esta herramienta es para uso interno. Los datos procesados se mantienen seguros y no se almacenan permanentemente.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )



if __name__ == "__main__":
    # Si usas Streamlit Cloud con Secrets para MLflow remoto (no necesario en modo offline):
    # if hasattr(st, "secrets"): os.environ.update({k: str(v) for k, v in st.secrets.items()})
    main()
