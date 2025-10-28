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
    # CONFIG + THEME (Dark)
    # ============================
    st.set_page_config(
        page_title="credit-mlp · Evaluador de Riesgo Crediticio",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ============================
    # GLOBAL STYLES (Dark UI)
    # ============================
    st.markdown(
        """
        <style>
            :root {
                --bg: #9BA2BA;           /* fondo app */
                --panel: #121a2b;        /* paneles/cards */
                --panel-2: #0e1626;      /* paneles secundarios */
                --text: #0D0D0D;         /* texto principal */
                --muted: #1F2F4D;        /* texto secundario */
                --primary: #60a5fa;      /* azul */
                --success: #34d399;      /* verde */
                --warning: #fbbf24;      /* amarillo */
                --danger: #f87171;       /* rojo */
                --border: #1f2a44;       /* bordes */
                --chip-bg: #0d2547;      /* fondo chip */
            }

            html, body, [data-testid="stAppViewContainer"] {
                background: var(--bg);
                color: var(--text);
                font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, "Helvetica Neue", Arial;
            }

            /* Cards */
            .card {
                background: var(--panel);
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 18px 18px;
            }
            .card-soft {
                background: var(--panel-2);
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 18px 18px;
            }

            /* Chips / Pills */
            .pill {
                display: inline-flex; align-items:center; gap:8px;
                padding: 6px 10px;
                background: var(--chip-bg);
                border: 1px solid var(--border);
                color: var(--text);
                border-radius: 999px;
                font-size: 0.85rem;
            }
            .pill .dot { width:8px; height:8px; border-radius:999px; display:inline-block; }

            /* Headline */
            .hero h1 { margin: 0; font-size: 1.85rem; font-weight: 700; letter-spacing: .2px; }
            .hero p  { margin: 4px 0 0 0; color: var(--muted); }

            /* Progress bar custom (risk bar) */
            .riskbar {
                width: 100%; height: 18px;
                background: linear-gradient(90deg, rgba(52,211,153,0.15), rgba(96,165,250,0.15), rgba(248,113,113,0.15));
                border: 1px solid var(--border);
                border-radius: 99px;
                position: relative;
                overflow: hidden;
            }
            .riskbar-fill {
                height: 100%;
                border-radius: 99px;
                transition: width .35s ease;
            }

            /* Badges result */
            .result-ok     { background: rgba(52,211,153,0.12); border: 1px solid rgba(52,211,153,0.35); }
            .result-warn   { background: rgba(251,191,36,0.12); border: 1px solid rgba(251,191,36,0.35); }
            .result-danger { background: rgba(248,113,113,0.12); border: 1px solid rgba(248,113,113,0.35); }

            /* Tables (make header sticky look darker) */
            .stDataFrame, .stTable { filter: saturate(1.02) contrast(1.02); }
            .small   { color: var(--muted); font-size: 0.9rem; }
            .smaller { color: var(--muted); font-size: 0.8rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ============================
    # HEADER
    # ============================
    c_logo, c_title, c_kpis = st.columns([0.1, 0.65, 0.25], gap="small")
    with c_logo:
        st.markdown(
            """
            <div class="pill">
                <span class="dot" style="background:#60a5fa;"></span>
                credit-mlp
            </div>
            """, unsafe_allow_html=True
        )
    with c_title:
        st.markdown(
            """
            <div class="hero">
                <h1>Evaluador de Riesgo Crediticio</h1>
                <p>Modelo MLP (PyTorch/MLflow) para estimar probabilidad de morosidad ≥ 90 días.</p>
            </div>
            """, unsafe_allow_html=True
        )

    # ============================
    # LOAD ARTIFACTS (same logic)
    # ============================
    run_id = read_text(RUN_ID_PATH, default=None)
    threshold = read_text(THRESHOLD_PATH, default=0.5, to_float=True)
    columns_order = load_columns_used()

    with c_kpis:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.metric("Umbral operativo", f"{threshold:.2f}")
        st.caption("Las solicitudes con probabilidad ≥ umbral se clasifican como **Riesgo**.")
        reload_btn = st.button("🔄 Recargar artefactos", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        if reload_btn:
            load_model.clear()
            fit_scaler_on_train.clear()
            st.rerun()

    # Load model & scaler (no changes)
    model = load_model(run_id or "")
    scaler = fit_scaler_on_train(columns_order)

    # MODEL TAGS ROW
    st.markdown(
        f"""
        <div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:8px;">
            <div class="pill"><span class="dot" style="background:#34d399;"></span> Modelo: <b>{run_id or 'NO DEFINIDO'}</b></div>
            <div class="pill"><span class="dot" style="background:#60a5fa;"></span> Features: <b>{len(columns_order)}</b></div>
            <div class="pill"><span class="dot" style="background:#9aa4b2;"></span> TorchScript</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")

    # ============================
    # TABS
    # ============================
    tab1, tab2 = st.tabs(["🧮 Simulador individual", "📁 Scoring por archivo (CSV)"])

    # ----------------------------
    # TAB 1: PREDICCIÓN MANUAL
    # ----------------------------
    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Completa los campos del solicitante", anchor=False)
        st.caption("Usa valores sin símbolos (p. ej., sin $). Los tooltips explican cada campo.")

        with st.form("manual_form_ui", border=False):
            c1, c2, c3 = st.columns([1, 1, 1], gap="large")

            with c1:
                RevolvingUtilizationOfUnsecuredLines = st.slider(
                    "Utilización de líneas no garantizadas",
                    min_value=0.00, max_value=1.00, value=0.10, step=0.01,
                    help="Saldo total en tarjetas/líneas de crédito personales dividido por límites totales (0–1)."
                )
                age = st.slider(
                    "Edad del solicitante (años)",
                    min_value=18, max_value=85, value=35, step=1,
                    help="Mínimo 18 años."
                )
                NumberOfTime30_59 = st.number_input(
                    "Atrasos 30–59 días (veces)",
                    min_value=0, value=1, step=1,
                    help="Número de veces con retrasos de pago de 30 a 59 días."
                )
                DebtRatio = st.slider(
                    "Relación de deuda",
                    min_value=0.00, max_value=2.50, value=0.20, step=0.01,
                    help="(Pagos de deudas + pensión + costos de vida) / Ingreso bruto mensual."
                )

            with c2:
                MonthlyIncome = st.number_input(
                    "Ingreso mensual (USD)",
                    min_value=0.0, value=2500.0, step=100.0, format="%0.2f",
                    help="Monto total de ingresos mensuales declarados."
                )
                NumberOfOpenCreditLinesAndLoans = st.slider(
                    "Líneas/préstamos abiertos",
                    min_value=0, max_value=40, value=3, step=1,
                    help="Total de tarjetas, préstamos de auto/hipoteca, etc., que están abiertos."
                )
                NumberOfTimes90DaysLate = st.number_input(
                    "Atrasos ≥ 90 días (veces)",
                    min_value=0, value=0, step=1,
                    help="Número de veces con retrasos de 90 días o más."
                )
                NumberRealEstateLoansOrLines = st.slider(
                    "Hipotecas/líneas inmobiliarias",
                    min_value=0, max_value=10, value=1, step=1,
                    help="Incluye créditos sobre el valor de la vivienda."
                )

            with c3:
                NumberOfTime60_89 = st.number_input(
                    "Atrasos 60–89 días (veces)",
                    min_value=0, value=0, step=1,
                    help="Número de veces con retrasos entre 60 y 89 días."
                )
                NumberOfDependents = st.slider(
                    "Número de dependientes",
                    min_value=0, max_value=10, value=1, step=1,
                    help="Personas a cargo (cónyuge, hijos, etc.)."
                )
                sex = st.selectbox(
                    "Sexo (según registro)",
                    ["female", "male"], index=1,
                    help="Sexo del solicitante (Hombre/Mujer) (Male/Female)."
                )
                Sex_num = 1.0 if sex == "male" else 0.0

            submitted = st.form_submit_button("⚡ Evaluar solicitud", use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)  # /card

        # ==== RESULTADOS ====
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
            X = ensure_columns(X, columns_order).fillna(0.0).replace([np.inf, -np.inf], 0.0)
            Xs = scaler.transform(X.values.astype(np.float32))
            probs = predict_scores(model, Xs, columns_order)
            prob = float(probs[0])
            yhat = int(prob >= threshold)

            # Colors based on prob
            def risk_color(p):
                if p < 0.33: return "var(--success)"
                if p < 0.66: return "var(--warning)"
                return "var(--danger)"

            # RISK HEADER
            st.markdown('<div class="card">', unsafe_allow_html=True)
            cA, cB = st.columns([0.6, 0.4], gap="large")

            with cA:
                pct_text = f"{prob*100:.2f}%"
                st.markdown("#### Resultado del modelo")
                st.markdown(
                    f"""
                    <div class="riskbar">
                        <div class="riskbar-fill" style="width:{prob*100:.2f}%; background:{risk_color(prob)};"></div>
                    </div>
                    <div style="display:flex;align-items:center;gap:10px;margin-top:8px;">
                        <div class="pill">
                            <span class="dot" style="background:{risk_color(prob)};"></span>
                            Probabilidad estimada: <b>{pct_text}</b>
                        </div>
                        <div class="pill">
                            Umbral: <b>{threshold:.2f}</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with cB:
                if yhat == 1:
                    st.markdown(
                        """
                        <div class="card-soft result-danger">
                            <h4>🚫 Decisión: <b>Riesgo</b></h4>
                            <p class="small">Interpretación: probabilidad alta de incumplimiento.</p>
                            <p class="smaller">Recomendación: no aprobar, o solicitar garantías adicionales según política.</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        """
                        <div class="card-soft result-ok">
                            <h4>✅ Decisión: <b>No riesgo</b></h4>
                            <p class="small">Interpretación: probabilidad baja de incumplimiento.</p>
                            <p class="smaller">Recomendación: proceder con proceso de aprobación conforme a políticas.</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            st.markdown('</div>', unsafe_allow_html=True)  # /card

            # LOG (no cambia)
            try:
                log_inference(pd.DataFrame([row]), np.array([prob]), np.array([yhat]), threshold)
            except Exception:
                pass

            # AYUDA/NOTAS
            with st.expander("ℹ️ Trazabilidad y notas del modelo"):
                st.write(
                    "- El score es una probabilidad en 0–1.\n"
                    "- La decisión binaria usa el umbral operativo.\n"
                    "- Esta herramienta es de apoyo; no reemplaza verificación documental ni políticas."
                )

    # ----------------------------
    # TAB 2: CSV / BATCH
    # ----------------------------
    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Carga masiva (CSV)", anchor=False)
        st.caption("El archivo debe incluir las columnas de entrada en el mismo orden. Puedes usar la plantilla.")

        with st.expander("Ver columnas esperadas"):
            st.code(",".join(columns_order), language="text")

        # Botón de plantilla CSV
        tpl = pd.DataFrame(columns=columns_order)
        st.download_button(
            "⬇️ Descargar plantilla CSV",
            data=tpl.to_csv(index=False).encode("utf-8"),
            file_name="plantilla_credit_mlp.csv",
            mime="text/csv",
            use_container_width=True
        )

        file = st.file_uploader("Sube un CSV con las columnas de entrada", type=["csv"], label_visibility="collapsed")

        st.markdown('</div>', unsafe_allow_html=True)  # /card

        if file is not None:
            try:
                df_in = pd.read_csv(file)

                # mapear 'Sex' -> 'Sex_num' si aplica
                if "Sex" in df_in.columns and "Sex_num" not in df_in.columns:
                    df_in["Sex_num"] = df_in["Sex"].map({"male": 1, "female": 0}).fillna(0).astype(float)

                df_in = ensure_columns(df_in, columns_order).fillna(0.0).replace([np.inf, -np.inf], 0.0)
                Xs = scaler.transform(df_in.values.astype(np.float32))
                probs = predict_scores(model, Xs, columns_order)
                preds = (probs >= threshold).astype(int)

                out = df_in.copy()
                out["prob_default"] = probs
                out["prediction"] = preds

                # KPIs lote
                st.markdown('<div class="card">', unsafe_allow_html=True)
                n_total = len(out)
                n_risk = int((out["prediction"] == 1).sum())
                n_ok = n_total - n_risk
                c1, c2, c3 = st.columns(3)
                c1.metric("Total registros", f"{n_total}")
                c2.metric("No riesgo", f"{n_ok}")
                c3.metric("Riesgo", f"{n_risk}")
                st.markdown('</div>', unsafe_allow_html=True)

                # Vista previa y descarga
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.write("Vista previa (primeras 20 filas):")
                st.dataframe(out.head(20), use_container_width=True)
                INFERENCE_DIR.mkdir(parents=True, exist_ok=True)
                out_path = INFERENCE_DIR / "batch_predictions.csv"
                out.to_csv(out_path, index=False)
                st.success(f"Predicciones guardadas en: {out_path.as_posix()}")
                st.markdown('</div>', unsafe_allow_html=True)

                # Log batch
                try:
                    log_inference(df_in, probs, preds, threshold)
                except Exception:
                    pass

                # Distribución simple (opcional)
                with st.expander("📊 Distribución de probabilidades (resumen rápido)"):
                    bins = pd.cut(out["prob_default"], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
                    st.bar_chart(bins.value_counts().sort_index())

            except Exception as e:
                st.error(f"Error procesando el archivo: {e}")

    # ----------------------------
    # SIDEBAR
    # ----------------------------
    with st.sidebar:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.header("ℹ️ Acerca de")
        st.write(
            "Dashboard de **scoring crediticio** con MLP (PyTorch + MLflow). "
            "Orientado a uso demostrativo y apoyo a decisión."
        )
        st.divider()
        st.subheader("Soporte")
        st.markdown(
            "- `models/mlflow_model/MLmodel`\n"
            "- `models/run_id.txt`\n"
            "- `models/threshold.txt`\n"
            "- `models/columns_used.json`"
        )
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    # Si usas Streamlit Cloud con Secrets para MLflow remoto (no necesario en modo offline):
    # if hasattr(st, "secrets"): os.environ.update({k: str(v) for k, v in st.secrets.items()})
    main()