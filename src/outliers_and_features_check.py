import pandas as pd
import numpy as np
from pathlib import Path

# -----------------------
# Configuración de paths
# -----------------------
DATA_TRAIN = Path("data/cs-training.csv")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# Cargar y limpiar
# -----------------------
df = pd.read_csv(DATA_TRAIN)

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

print("=== RESUMEN INICIAL ===")
print("Shape:", df.shape)
print("Columnas:", df.columns.tolist(), "\n")

# ----------------------------------------
# Percentiles para revisar posibles outliers
# ----------------------------------------
cols_to_check = [
    "RevolvingUtilizationOfUnsecuredLines",
    "DebtRatio",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "MonthlyIncome",
    "NumberOfDependents",
]

percentiles = df[cols_to_check].quantile([0.01, 0.99])
print("=== Percentiles 1% y 99% (train) ===\n", percentiles, "\n")

# Señales de valores muy extremos vs p99
for col in cols_to_check:
    mx = df[col].max()
    p99 = percentiles.loc[0.99, col]
    if pd.notna(mx) and pd.notna(p99) and mx > p99 * 5:
        print(f"⚠️ {col}: max={mx:.4f} >> p99={p99:.4f}  (posible outlier severo)")

# ----------------------------------------
# Agregar columna 'Sex' y versión numérica
# ----------------------------------------
np.random.seed(42)  # reproducible
df["Sex"] = np.random.choice(["male", "female"], size=len(df))

# Para correlaciones numéricas solamente:
df["Sex_num"] = (df["Sex"] == "male").astype(int)

# ----------------------------------------
# Selección de columnas numéricas
# ----------------------------------------
# Tomamos todas las numéricas + Sex_num (target incluido si existe)
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Por claridad, aseguramos un orden (opcional)
preferred_order = [
    "SeriousDlqin2yrs",
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
num_cols = [c for c in preferred_order if c in num_cols]

# ----------------------------------------
# Correlaciones: Pearson y Spearman (matrices)
# ----------------------------------------
corr_pearson = df[num_cols].corr(method="pearson")
corr_spearman = df[num_cols].corr(method="spearman")

# Guardar matrices completas
corr_pearson.to_csv(REPORTS_DIR / "correlation_matrix_pearson.csv", index=True)
corr_spearman.to_csv(REPORTS_DIR / "correlation_matrix_spearman.csv", index=True)

print("\n✅ Guardado: reports/correlation_matrix_pearson.csv")
print("✅ Guardado: reports/correlation_matrix_spearman.csv")

# ----------------------------------------
# Correlación contra el target (ordenada)
# ----------------------------------------
target_col = "SeriousDlqin2yrs"
if target_col in df.columns:
    # Pearson
    corr_target_p = corr_pearson[target_col].drop(labels=[target_col]).sort_values(key=lambda s: s.abs(), ascending=False)
    # Spearman
    corr_target_s = corr_spearman[target_col].drop(labels=[target_col]).sort_values(key=lambda s: s.abs(), ascending=False)

    corr_target_p.to_csv(REPORTS_DIR / "corr_with_target_pearson.csv", header=["corr_with_target"])
    corr_target_s.to_csv(REPORTS_DIR / "corr_with_target_spearman.csv", header=["corr_with_target"])

    print("\n=== Top correlaciones con el target (|Pearson|) ===")
    print(corr_target_p.head(10), "\n")

    print("=== Top correlaciones con el target (|Spearman|) ===")
    print(corr_target_s.head(10), "\n")

else:
    print("\nℹ️ No se encontró la columna de target para correlación puntual.")

# ----------------------------------------
# Guardar dataset extendido (con Sex)
# ----------------------------------------
out_path = Path("data/cs-training-extended.csv")
df.to_csv(out_path, index=False)
print(f"✅ Dataset extendido guardado: {out_path}")