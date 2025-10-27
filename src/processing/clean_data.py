from pathlib import Path
import pandas as pd
import numpy as np
import json

# -----------------------
# Configuración de paths
# -----------------------
TRAIN_PATH = Path("data/cs-training.csv")
TEST_PATH  = Path("data/cs-test.csv")
OUT_TRAIN  = Path("data/train_clean.csv")
OUT_TEST   = Path("data/test_clean.csv")
REPORTS    = Path("reports")
REPORTS.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH = REPORTS / "cleaning_summary.json"

# -----------------------
# Columnas y reglas
# -----------------------
TARGET = "SeriousDlqin2yrs"

CAP_RULES = {
    "RevolvingUtilizationOfUnsecuredLines": ("upper", 1.0),
    "DebtRatio": ("upper", 10.0),
    "NumberOfTime30-59DaysPastDueNotWorse": ("upper", 10),
    "NumberOfTimes90DaysLate": ("upper", 10),
    "NumberOfTime60-89DaysPastDueNotWorse": ("upper", 10),
    "MonthlyIncome": ("upper", 30000.0),
    "NumberOfDependents": ("upper", 10),
    "age": ("lower", 18),
}

IMPUTE_WITH_MEDIAN = ["MonthlyIncome", "NumberOfDependents"]

RANDOM_SEED = 42  # para reproducibilidad de la columna Sex

def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df

def add_sex_column(df: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    # añade 'Sex' (male/female) y 'Sex_num' (1 male, 0 female)
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["Sex"] = rng.choice(["male", "female"], size=len(df))
    df["Sex_num"] = (df["Sex"] == "male").astype(int)
    return df

def apply_capping(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, (mode, limit) in CAP_RULES.items():
        if col not in df.columns:
            continue
        if mode == "upper":
            df[col] = np.where(df[col].notna(), np.minimum(df[col], limit), df[col])
        elif mode == "lower":
            df[col] = np.where(df[col].notna(), np.maximum(df[col], limit), df[col])
    return df

def compute_medians_for_impute(df_train_capped: pd.DataFrame) -> dict:
    med = {}
    for col in IMPUTE_WITH_MEDIAN:
        if col in df_train_capped.columns:
            med[col] = float(df_train_capped[col].median(skipna=True))
    return med

def impute_with_medians(df: pd.DataFrame, medians: dict) -> pd.DataFrame:
    df = df.copy()
    for col, m in medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(m)
    return df

def main():
    # 1) Cargar datasets
    train_df = load_csv(TRAIN_PATH)
    test_df  = load_csv(TEST_PATH)

    print("=== Antes de limpieza ===")
    print("train shape:", train_df.shape, " | test shape:", test_df.shape)

    # 2) Agregar columna Sex y versión numérica (en ambos)
    train_df = add_sex_column(train_df, seed=RANDOM_SEED)
    test_df  = add_sex_column(test_df,  seed=RANDOM_SEED + 1)  # otra semilla para variar

    # 3) Aplicar capping (solo afecta valores no nulos)
    train_capped = apply_capping(train_df)
    test_capped  = apply_capping(test_df)

    # 4) Calcular medianas para imputar (usando solo el train ya capado)
    medians = compute_medians_for_impute(train_capped)

    # 5) Imputar en train y test con medianas del train
    train_clean = impute_with_medians(train_capped, medians)
    test_clean  = impute_with_medians(test_capped,  medians)

    # 6) Guardar resultados
    OUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    train_clean.to_csv(OUT_TRAIN, index=False)
    test_clean.to_csv(OUT_TEST, index=False)

    # 7) Guardar resumen de reglas aplicadas
    summary = {
        "cap_rules": CAP_RULES,
        "impute_medians_from_train": medians,
        "added_columns": ["Sex", "Sex_num"],
        "notes": [
            "Capping aplicado antes de imputación",
            "Imputación con mediana del train para MonthlyIncome y NumberOfDependents",
            "age mínima forzada a 18",
        ],
        "input_train": str(TRAIN_PATH),
        "input_test": str(TEST_PATH),
        "output_train": str(OUT_TRAIN),
        "output_test": str(OUT_TEST),
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== Limpieza terminada ===")
    print(f"Train limpio: {OUT_TRAIN}")
    print(f"Test limpio:  {OUT_TEST}")
    print(f"Resumen JSON: {SUMMARY_PATH}")
    print("\nMedians utilizadas:", medians)

if __name__ == "__main__":
    main()
