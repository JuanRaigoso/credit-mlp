import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# ------------------------
# Paths de entrada/salida
# ------------------------
DATA_PATH = Path("data/train_clean.csv")
OUT_DIR = Path("data/scaled")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------
# Cargar datos
# ------------------------
df = pd.read_csv(DATA_PATH)

print("=== Escalamiento de variables ===")
print("Shape inicial:", df.shape)

# ------------------------
# Separar target y features
# ------------------------
TARGET = "SeriousDlqin2yrs"
y = df[TARGET]
X = df.drop(columns=[TARGET, "Sex"])  # eliminamos la versión textual

print("Variables usadas para modelado:")
print(X.columns.tolist())

# ------------------------
# División en train/valid
# ------------------------
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTamaño train: {X_train.shape}, valid: {X_valid.shape}")

# ------------------------
# Escalamiento con StandardScaler
# ------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

# Convertir a DataFrame con mismas columnas
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
X_valid_scaled_df = pd.DataFrame(X_valid_scaled, columns=X.columns, index=X_valid.index)

# ------------------------
# Guardar resultados
# ------------------------
X_train_scaled_df.to_csv(OUT_DIR / "X_train_scaled.csv", index=False)
X_valid_scaled_df.to_csv(OUT_DIR / "X_valid_scaled.csv", index=False)
y_train.to_csv(OUT_DIR / "y_train.csv", index=False)
y_valid.to_csv(OUT_DIR / "y_valid.csv", index=False)

# Guardar el scaler para usarlo luego en la app o el modelo
joblib.dump(scaler, OUT_DIR / "scaler.pkl")

print("\n✅ Escalamiento completado y archivos guardados en data/scaled/")
print("Archivos generados:")
print("- X_train_scaled.csv")
print("- X_valid_scaled.csv")
print("- y_train.csv")
print("- y_valid.csv")
print("- scaler.pkl")
