import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuración visual
plt.style.use("ggplot")

# Rutas
train_path = Path("data/cs-training.csv")
test_path  = Path("data/cs-test.csv")

# Cargar datasets
train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)

# Eliminar columnas innecesarias
for df in [train_df, test_df]:
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

# Mostrar estructura
print("=== TRAIN ===")
print("Shape:", train_df.shape)
print(train_df.head(), "\n")

print("=== TEST ===")
print("Shape:", test_df.shape)
print(test_df.head(), "\n")

# Columnas que están en train y no en test
missing_cols = set(train_df.columns) - set(test_df.columns)
print("Columnas en train y no en test:", missing_cols)

# Revisar valores nulos
print("\nValores nulos en TRAIN:")
print(train_df.isnull().sum())

print("\nValores nulos en TEST:")
print(test_df.isnull().sum())

# Estadísticas descriptivas
print("\nDescripción TRAIN:")
print(train_df.describe())

# Distribución del target
plt.figure(figsize=(5,4))
sns.countplot(x="SeriousDlqin2yrs", data=train_df)
plt.title("Distribución del riesgo de mora (Train)")
plt.savefig("reports/target_distribution_train.png")
plt.show()

# Distribución de la edad
plt.figure(figsize=(6,4))
sns.histplot(train_df["age"], bins=30, kde=True)
plt.title("Distribución de la edad (Train)")
plt.savefig("reports/age_distribution_train.png")
plt.show()

# Correlaciones (solo en train)
plt.figure(figsize=(10,8))
corr = train_df.corr()
sns.heatmap(corr, cmap="coolwarm", annot=False)
plt.title("Mapa de correlaciones (Train)")
plt.savefig("reports/correlation_heatmap_train.png")
plt.show()

# Compara estructuras
print("\nComparación de columnas:")
for col in train_df.columns:
    if col in test_df.columns:
        print(f"{col:<40} | train dtype={train_df[col].dtype} | test dtype={test_df[col].dtype}")
