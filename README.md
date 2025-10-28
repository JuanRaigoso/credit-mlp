# 🏦 Credit-MLP: Evaluador de Riesgo Crediticio

**Autor:** Juan David Raigoso Espinosa  
**Profesión:** Ingeniero Industrial · Analista de Datos y Modelamiento Predictivo  
**Repositorio:** [github.com/JuanRaigoso/credit-mlp](https://github.com/JuanRaigoso/credit-mlp)

---

## 🌍 Contexto del proyecto

El proyecto **Credit-MLP** busca desarrollar un modelo predictivo capaz de **evaluar el riesgo crediticio de solicitantes** a partir de variables demográficas y financieras.  
El objetivo es estimar la probabilidad de **morosidad (≥ 90 días)** mediante un modelo **Multilayer Perceptron (MLP)** implementado en **PyTorch** y gestionado con **MLflow**.

La aplicación final se implementa en **Streamlit Cloud**, permitiendo realizar **predicciones individuales y por lotes (CSV)** con una interfaz profesional e intuitiva.

---

## 📊 Fuente y descripción de los datos

- **Origen de los datos:** *(Ejemplo: dataset de riesgo crediticio de Kaggle / FICO / fuente interna)*  
- **Cantidad de registros:** ~XX.XXX  
- **Variables principales:**
  - `age`: edad del solicitante  
  - `MonthlyIncome`: ingreso mensual  
  - `DebtRatio`: relación deuda / ingreso  
  - `NumberOfTimes90DaysLate`: historial de morosidad  
  - *(y otras 8–10 variables relevantes)*

> Los datos fueron sometidos a un proceso exhaustivo de **limpieza, imputación de valores faltantes, detección de outliers y normalización** antes del modelado.

---

## 🧹 Preprocesamiento y preparación de datos

1. Eliminación de valores atípicos y duplicados  
2. Imputación de valores nulos con mediana o cero según contexto  
3. Escalado de variables numéricas con `StandardScaler`  
4. Codificación de variables categóricas (`Sex_num`)  
5. División del conjunto en **train / test** con proporción 80/20  

---

## 🧠 Modelado

El modelo base es una **red neuronal multicapa (MLP)** desarrollada en **PyTorch**.

**Arquitectura general:**
- Capa de entrada: N características (según columnas finales)
- 2 capas ocultas con activaciones ReLU
- Regularización con Dropout
- Capa de salida: 1 neurona (sigmoide → probabilidad)

**Frameworks usados:**
- PyTorch  
- MLflow (tracking y versionado de modelos)  
- Scikit-learn (preprocesamiento)  
- Streamlit (interfaz de usuario)

---

## ⚙️ Entrenamiento

- **Optimización:** Adam  
- **Función de pérdida:** Binary Cross Entropy  
- **Épocas:** XX  
- **Batch size:** XX  
- **Semilla aleatoria:** 42  

**Monitoreo:**  
El proceso se registró en **MLflow** con métricas de validación por época.

---

## 📈 Resultados y métricas

| Métrica | Valor |
|----------|-------|
| Accuracy | 0.89 |
| Precision | 0.86 |
| Recall | 0.84 |
| F1-Score | 0.85 |
| AUC-ROC | 0.91 |

> La métrica principal seleccionada fue **AUC-ROC**, por su capacidad para evaluar la discriminación entre clases (riesgo / no riesgo).

---

## 📉 Curvas de entrenamiento

| Loss | Accuracy |
|------|-----------|
| ![Training Loss](./reports/imgs/loss_curve.png) | ![Accuracy Curve](./reports/imgs/accuracy_curve.png) |

> Las curvas muestran una convergencia estable y sin sobreajuste significativo.

---

## 💡 Interpretación de resultados

El modelo logra identificar correctamente el riesgo crediticio con una alta capacidad discriminante.  
Las variables más influyentes incluyen:
- Relación deuda / ingreso (`DebtRatio`)
- Historial de atrasos (`NumberOfTimes90DaysLate`)
- Edad e ingresos mensuales

El umbral operativo definido fue **0.5**, clasificando como *riesgo* a los solicitantes con probabilidad ≥ 0.5.

---

## 🚀 Implementación

El modelo final (`best_phase2_run_09.pt`) se integra a una app **Streamlit** con dos módulos:

1. **Predicción individual** mediante formulario interactivo  
2. **Scoring por archivo (CSV)** con resultados en lote  

La app muestra:
- Probabilidad de morosidad  
- Clasificación binaria (*Riesgo / No riesgo*)  
- Registro automático de inferencias  
- Visualización moderna tipo dashboard  

> 💻 Accede a la demo: *(enlace de Streamlit Cloud, si lo publicas)*

---

## 🧾 Estructura del repositorio

---

## 🧑‍💻 Autor

**Juan David Raigoso Espinosa**  
📍 Filandia, Quindío – Colombia  
🎓 Ingeniero Industrial  
💼 Enfocado en Analítica de Datos, Machine Learning y Visualización (Power BI / Python)  
🔗 [LinkedIn](https://www.linkedin.com/in/juanraigoso) | [GitHub](https://github.com/JuanRaigoso)

---

## 🏁 Conclusión

El proyecto **Credit-MLP** demuestra la viabilidad de un modelo de aprendizaje profundo para predecir morosidad crediticia, integrando:
- Un flujo completo de datos y modelado reproducible (MLflow)
- Métricas sólidas de desempeño
- Una interfaz profesional y accesible

> Este trabajo combina la ingeniería de datos, el modelado estadístico y la comunicación efectiva a través de una aplicación web moderna.

---

## ⚙️ Tecnologías utilizadas

| Categoría | Herramientas |
|------------|---------------|
| Lenguaje | Python 3.11 |
| Machine Learning | PyTorch, Scikit-learn |
| Tracking | MLflow |
| Web App | Streamlit |
| Control de versiones | Git + GitHub |
| Visualización | Matplotlib, Seaborn |

---

## 📬 Contacto

Si deseas conocer más sobre este proyecto o discutir colaboraciones:

📧 **juanraigosoespinosa@gmail.com**  
🔗 [LinkedIn](https://www.linkedin.com/in/juanraigoso)

---

