# regresion_lineal.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from customer_data import CustomerData

class RegresionLineal:
    def __init__(self):
        self.data = CustomerData().get_data()

    def ejecutar(self):
        print("\n[INFO] Iniciando modelo de Regresión Lineal Simple...")

        # Usaremos CREDIT_LIMIT para predecir el BALANCE
        df = self.data[['BALANCE', 'CREDIT_LIMIT']].dropna()

        X = df['CREDIT_LIMIT'].values.reshape(-1, 1)
        y = df['BALANCE'].values.reshape(-1, 1)

        # Entrenamiento del modelo
        modelo = LinearRegression()
        inicio_entrenamiento = time.time()
        modelo.fit(X, y)
        fin_entrenamiento = time.time()
        tiempo_entrenamiento = fin_entrenamiento - inicio_entrenamiento
        print(f"[INFO] Tiempo de entrenamiento: {tiempo_entrenamiento:.4f} segundos")

        # Predicción del modelo
        inicio_interpretacion = time.time()
        y_pred = modelo.predict(X)
        fin_interpretacion = time.time()
        tiempo_interpretacion = fin_interpretacion - inicio_interpretacion
        print(f"[INFO] Tiempo de interpretación: {tiempo_interpretacion:.4f} segundos")

        # Métricas
        print("\n[RESULTADOS DEL MODELO]")
        print("Coeficiente (pendiente):", modelo.coef_[0][0])
        print("Término independiente (intersección con y):", modelo.intercept_[0])
        print("Error cuadrático medio (MSE):", mean_squared_error(y, y_pred))
        print("Puntaje de varianza (R2):", r2_score(y, y_pred))

        # Visualización
        sns.set_theme()
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, label='Datos reales', alpha=0.5)
        plt.plot(X, y_pred, color='red', label='Línea de regresión')
        plt.title('Regresión lineal: Credit Limit vs Balance')
        plt.xlabel('Credit Limit')
        plt.ylabel('Balance')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Predicción manual
        ejemplo = 5000
        inicio_pred_manual = time.time()
        prediccion = modelo.predict([[ejemplo]])
        fin_pred_manual = time.time()
        tiempo_pred_manual = fin_pred_manual - inicio_pred_manual

        print(f"\n[INFO] Predicción: Si el Credit Limit es {ejemplo}, el Balance estimado es {prediccion[0][0]:.2f}")
        print(f"[INFO] Tiempo de predicción manual: {tiempo_pred_manual:.6f} segundos")


# ---------- PRUEBA INTERNA ----------
if __name__ == "__main__":
    rl = RegresionLineal()
    rl.ejecutar()
