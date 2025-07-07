# regresion_lineal.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from customer_data import CustomerData

class RegresionLineal:
    VALOR_PREDICCION = 5000  # valor consistente para todos los modelos

    def __init__(self):
        self.data = CustomerData().get_data()

    def ejecutar(self):
        print("\n[INFO] Iniciando modelo de Regresión Lineal Simple...")

        # Datos usados
        df = self.data[['BALANCE', 'CREDIT_LIMIT']].dropna()

        X = df['CREDIT_LIMIT'].values.reshape(-1, 1)
        y = df['BALANCE'].values.reshape(-1, 1)

        # Entrenamiento
        modelo = LinearRegression()
        t0 = time.time()
        modelo.fit(X, y)
        t1 = time.time()

        print(f"[INFO] Tiempo de entrenamiento: {t1 - t0:.4f} segundos")

        # Predicción
        t2 = time.time()
        y_pred = modelo.predict(X)
        t3 = time.time()
        print(f"[INFO] Tiempo de interpretación: {t3 - t2:.4f} segundos")

        # Métricas
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        print("\n[RESULTADOS DEL MODELO]")
        print("Coeficiente (pendiente):", modelo.coef_[0][0])
        print("Intersección (término independiente):", modelo.intercept_[0])
        print("MSE (Error Cuadrático Medio):", mse)
        print("RMSE (Raíz del Error Cuadrático Medio):", rmse)
        print("MAE (Error Absoluto Medio):", mae)
        print("R² (Coeficiente de Determinación):", r2)

        # Gráfico de regresión
        sns.set_theme()
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, label='Datos reales', alpha=0.5)
        plt.plot(X, y_pred, color='red', label='Línea de regresión')
        plt.title('Regresión Lineal: Credit Limit vs Balance')
        plt.xlabel('Credit Limit')
        plt.ylabel('Balance')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Gráfico de residuos
        plt.figure(figsize=(10, 4))
        plt.scatter(X, y - y_pred, color='purple', alpha=0.6)
        plt.axhline(y=0, color='black', linestyle='--')
        plt.title("Gráfico de Residuos")
        plt.xlabel("Credit Limit")
        plt.ylabel("Error de predicción (Residual)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Predicción manual
        ejemplo = self.VALOR_PREDICCION
        t4 = time.time()
        prediccion = modelo.predict([[ejemplo]])
        t5 = time.time()

        print(f"\n[INFO] Predicción para Credit Limit = {ejemplo}")
        print(f"[INFO] Balance estimado: {prediccion[0][0]:.2f}")
        print(f"[INFO] Tiempo de predicción manual: {t5 - t4:.6f} segundos")


# ---------- PRUEBA INTERNA ----------
if __name__ == "__main__":
    rl = RegresionLineal()
    rl.ejecutar()
