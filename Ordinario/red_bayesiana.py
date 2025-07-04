import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import KBinsDiscretizer
from customer_data import CustomerData

class RedBayesiana:
    def __init__(self):
        self.customer_data = CustomerData()

    def ejecutar(self):
        print("[INFO] Iniciando modelo de Red Bayesiana para predicción de alta deuda...")

        df = self.customer_data.get_data().copy()

        # Eliminar identificador y nulos
        if 'CUST_ID' in df.columns:
            df.drop(columns=['CUST_ID'], inplace=True)
        df.dropna(inplace=True)

        # Crear variable binaria objetivo: Alta deuda = 1 si balance > Q3
        umbral = df['BALANCE'].quantile(0.75)
        df['Alta_Deuda'] = (df['BALANCE'] > umbral).astype(int)

        # Conteo de clases
        conteo = df['Alta_Deuda'].value_counts()
        print(f"\n[INFO] Conteo de clases (0 = deuda baja o media, 1 = deuda alta):\n{conteo}\n")

        # Discretizar variables predictoras
        variables = ['PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS', 'TENURE']
        discretizador = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
        df[variables] = discretizador.fit_transform(df[variables])

        # Separar X y y
        X = df[variables]
        y = df['Alta_Deuda']

        # Modelo bayesiano
        modelo = GaussianNB()
        inicio_entrenamiento = time.time()
        modelo.fit(X, y)
        fin_entrenamiento = time.time()
        tiempo_entrenamiento = fin_entrenamiento - inicio_entrenamiento
        print(f"[INFO] Tiempo de entrenamiento: {tiempo_entrenamiento:.4f} segundos")

        # Predicción e interpretación
        inicio_interpretacion = time.time()
        probabilidades = modelo.predict_proba(X)
        fin_interpretacion = time.time()
        tiempo_interpretacion = fin_interpretacion - inicio_interpretacion
        print(f"[INFO] Tiempo de interpretación: {tiempo_interpretacion:.4f} segundos")

        # Agregar columnas de probabilidad
        df['Prob_Baja'] = probabilidades[:, 0]
        df['Prob_Alta'] = probabilidades[:, 1]

        print("\n[INFO] Probabilidades promedio:")
        print("   - Baja deuda: {:.2f}".format(df['Prob_Baja'].mean()))
        print("   - Alta deuda: {:.2f}".format(df['Prob_Alta'].mean()))

        # Muestra de predicciones
        muestra = df.sample(10, random_state=42)
        print("\n[INFO] Muestra de predicciones para 10 clientes:")
        print(muestra[variables + ['Prob_Baja', 'Prob_Alta']])

        # Visualización
        muestra[['Prob_Baja', 'Prob_Alta']].plot(kind='bar', stacked=True)
        plt.title("Probabilidad de Alta Deuda (Muestra de Clientes)")
        plt.xlabel("Clientes (muestra aleatoria)")
        plt.ylabel("Probabilidad")
        plt.xticks(rotation=0)
        plt.legend(["Baja deuda", "Alta deuda"])
        plt.tight_layout()
        plt.savefig("prediccion_alta_deuda_bayes.png")
        plt.show()

# ---------- PRUEBA INTERNA ----------
if __name__ == "__main__":
    modelo = RedBayesiana()
    modelo.ejecutar()
