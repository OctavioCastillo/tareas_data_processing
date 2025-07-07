# red_bayesiana_balance_clasificada.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import KBinsDiscretizer
from customer_data import CustomerData

class RedBayesiana:
    VALOR_PREDICCION = 5000

    def __init__(self):
        self.customer_data = CustomerData()

    def ejecutar(self):
        print("[INFO] Iniciando modelo bayesiano para predicción categórica de BALANCE en función de CREDIT_LIMIT...")

        df = self.customer_data.get_data().copy()

        # Preprocesamiento
        if 'CUST_ID' in df.columns:
            df.drop(columns=['CUST_ID'], inplace=True)
        df.dropna(subset=['BALANCE', 'CREDIT_LIMIT'], inplace=True)

        # Crear variable objetivo: Categorías de BALANCE (baja, media, alta)
        df['Categoria_Balance'] = pd.cut(df['BALANCE'], bins=3, labels=["Baja", "Media", "Alta"])

        # Discretizar CREDIT_LIMIT para usar como predictor
        discretizador = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
        df['CREDIT_LIMIT_CAT'] = discretizador.fit_transform(df[['CREDIT_LIMIT']]).ravel()

        X = df[['CREDIT_LIMIT_CAT']]
        y = df['Categoria_Balance']

        # Conteo de clases
        conteo = y.value_counts()
        print(f"\n[INFO] Conteo de clases (Categoría de balance):\n{conteo}\n")

        # Entrenamiento del modelo
        modelo = GaussianNB()
        t0 = time.time()
        modelo.fit(X, y)
        t1 = time.time()
        print(f"[INFO] Tiempo de entrenamiento: {t1 - t0:.4f} segundos")

        # Predicción de probabilidades
        t2 = time.time()
        probas = modelo.predict_proba(X)
        t3 = time.time()
        print(f"[INFO] Tiempo de interpretación: {t3 - t2:.4f} segundos")

        # Obtener nombres de clases
        clases = modelo.classes_
        for i, clase in enumerate(clases):
            df[f'Prob_{clase}'] = probas[:, i]

        print("\n[INFO] Probabilidades promedio por clase:")
        for clase in clases:
            print(f"   - {clase}: {df[f'Prob_{clase}'].mean():.2f}")

        # Muestra de predicciones para clientes
        muestra = df.sample(10, random_state=42)
        print("\n[INFO] Muestra de predicciones para 10 clientes:")
        print(muestra[['CREDIT_LIMIT', 'Categoria_Balance'] + [f'Prob_{c}' for c in clases]])

        # Gráfica de predicción (stacked bar)
        muestra[[f'Prob_{c}' for c in clases]].plot(kind='bar', stacked=True)
        plt.title("Probabilidad de categoría de balance (muestra aleatoria)")
        plt.xlabel("Clientes")
        plt.ylabel("Probabilidad")
        plt.xticks(rotation=0)
        plt.legend(clases)
        plt.tight_layout()
        plt.savefig("prediccion_categoria_balance_bayes.png")
        plt.show()

        # ----------- PREDICCIÓN MANUAL CON CREDIT_LIMIT = 5000 ----------- #
        credit_limit_cat = discretizador.transform([[self.VALOR_PREDICCION]])[0][0]
        prob_pred = modelo.predict_proba([[credit_limit_cat]])[0]
        pred_clase = clases[np.argmax(prob_pred)]

        print(f"\n[INFO] Predicción manual para Credit Limit = {self.VALOR_PREDICCION}:")
        for i, clase in enumerate(clases):
            print(f"   - Probabilidad {clase}: {prob_pred[i]:.2f}")
        print(f"[INFO] Categoría de balance más probable: {pred_clase}")

# ---------- PRUEBA INTERNA ----------
if __name__ == "__main__":
    modelo = RedBayesiana()
    modelo.ejecutar()
