# red_neuronal_balance.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from customer_data import CustomerData

class RedNeuronalBalance:
    VALOR_PREDICCION = 5000

    def __init__(self):
        self.customer_data = CustomerData()
        self.df = self.customer_data.get_data()

    def ejecutar(self):
        print("[INFO] Iniciando modelo de red neuronal para predecir BALANCE a partir de CREDIT_LIMIT...")

        df = self.df.copy()
        df.drop(columns=["CUST_ID"], inplace=True, errors='ignore')
        df.dropna(subset=["CREDIT_LIMIT", "BALANCE"], inplace=True)

        # Variables
        X = df[["CREDIT_LIMIT"]].values
        y = df["BALANCE"].values

        # Escalar X
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Modelo
        print("[INFO] Construyendo red neuronal...")
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')

        # Entrenamiento
        print("[INFO] Entrenando...")
        t0 = time.time()
        history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=False)
        t1 = time.time()
        print(f"[TIEMPO] Entrenamiento completado en {t1 - t0:.2f} segundos.")

        # Evaluación
        loss = model.evaluate(X_test, y_test, verbose=False)
        print(f"[RESULTADO] Pérdida (loss) en conjunto de prueba: {loss:.2f}")

        # Gráfica de pérdida
        plt.plot(history.history['loss'])
        plt.title('Progreso de la pérdida durante el entrenamiento')
        plt.xlabel('Época')
        plt.ylabel('Pérdida (loss)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Predicción manual
        credit_limit_estandarizado = scaler.transform([[self.VALOR_PREDICCION]])
        t2 = time.time()
        prediccion = model.predict(credit_limit_estandarizado)
        t3 = time.time()

        print(f"[TIEMPO] Predicción manual realizada en {t3 - t2:.4f} segundos.")
        print(f"[PREDICCIÓN] Si el Credit Limit es {self.VALOR_PREDICCION}, el Balance estimado es: {prediccion[0][0]:.2f}")

# ---------- PRUEBA INTERNA ----------
if __name__ == "__main__":
    modelo = RedNeuronalBalance()
    modelo.ejecutar()
