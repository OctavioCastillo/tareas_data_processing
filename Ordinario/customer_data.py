# customer_data.py
import pandas as pd
import os

class CustomerData:
    _instance = None

    def __new__(cls, file_path="./Ordinario/data.csv"):
        if cls._instance is None:
            print("[INFO] Creando nueva instancia de CustomerData...")
            cls._instance = super(CustomerData, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, file_path="./Ordinario/data.csv"):
        if self._initialized:
            return
        print(f"[INFO] Cargando datos desde: {file_path}")
        self.file_path = file_path
        self.data = self._load_data()
        self._initialized = True

    def _load_data(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"No se encontró el archivo: {self.file_path}")
        return pd.read_csv(self.file_path)

    def get_data(self):
        return self.data


# ---------- PRUEBA INTERNA ----------
if __name__ == "__main__":
    # Primera instancia
    dataset1 = CustomerData()
    print("Primeras filas del dataset:")
    print(dataset1.get_data().head())

    # Segunda instancia (no debe recargar el archivo)
    dataset2 = CustomerData()
    print("\n¿Ambas instancias son iguales?:", dataset1 is dataset2)

    # Mostrar que usan la misma data
    print("\nVerificación de datos compartidos:")
    print(dataset2.get_data().head())
