import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from customer_data import CustomerData  # Asegúrate de que esté correctamente ubicado
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class SeleccionClonal:

    VALOR_PREDICCION = 5000  # mismo valor que en otros modelos

    def __init__(self):
        print("[INFO] Inicializando algoritmo de Selección Clonal para predicción de BALANCE...")

    def _fitness_function(self, individual, credit_limit, balance):
        prediction = credit_limit * individual
        return -mean_squared_error(balance, prediction)  # Queremos minimizar el error

    def _generate_population(self, pop_size):
        return np.random.rand(pop_size)

    def _evaluate_population(self, population, credit_limit, balance):
        return np.array([
            self._fitness_function(ind, credit_limit, balance)
            for ind in population
        ])

    def _select_best(self, population, fitness, n):
        indices = np.argsort(fitness)[::-1]  # descendente (mejor fitness primero)
        return population[indices[:n]], fitness[indices[:n]]

    def _clone_candidates(self, candidates, num_clones):
        return np.repeat(candidates, num_clones)

    def _mutate(self, clones, mutation_rate):
        mutations = np.random.rand(len(clones)) < mutation_rate
        clones[mutations] += np.random.uniform(-0.05, 0.05, np.sum(mutations))
        return np.clip(clones, 0, None)

    def ejecutar(self):
        print("[INFO] Cargando datos...")
        df = CustomerData().get_data()
        df = df.dropna(subset=["BALANCE", "CREDIT_LIMIT"])

        credit_limit = df["CREDIT_LIMIT"].values
        balance = df["BALANCE"].values

        pop_size = 30
        num_generations = 50
        top_k = 5
        clones_per = 10
        mutation_rate = 0.1

        print("[INFO] Iniciando optimización...")
        inicio_entrenamiento = time.time()

        population = self._generate_population(pop_size)
        best_fitness_history = []

        for gen in range(num_generations):
            fitness = self._evaluate_population(population, credit_limit, balance)
            best_individuals, best_fitness = self._select_best(population, fitness, top_k)

            clones = self._clone_candidates(best_individuals, clones_per)
            mutated_clones = self._mutate(clones, mutation_rate)

            clone_fitness = self._evaluate_population(mutated_clones, credit_limit, balance)

            new_population = mutated_clones[:pop_size - top_k]
            population = np.concatenate((best_individuals, new_population))

            gen_best = np.max(fitness)
            best_fitness_history.append(gen_best)
            print(f"[GEN {gen+1}] Mejor fitness: {gen_best:.6f}")

        fin_entrenamiento = time.time()
        tiempo_entrenamiento = fin_entrenamiento - inicio_entrenamiento
        print(f"[INFO] Optimización completada en {tiempo_entrenamiento:.2f} segundos.")

        best_overall_index = np.argmax(fitness)
        best_solution = population[best_overall_index]

        # Predicción manual para valor específico
        prediccion = self.VALOR_PREDICCION * best_solution
        print(f"\n[INFO] Predicción manual para Credit Limit = {self.VALOR_PREDICCION}")
        print(f"Balance estimado: {prediccion:.2f}")

        # Evaluación general del modelo
        balance_predicho = credit_limit * best_solution
        mse = mean_squared_error(balance, balance_predicho)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(balance, balance_predicho)
        r2 = r2_score(balance, balance_predicho)

        print("\n[RESULTADOS DEL MODELO CLONAL]")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R2 Score: {r2:.4f}")

        # Visualización
        plt.figure(figsize=(10, 6))
        plt.scatter(credit_limit, balance, label="Balance real", alpha=0.5)
        plt.plot(credit_limit, balance_predicho, color="red", label="Balance estimado (Clonal)")
        plt.title("Predicción de Balance según Credit Limit - Selección Clonal")
        plt.xlabel("Credit Limit")
        plt.ylabel("Balance")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Evolución del fitness
        plt.figure(figsize=(10, 6))
        plt.plot(best_fitness_history, marker='o', color='green')
        plt.title("Mejora del Fitness por Generación")
        plt.xlabel("Generación")
        plt.ylabel("Fitness (-MSE)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# ---------- PRUEBA INTERNA ----------
if __name__ == "__main__":
    modelo = SeleccionClonal()
    modelo.ejecutar()
