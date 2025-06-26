import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

while True:
    print("Escoge una opción:")
    print("1. Selección Clonal")
    print("2. Selección Negativa")
    print("3. Redes Inmunitarias")
    print("4. Salir")

    opcion = int(input("Introduce tu opción: "))

    if opcion == 1:
        # Define the Rastrigin function
        def rastrigin(X):
            n = len(X)
            return 10 * n + np.sum(X**2 - 10 * np.cos(2 * np.pi * X))

        # Generate the initial population of potential solutions
        def generate_initial_population(pop_size, solution_size):
            return np.random.uniform(-5.12, 5.12, size=(pop_size, solution_size))  # Initialize within the search space of Rastrigin

        # Evaluate the fitness of each individual in the population (lower is better)
        def evaluate_population(population):
            return np.array([rastrigin(individual) for individual in population])

        # Select the best candidates from the population based on their fitness
        def select_best_candidates(population, fitness, num_candidates):
            indices = np.argsort(fitness)
            return population[indices[:num_candidates]], fitness[indices[:num_candidates]]

        # Clone the best candidates multiple times
        def clone_candidates(candidates, num_clones):
            return np.repeat(candidates, num_clones, axis=0)

        # Introduce random mutations to the cloned candidates to explore new solutions
        def mutate_clones(clones, mutation_rate):
            mutations = np.random.rand(*clones.shape) < mutation_rate
            clones[mutations] += np.random.uniform(-1, 1, np.sum(mutations))  # Mutate by adding a random value
            return clones

        # Main function implementing the Clonal Selection Algorithm
        def clonal_selection_algorithm(solution_size=2, pop_size=100, num_candidates=10, num_clones=10, mutation_rate=0.05, generations=100):
            population = generate_initial_population(pop_size, solution_size)

            best_fitness_per_generation = []  # Track the best fitness in each generation

            for generation in range(generations):
                fitness = evaluate_population(population)
                candidates, candidate_fitness = select_best_candidates(population, fitness, num_candidates)
                clones = clone_candidates(candidates, num_clones)
                mutated_clones = mutate_clones(clones, mutation_rate)
                clone_fitness = evaluate_population(mutated_clones)

                # Replace the worst individuals in the population with the new mutated clones
                population[:len(mutated_clones)] = mutated_clones
                fitness[:len(clone_fitness)] = clone_fitness

                # Track the best fitness of this generation
                best_fitness = np.min(fitness)
                best_fitness_per_generation.append(best_fitness)

                # Stop early if we've found a solution close to the global minimum
                if best_fitness < 1e-6:
                    print(f"Optimal solution found in {generation + 1} generations.")
                    break

            # Plot the fitness improvement over generations
            plt.figure(figsize=(10, 6))
            plt.plot(best_fitness_per_generation, marker='o', color='blue', label='Best Fitness per Generation')
            plt.xlabel('Generations')
            plt.ylabel('Fitness (Rastrigin Function Value)')
            plt.title('Fitness Improvement Over Generations')
            plt.grid(True)
            plt.show()

            # Return the best solution found (the one with the lowest fitness score)
            best_solution = population[np.argmin(fitness)]
            return best_solution

        # Example Usage
        best_solution = clonal_selection_algorithm(solution_size=2)  # Using 2D Rastrigin function
        print("Best solution found:", best_solution)
        print("Rastrigin function value at best solution:", rastrigin(best_solution))

        # Plot the surface of the Rastrigin function with the best solution found
        x = np.linspace(-5.12, 5.12, 200)
        y = np.linspace(-5.12, 5.12, 200)
        X, Y = np.meshgrid(x, y)
        Z = 10 * 2 + (X**2 - 10 * np.cos(2 * np.pi * X)) + (Y**2 - 10 * np.cos(2 * np.pi * Y))

        plt.figure(figsize=(8, 6))
        plt.contourf(X, Y, Z, levels=50, cmap='viridis')
        plt.colorbar(label='Function Value')
        plt.scatter(best_solution[0], best_solution[1], c='red', s=100, label='Best Solution')
        plt.title('Rastrigin Function Optimization with CSA')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        plt.show()

    elif opcion ==2:
        # Generate a synthetic dataset
        np.random.seed()

        # Parameters for bimodal distribution
        num_normal_points = 80  # total number of normal transactions
        num_points_per_cluster = num_normal_points // 2  # number of points in each cluster

        # Generate normal transactions for two clusters
        cluster1_center = [2, 2]
        cluster2_center = [8, 8]

        # Generate points around the first cluster center
        normal_cluster1 = np.random.normal(loc=cluster1_center, scale=0.5, size=(num_points_per_cluster, 2))

        # Generate points around the second cluster center
        normal_cluster2 = np.random.normal(loc=cluster2_center, scale=0.5, size=(num_points_per_cluster, 2))

        # Combine clusters into one dataset
        normal_transactions = np.vstack([normal_cluster1, normal_cluster2])

        # Define random distribution for fraudulent transactions
        num_fraud_points = 5  # number of fraudulent transactions
        fraudulent_transactions = np.random.uniform(low=0, high=10, size=(num_fraud_points, 2))

        # Combine into one dataset
        data = np.vstack([normal_transactions, fraudulent_transactions])
        labels = np.array([0] * len(normal_transactions) + [1] * len(fraudulent_transactions))

        # Function to generate detectors (random points) that don't match any of the normal data
        def generate_detectors(normal_data, num_detectors, detector_size):
            detectors = []
            while len(detectors) < num_detectors:
                detector = np.random.rand(detector_size) * 10  # Scale to cover the data range
                if not any(np.allclose(detector, pattern, atol=0.5) for pattern in normal_data):
                    detectors.append(detector)
            return np.array(detectors)

        # Function to detect anomalies (points in the data that are close to any detector)
        def detect_anomalies(detectors, data, threshold=0.5):
            anomalies = []
            for point in data:
                if any(np.linalg.norm(detector - point) < threshold for detector in detectors):
                    anomalies.append(point)
            return anomalies

        # Generate detectors that do not match the normal data
        detectors = generate_detectors(normal_transactions, num_detectors=300, detector_size=2)

        # Detect anomalies within the entire dataset using the detectors
        anomalies = detect_anomalies(detectors, data)
        print("Number of anomalies detected:", len(anomalies))

        # Convert anomalies to a numpy array for visualization
        anomalies = np.array(anomalies) if anomalies else np.array([])

        # Define axis limits
        x_min, x_max = 0, 10
        y_min, y_max = -1, 11

        # Visualize the dataset, detectors, and anomalies
        plt.figure(figsize=(14, 6))

        # Plot the normal transactions and fraudulent transactions
        plt.subplot(1, 2, 1)
        plt.scatter(normal_transactions[:, 0], normal_transactions[:, 1], color='red', marker='x', label='Normal Transactions')
        plt.scatter(fraudulent_transactions[:, 0], fraudulent_transactions[:, 1], color='green', marker='o', label='Fraudulent Transactions')
        plt.scatter(detectors[:, 0], detectors[:, 1], color='blue', marker='o', alpha=0.5, label='Detectors')
        if len(anomalies) > 0:
            plt.scatter(anomalies[:, 0], anomalies[:, 1], color='yellow', marker='*', s=100, label='Detected Anomalies')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Fraud Detection Using Negative Selection Algorithm (NSA)')
        plt.legend(loc='lower right')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.grid(False)

        # Create a grid of points to classify
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        # Classify grid points
        decision = np.array([any(np.linalg.norm(detector - point) < 0.5 for detector in detectors) for point in grid_points])
        decision = decision.reshape(xx.shape)

        # Plot the decision boundary
        plt.subplot(1, 2, 2)
        plt.contourf(xx, yy, decision, cmap='coolwarm', alpha=0.3)
        plt.scatter(normal_transactions[:, 0], normal_transactions[:, 1], color='red', marker='x', label='Normal Transactions')
        plt.scatter(fraudulent_transactions[:, 0], fraudulent_transactions[:, 1], color='green', marker='o', label='Fraudulent Transactions')
        plt.scatter(detectors[:, 0], detectors[:, 1], color='blue', marker='o', alpha=0.5, label='Detectors')
        if len(anomalies) > 0:
            plt.scatter(anomalies[:, 0], anomalies[:, 1], color='yellow', marker='*', s=100, label='Detected Anomalies')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Decision Boundary Visualization')
        plt.legend(loc='lower right')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.grid(False)

        # Show the plot
        plt.show()

    elif opcion == 3:
        # Generate synthetic data for illustration
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        X = np.random.rand(n_samples, n_features)  # Random economic indicators
        true_weights = np.array([0.5, -0.2, 0.3, 0.1, -0.1])
        y = X @ true_weights + np.random.normal(0, 0.1, n_samples)  # Stock prices with noise

        # Define the fitness function
        def fitness_function(solution, X, y):
            model = LinearRegression()
            model.coef_ = solution
            predictions = X @ model.coef_
            return mean_squared_error(y, predictions)

        # Generate the initial population of potential solutions
        def generate_initial_population(pop_size, solution_size):
            return np.random.uniform(-1, 1, size=(pop_size, solution_size))

        # Create the immune network
        def create_immune_network(population, fitness, num_neighbors):
            network = []
            for i, individual in enumerate(population):
                distances = np.linalg.norm(population - individual, axis=1)
                neighbors = np.argsort(distances)[1:num_neighbors+1]  # Exclude self
                network.append(neighbors)
            return network

        # Update the immune network
        def update_network(network, population, fitness, mutation_rate):
            new_population = np.copy(population)
            for i, neighbors in enumerate(network):
                if np.random.rand() < mutation_rate:
                    # Apply mutation with a smaller range
                    mutation = np.random.uniform(-0.05, 0.05, population.shape[1])
                    new_population[i] += mutation
            return new_population

        # Main function implementing Immune Network Theory
        def immune_network_theory(solution_size=n_features, pop_size=50, num_neighbors=5, mutation_rate=0.1, generations=50):
            population = generate_initial_population(pop_size, solution_size)
            best_fitness_per_generation = []  # Track the best fitness in each generation

            for generation in range(generations):
                fitness = np.array([fitness_function(ind, X, y) for ind in population])
                network = create_immune_network(population, fitness, num_neighbors)
                new_population = update_network(network, population, fitness, mutation_rate)

                # Evaluate the fitness of the new population
                fitness_new = np.array([fitness_function(ind, X, y) for ind in new_population])

                # Combine the old and new populations
                combined_population = np.vstack((population, new_population))
                combined_fitness = np.hstack((fitness, fitness_new))

                # Select the best individuals
                best_indices = np.argsort(combined_fitness)[:pop_size]
                population = combined_population[best_indices]
                fitness = combined_fitness[best_indices]

                # Track the best fitness of this generation
                best_fitness = np.min(fitness)
                best_fitness_per_generation.append(best_fitness)

                # Stop early if the fitness is good enough
                if best_fitness < 0.01:
                    print(f"Optimal solution found in {generation + 1} generations.")
                    break

            # Plot the fitness improvement over generations
            plt.figure(figsize=(10, 6))
            plt.plot(best_fitness_per_generation, marker='o', color='blue', label='Best Fitness per Generation')
            plt.xlabel('Generations')
            plt.ylabel('Mean Squared Error')
            plt.title('Fitness Improvement Over Generations')
            plt.grid(True)
            plt.show()

            # Return the best solution found
            best_solution = population[np.argmin(fitness)]
            return best_solution

        # Example Usage
        best_solution = immune_network_theory()
        print("Best solution found (economic indicators weights):", best_solution)
        print("Mean Squared Error at best solution:", fitness_function(best_solution, X, y))

        # Plot the predicted vs. actual values using the best solution
        model = LinearRegression()
        model.coef_ = best_solution
        predictions = X @ model.coef_

        plt.figure(figsize=(8, 6))
        plt.scatter(y, predictions, c='blue', label='Predicted vs Actual')
        plt.plot([min(y), max(y)], [min(y), max(y)], 'r--', label='Ideal Fit')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Stock Price Prediction with INT')
        plt.legend()
        plt.grid(True)
        plt.show()

    else:
        break