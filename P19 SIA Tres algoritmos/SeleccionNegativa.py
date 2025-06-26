import numpy as np
import matplotlib.pyplot as plt

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