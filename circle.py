import jax.numpy as jnp
from jax import random, grad, jit
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Activation function
def tanh(x): return jnp.tanh(x)
def relu(x): return jnp.maximum(0, x)

# Generate circle data with classification logic
def generate_circle_data(key, n_samples=1000, inner_radius=1.0, outer_radius=2.5, noise_level=0.1):
    key_angle, key_radius, key_noise = random.split(key, 3)
    angles = random.uniform(key_angle, (n_samples,), minval=0.0, maxval=2 * jnp.pi)
    radii = random.uniform(key_radius, (n_samples,), minval=0.0, maxval=outer_radius) + \
            noise_level * random.normal(key_noise, (n_samples,))
    x = radii * jnp.cos(angles)
    y = radii * jnp.sin(angles)
    labels = (radii >= inner_radius).astype(jnp.float32)
    data = jnp.hstack([x[:, None], y[:, None]])
    return data, labels

# Initialize network parameters
def init_params(key, input_dim=2, hidden_dim=2, output_dim=1):
    key1, key2 = random.split(key)
    w1 = random.normal(key1, (input_dim, hidden_dim))
    b1 = jnp.zeros((hidden_dim,))
    w2 = random.normal(key2, (hidden_dim, output_dim))
    b2 = jnp.zeros((output_dim,))
    return (w1, b1), (w2, b2)

# Forward pass
def forward(params, x):
    (w1, b1), (w2, b2) = params
    hidden = relu(jnp.dot(x, w1) + b1)
    output = jnp.dot(hidden, w2) + b2
    return jnp.squeeze(output)

# Loss function with connection penalty
def loss_fn(params, x, y, connection_penalty):
    predictions = forward(params, x)
    loss = jnp.mean((predictions - y) ** 2)
    (w1, _), (w2, _) = params
    penalty = connection_penalty * (jnp.sum(jnp.abs(w1)) + jnp.sum(jnp.abs(w2)))
    return loss + penalty

grad_fn = jit(grad(loss_fn))

# Update parameters using gradient descent
def update_params(params, x, y, lr, connection_penalty):
    grads = grad_fn(params, x, y, connection_penalty)
    (w1, b1), (w2, b2) = params
    (dw1, db1), (dw2, db2) = grads
    return (w1 - lr * dw1, b1 - lr * db1), (w2 - lr * dw2, b2 - lr * db2)

# Mutation: Add random noise to parameters
def mutate_params(params, mutation_rate=0.05):
    (w1, b1), (w2, b2) = params
    key = random.PRNGKey(np.random.randint(0, 10000))
    noise1 = mutation_rate * random.normal(key, w1.shape)
    noise2 = mutation_rate * random.normal(key, w2.shape)
    return (w1 + noise1, b1), (w2 + noise2, b2)

# Add a new hidden node to the network
def add_node(params):
    (w1, b1), (w2, b2) = params

    # Add a new node to the hidden layer
    new_weights = random.normal(random.PRNGKey(np.random.randint(0, 10000)), (w1.shape[0], 1))
    w1 = jnp.hstack([w1, new_weights])
    b1 = jnp.append(b1, 0.0)

    # Extend the output layer to connect to the new hidden node
    new_output_weight = random.normal(random.PRNGKey(np.random.randint(0, 10000)), (1,))
    w2 = jnp.vstack([w2, new_output_weight])

    return (w1, b1), (w2, b2)

# Calculate accuracy
def calculate_accuracy(params, x, y):
    predictions = forward(params, x)
    predicted_labels = (predictions > 0.5).astype(jnp.float32)
    return jnp.mean(predicted_labels == y) * 100

# Plot test data with decision boundary
# Plot test data with decision boundary
def plot_data_with_accuracy(params, x, y, accuracy, title="Test Data with Accuracy", resolution=300):
    plt.figure(figsize=(8, 8))
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1  # Use only 'x' here

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    grid_points = jnp.c_[xx.ravel(), yy.ravel()]

    # Forward pass on grid points
    predictions = forward(params, grid_points).reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(xx, yy, predictions, levels=np.linspace(0, 1, 100), cmap="coolwarm", alpha=0.8)
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap="coolwarm", edgecolor="black", s=50, alpha=0.8)
    plt.title(f"{title}\nAccuracy: {accuracy:.2f}%")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()

# Evolve the population with dynamic node addition
def evolve_population(population, mutation_rate=0.1, add_node_prob=0.2):
    new_population = []
    for params in population:
        if np.random.rand() < add_node_prob:
            params = add_node(params)
        params = mutate_params(params, mutation_rate)
        new_population.append(params)
    return new_population

# Main script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural network on a circle dataset.")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of samples in the dataset.")
    parser.add_argument("--data_noise_level", type=float, default=0.1, help="Noise level in the dataset.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for training.")
    parser.add_argument("--connection_count_penalty", type=float, default=0.03, help="Penalty for network size.")
    parser.add_argument("--backprop_steps", type=int, default=600, help="Number of backpropagation steps.")
    args = parser.parse_args()

    key = random.PRNGKey(0)
    x_train, y_train = generate_circle_data(key, args.n_samples, noise_level=args.data_noise_level)
    x_test, y_test = generate_circle_data(key, args.n_samples, noise_level=args.data_noise_level)

    population_size = 128
    population = [init_params(random.split(key, 2)[0]) for _ in range(population_size)]

    for epoch in range(args.epochs):
        fitness_scores = [(calculate_accuracy(params, x_train, y_train), params) for params in population]
        best_fitness, best_params = max(fitness_scores, key=lambda x: x[0])

        print(f"Epoch {epoch + 1}/{args.epochs}, Best Fitness: {best_fitness:.2f}%")

        population = evolve_population([best_params], mutation_rate=0.1, add_node_prob=0.1)

        for step in range(args.backprop_steps):
            best_params = update_params(best_params, x_train, y_train, args.learning_rate,
                                        args.connection_count_penalty)

    test_accuracy = calculate_accuracy(best_params, x_test, y_test)
    print(f"Best Test Accuracy: {test_accuracy:.2f}%")

    plot_data_with_accuracy(best_params, x_test, y_test, test_accuracy)
