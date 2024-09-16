import numpy as np
import matplotlib.pyplot as plt


def generate_sinusoidal_data(num_samples, num_features):
    X = np.linspace (0, 4 * np.pi, num_samples)
    y = np.sin (X)
    y = (y - y.min ()) / (y.max () - y.min ()) * 9  # Normalize to 0-9 range
    y = y.astype (int)
    X = np.tile (X, (num_features, 1)).T  # Repeat the sinusoidal pattern for each feature
    return X, y


def plot_sinusoidal_data(X, y):
    plt.figure (figsize=(10, 5))
    for i in range (X.shape [1]):
        plt.plot (X [:, i], label=f'Feature {i + 1}')
    plt.plot (y, label='Target', linewidth=3, linestyle='--')
    plt.legend ()
    plt.title ('Sinusoidal Data')
    plt.xlabel ('Samples')
    plt.ylabel ('Values')
    plt.show ()


# Generate synthetic sinusoidal data
X, y = generate_sinusoidal_data (1000, 5)

# Plot the synthetic data
plot_sinusoidal_data (X, y)
