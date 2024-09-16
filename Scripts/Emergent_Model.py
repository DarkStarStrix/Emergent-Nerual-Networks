import numpy as np
import matplotlib.pyplot as plt

# Generate some example data: Model predictions (normally distributed)
np.random.seed (42)
predictions = np.random.normal (loc=50, scale=5, size=1000)  # mean = 50, std = 5

# Calculate mean and standard deviation
mean = np.mean (predictions)
std = np.std (predictions)

# Define the +3σ and -3σ thresholds
threshold_upper = mean + 3 * std
threshold_lower = mean - 3 * std


# Function to detect emergent behavior
def detect_emergent_behavior(predictions, mean, std, threshold_upper, threshold_lower):
    # Identify predictions that are outside the +3σ or -3σ range
    emergent_behavior = [pred for pred in predictions if pred > threshold_upper or pred < threshold_lower]
    normal_behavior = [pred for pred in predictions if threshold_lower <= pred <= threshold_upper]

    return emergent_behavior, normal_behavior


# Detect emergent and normal behavior in predictions
emergent_behavior, normal_behavior = detect_emergent_behavior (predictions, mean, std, threshold_upper, threshold_lower)

# Print summary
print (f"Mean of predictions: {mean:.2f}")
print (f"Standard deviation of predictions: {std:.2f}")
print (f"Upper threshold (+3σ): {threshold_upper:.2f}")
print (f"Lower threshold (-3σ): {threshold_lower:.2f}")
print (f"Number of emergent behaviors detected: {len (emergent_behavior)}")
print (f"Number of normal behaviors detected: {len (normal_behavior)}")

# Plot the distribution and highlight emergent behavior
plt.figure (figsize=(10, 6))
plt.hist (predictions, bins=50, alpha=0.7, label='Predictions', color='b')
plt.axvline (threshold_upper, color='r', linestyle='--', label='+3σ Threshold')
plt.axvline (threshold_lower, color='r', linestyle='--', label='-3σ Threshold')
plt.scatter (emergent_behavior, np.zeros_like (emergent_behavior), color='r', label='Emergent Behaviors')
plt.title ("Emergent Behavior Detection (+3σ Threshold)")
plt.xlabel ("Prediction Values")
plt.ylabel ("Frequency")
plt.legend ()
plt.grid (True)
plt.show ()
