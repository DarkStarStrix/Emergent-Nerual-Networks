import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# More complex Neural Network Model
class ComplexNN (nn.Module):
    def __init__(self):
        super (ComplexNN, self).__init__ ()
        self.fc1 = nn.Linear (1, 128)
        self.fc2 = nn.Linear (128, 128)
        self.fc3 = nn.Linear (128, 64)
        self.fc4 = nn.Linear (64, 1)

    def forward(self, x):
        x = torch.relu (self.fc1 (x))
        x = torch.relu (self.fc2 (x))
        x = torch.relu (self.fc3 (x))
        return self.fc4 (x)


# Logistic Regression Model for Emergent Phenomena
class LogisticRegressionModel (nn.Module):
    def __init__(self, input_size):
        super (LogisticRegressionModel, self).__init__ ()
        self.linear = nn.Linear (input_size, 1)  # 1 output for binary classification

    def forward(self, x):
        logits = self.linear (x)
        out = torch.sigmoid (logits)
        return out


# Generate data for a sinusoidal task
def generate_sine_wave_data(phase_shift, num_samples=100):
    x = np.random.uniform (-5, 5, size=(num_samples, 1))
    y = np.sin (x + phase_shift)
    return torch.tensor (x, dtype=torch.float32), torch.tensor (y, dtype=torch.float32)


# Train the model on one task (Task 1)
def train_model_on_task(model, optimizer, criterion, x_train, y_train, num_epochs=2000):
    for epoch in range (num_epochs):
        optimizer.zero_grad ()
        y_pred = model (x_train)
        loss = criterion (y_pred, y_train)
        loss.backward ()
        optimizer.step ()
        if (epoch + 1) % 100 == 0:
            print (f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item ():.4f}")


# Meta-Learning: Test how the model generalizes to a new task (Task 2)
def test_generalization(model, x_test, y_test):
    with torch.no_grad ():
        y_pred = model (x_test)
        loss = nn.MSELoss () (y_pred, y_test)
        return loss.item ()


# Visualize predictions
def plot_predictions(x_train, y_train, x_test, y_test, model):
    with torch.no_grad ():
        y_train_pred = model (x_train)
        y_test_pred = model (x_test)

    plt.figure (figsize=(10, 5))

    # Plot training task (Task 1)
    plt.subplot (1, 2, 1)
    plt.scatter (x_train, y_train, label='True (Task 1)')
    plt.scatter (x_train, y_train_pred, label='Predicted (Task 1)', marker='x')
    plt.title ('Task 1: Training Data')
    plt.legend ()

    # Plot test task (Task 2)
    plt.subplot (1, 2, 2)
    plt.scatter (x_test, y_test, label='True (Task 2)')
    plt.scatter (x_test, y_test_pred, label='Predicted (Task 2)', marker='x')
    plt.title ('Task 2: Generalization to Unseen Data')
    plt.legend ()

    plt.show ()


# Logistic Regression to quantify emergent phenomena
def logistic_regression_on_parameters(model):
    params = np.concatenate ([p.detach ().numpy ().flatten () for p in model.parameters ()])
    scaler = StandardScaler ()
    params_scaled = scaler.fit_transform (params.reshape (-1, 1))
    # Generate synthetic labels for logistic regression
    labels = np.random.randint (0, 2, size=params_scaled.shape [0])
    logistic_model = LogisticRegression ()
    logistic_model.fit (params_scaled, labels)
    return logistic_model, scaler


# Plot normal distribution of emergent behaviors
def plot_normal_distribution(predictions, mean, std):
    plt.figure(figsize=(10, 6))
    plt.hist(predictions, bins=50, alpha=0.7, label='Predictions', color='b')
    plt.axvline(mean + 3 * std, color='r', linestyle='--', label='+3σ Threshold')
    plt.axvline(mean - 3 * std, color='r', linestyle='--', label='-3σ Threshold')

    # Mark emergent behaviors
    emergent_points = predictions[(predictions > mean + 3 * std) | (predictions < mean - 3 * std)]
    plt.scatter(emergent_points, np.zeros_like(emergent_points), color='r', label='Emergent Behaviors')

    plt.title("Emergent Behavior Detection (+3σ Threshold)")
    plt.xlabel("Prediction Values")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()


# Plotting the metrics
def plot_training_metrics(train_losses, train_accuracies, num_epochs):
    epochs = range (1, num_epochs + 1)
    plt.figure (figsize=(12, 5))

    plt.subplot (1, 2, 1)
    plt.plot (epochs, train_losses, 'b', label='Training loss')
    plt.title ('Training loss')
    plt.xlabel ('Epochs')
    plt.ylabel ('Loss')
    plt.legend ()

    plt.subplot (1, 2, 2)
    plt.plot (epochs, train_accuracies, 'r', label='Training accuracy')
    plt.title ('Training accuracy')
    plt.xlabel ('Epochs')
    plt.ylabel ('Accuracy')
    plt.legend ()

    plt.tight_layout ()
    plt.show ()


# Plotting the predicted probabilities
def plot_predicted_probabilities(X_test, predictions):
    plt.figure (figsize=(6, 5))
    plt.scatter (X_test [:, 0], X_test [:, 1], c=predictions.numpy ().flatten (), cmap='viridis', alpha=0.7)
    plt.colorbar (label='Predicted Probability')
    plt.xlabel ('Data Complexity')
    plt.ylabel ('Architectural Complexity')
    plt.title ('Predicted Probabilities of Emergent Phenomena')
    plt.show ()


# Main code
if __name__ == "__main__":
    # Task 1: Sine wave with no phase shift (Training)
    x_train, y_train = generate_sine_wave_data (phase_shift=0)

    # Task 2: Sine wave with a phase shift (Testing Generalization)
    x_test, y_test = generate_sine_wave_data (phase_shift=np.pi / 2)

    # Initialize the model, optimizer, and loss function
    model = ComplexNN ()
    optimizer = optim.Adam (model.parameters (), lr=0.01)
    criterion = nn.MSELoss ()

    # Train the model on Task 1 (no phase shift)
    print ("Training on Task 1...")
    train_model_on_task (model, optimizer, criterion, x_train, y_train, num_epochs=2000)

    # Test generalization to Task 2 (with phase shift)
    print ("\nTesting generalization on Task 2...")
    generalization_loss = test_generalization (model, x_test, y_test)
    print (f"Generalization Loss on Task 2: {generalization_loss:.4f}")

    # Visualize the results
    plot_predictions (x_train, y_train, x_test, y_test, model)

    # Logistic Regression on model parameters
    logistic_model, scaler = logistic_regression_on_parameters (model)

    # Generate predictions and plot normal distribution
    params = np.concatenate ([p.detach ().numpy ().flatten () for p in model.parameters ()])
    params_scaled = scaler.transform (params.reshape (-1, 1))
    predictions = logistic_model.predict_proba (params_scaled) [:, 1]
    mean = np.mean (predictions)
    std = np.std (predictions)
    plot_normal_distribution (predictions, mean, std)

    # Save the model
    torch.save (model.state_dict (), 'complex_nn_model.pkl')
    print ("Model saved as 'complex_nn_model.pth'")

    # Generate synthetic data for data complexity (X1) and architectural complexity (X2)
    np.random.seed (42)
    num_samples = 1000
    data_complexity = np.random.rand (num_samples, 1) * 10  # Example complexity range [0, 10]
    arch_complexity = np.random.rand (num_samples, 1) * 5  # Example architecture complexity [0, 5]

    # Combine into input features
    X_data = np.hstack ((data_complexity, arch_complexity))

    # Generate labels (emergence: 1 for occurrence, 0 for non-occurrence)
    y_data = (data_complexity + arch_complexity > 10).astype (np.float32)  # Example threshold

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split (X_data, y_data, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor (X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor (X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor (y_train, dtype=torch.float32).view (-1, 1)
    y_test_tensor = torch.tensor (y_test, dtype=torch.float32).view (-1, 1)

    # Initialize logistic regression model
    input_size = 2  # Two input features (data complexity and architectural complexity)
    logistic_model = LogisticRegressionModel (input_size=input_size)

    # Loss function and optimizer
    criterion = nn.BCELoss ()  # Binary Cross-Entropy loss for logistic regression
    optimizer = optim.SGD (logistic_model.parameters (), lr=0.01)

    # Training loop
    num_epochs = 1000
    train_losses = []
    train_accuracies = []

    for epoch in range (num_epochs):
        logistic_model.train ()

        # Forward pass
        outputs = logistic_model (X_train_tensor)
        loss = criterion (outputs, y_train_tensor)

        # Backward pass and optimization
        optimizer.zero_grad ()
        loss.backward ()
        optimizer.step ()

        # Calculate accuracy
        predicted_classes = (outputs >= 0.5).float ()
        accuracy = (predicted_classes == y_train_tensor).float ().mean ()

        # Record loss and accuracy
        train_losses.append (loss.item ())
        train_accuracies.append (accuracy.item ())

        # Print loss every 100 epochs
        if (epoch + 1) % 100 == 0:
            print (
                f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item ():.4f}, Accuracy: {accuracy.item () * 100:.2f}%')

    print ("Training complete.")

    # Evaluate the model
    logistic_model.eval ()
    with torch.no_grad ():
        predictions = logistic_model (X_test_tensor)
        predicted_classes = (predictions >= 0.5).float ()
        accuracy = (predicted_classes == y_test_tensor).float ().mean ()
        print (f'Test Accuracy: {accuracy.item () * 100:.2f}%')

    # Plotting the metrics
    plot_training_metrics (train_losses, train_accuracies, num_epochs)

    # Plotting the predicted probabilities
    plot_predicted_probabilities (X_test, predictions)
