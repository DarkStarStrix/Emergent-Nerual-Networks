import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# Simple Neural Network Model
class SimpleNN (nn.Module):
    def __init__(self):
        super (SimpleNN, self).__init__ ()
        self.fc1 = nn.Linear (1, 64)
        self.fc2 = nn.Linear (64, 64)
        self.fc3 = nn.Linear (64, 1)

    def forward(self, x):
        x = torch.relu (self.fc1 (x))
        x = torch.relu (self.fc2 (x))
        return self.fc3 (x)


# Generate data for a sinusoidal task
def generate_sine_wave_data(phase_shift, num_samples=100):
    x = np.random.uniform (-5, 5, size=(num_samples, 1))
    y = np.sin (x + phase_shift)
    return torch.tensor (x, dtype=torch.float32), torch.tensor (y, dtype=torch.float32)


# Train the model on one task (Task 1)
def train_model_on_task(model, optimizer, criterion, x_train, y_train, num_epochs=1000):
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


# Main code
if __name__ == "__main__":
    # Task 1: Sine wave with no phase shift (Training)
    x_train, y_train = generate_sine_wave_data (phase_shift=0)

    # Task 2: Sine wave with a phase shift (Testing Generalization)
    x_test, y_test = generate_sine_wave_data (phase_shift=np.pi / 2)

    # Initialize the model, optimizer, and loss function
    model = SimpleNN ()
    optimizer = optim.Adam (model.parameters (), lr=0.01)
    criterion = nn.MSELoss ()

    # Train the model on Task 1 (no phase shift)
    print ("Training on Task 1...")
    train_model_on_task (model, optimizer, criterion, x_train, y_train)

    # Test generalization to Task 2 (with phase shift)
    print ("\nTesting generalization on Task 2...")
    generalization_loss = test_generalization (model, x_test, y_test)
    print (f"Generalization Loss on Task 2: {generalization_loss:.4f}")

    # Visualize the results
    plot_predictions (x_train, y_train, x_test, y_test, model)
