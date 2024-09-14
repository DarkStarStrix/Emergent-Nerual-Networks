import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Logistic Regression Model for Emergent Phenomena
class LogisticRegressionModel (nn.Module):
    def __init__(self, input_size):
        super (LogisticRegressionModel, self).__init__ ()
        self.linear = nn.Linear (input_size, 1)  # 1 output for binary classification

    def forward(self, x):
        logits = self.linear (x)
        out = torch.sigmoid (logits)
        return out


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
model = LogisticRegressionModel (input_size=input_size)

# Loss function and optimizer
criterion = nn.BCELoss ()  # Binary Cross-Entropy loss for logistic regression
optimizer = optim.SGD (model.parameters (), lr=0.01)

# Training loop
num_epochs = 1000
train_losses = []
train_accuracies = []

for epoch in range (num_epochs):
    model.train ()

    # Forward pass
    outputs = model (X_train_tensor)
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
        print (f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item ():.4f}, Accuracy: {accuracy.item () * 100:.2f}%')

print ("Training complete.")

# Evaluate the model
model.eval ()
with torch.no_grad ():
    predictions = model (X_test_tensor)
    predicted_classes = (predictions >= 0.5).float ()
    accuracy = (predicted_classes == y_test_tensor).float ().mean ()
    print (f'Test Accuracy: {accuracy.item () * 100:.2f}%')

# Plotting the metrics
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
plt.figure (figsize=(6, 5))
plt.scatter (X_test [:, 0], X_test [:, 1], c=predictions.numpy ().flatten (), cmap='viridis', alpha=0.7)
plt.colorbar (label='Predicted Probability')
plt.xlabel ('Data Complexity')
plt.ylabel ('Architectural Complexity')
plt.title ('Predicted Probabilities of Emergent Phenomena')
plt.show ()
