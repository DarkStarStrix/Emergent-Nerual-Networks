# Topic title: PyTorch Hybrid CNN-RNN Model for Sequence Processing

The provided code is a PyTorch implementation of a hybrid Convolutional Neural Network (CNN) and Recurrent Neural Network (RNN) model designed to process sequences of images and predict emergent phenomena. The model is trained and evaluated using synthetic data, and the training process is visualized using `matplotlib`.

First, the code imports necessary libraries, including PyTorch for building and training the model, `numpy` for generating synthetic data, `sklearn` for splitting the data into training and testing sets, and `matplotlib` for visualizing the training metrics.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score
```

The `generate_random_image_data` function creates synthetic image data and labels. The data consists of sequences of images, each with a specified number of channels, height, and width. The labels are randomly generated integers representing different classes.

```python
def generate_random_image_data(num_sequences, seq_len, image_size, num_channels, num_classes):
    X_data = np.random.rand(num_sequences, seq_len, num_channels, image_size, image_size).astype(np.float32)
    y_data = np.random.randint(0, num_classes, num_sequences).astype(np.long)
    return X_data, y_data
```

The `CNN_RNN_Model` class defines the hybrid model. It consists of two convolutional layers followed by a max-pooling layer to extract features from each image in the sequence. The output of the convolutional layers is flattened and passed to an LSTM layer, which processes the sequence of features. Finally, a fully connected layer produces the output.

```python
class CNN_RNN_Model(nn.Module):
    def __init__(self, input_channels, conv_out_channels, rnn_hidden_size, output_size, num_layers=1):
        super(CNN_RNN_Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=conv_out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=conv_out_channels, out_channels=conv_out_channels, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self._to_linear = None
        self.convs(torch.randn(1, input_channels, 32, 32))
        self.rnn = nn.LSTM(input_size=self._to_linear, hidden_size=rnn_hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(rnn_hidden_size, output_size)
```

The `forward` method of the `CNN_RNN_Model` class processes each image in the sequence through the convolutional layers, flattens the output, and stacks them into a tensor. This tensor is then passed through the LSTM layer, and the output from the last time step is fed into the fully connected layer to produce the final prediction.

```python
def forward(self, x):
    batch_size, seq_len, _, _, _ = x.shape
    cnn_out = []
    for t in range(seq_len):
        x_t = x[:, t, :, :, :]
        x_t = self.convs(x_t)
        x_t = x_t.view(batch_size, -1)
        cnn_out.append(x_t)
    cnn_out = torch.stack(cnn_out, dim=1)
    rnn_out, _ = self.rnn(cnn_out)
    out = self.fc(rnn_out[:, -1, :])
    return out
```

The training loop initializes the model, optimizer, and loss function. It then iterates over the training data for a specified number of epochs, performing forward and backward passes, updating the model parameters, and recording the loss, accuracy, and recall for each epoch.

```python
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        labels = labels.long()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct / total
    epoch_recall = recall_score(all_labels, all_preds, average='macro')
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)
    train_recalls.append(epoch_recall)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, Recall: {epoch_recall:.4f}')
```

Finally, the code plots the training loss, accuracy, and recall over the epochs using `matplotlib`. This visualization helps in understanding the model's performance and training progress.

```python
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(epochs, train_losses, 'b', label='Training loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 3, 2)
plt.plot(epochs, train_accuracies, 'r', label='Training accuracy')
plt.title('Training accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1, 3, 3)
plt.plot(epochs, train_recalls, 'g', label='Training recall')
plt.title('Training recall')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.legend()
plt.tight_layout()
plt.show()
```

## Logistic regression for predicting emergent phenomena
The provided code is a PyTorch implementation of a logistic regression model designed to predict emergent phenomena based on data complexity and architectural complexity. The code includes data generation, model definition, training, evaluation, and visualization of the results.

First, the necessary libraries are imported, including PyTorch for building and training the model, `numpy` for generating synthetic data, `sklearn` for splitting the data into training and testing sets, and `matplotlib` for visualizing the training metrics.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
```

The `LogisticRegressionModel` class defines a simple logistic regression model with a single linear layer. The `forward` method applies the linear transformation and the sigmoid activation function to produce the output probabilities.

```python
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)  # 1 output for binary classification

    def forward(self, x):
        logits = self.linear(x)
        out = torch.sigmoid(logits)
        return out
```

Synthetic data is generated for data complexity and architectural complexity. The data consists of two features, and the labels are binary, indicating the occurrence of emergent phenomena. The data is then split into training and testing sets.

```python
np.random.seed(42)
num_samples = 1000
data_complexity = np.random.rand(num_samples, 1) * 10  # Example complexity range [0, 10]
arch_complexity = np.random.rand(num_samples, 1) * 5  # Example architecture complexity [0, 5]
X_data = np.hstack((data_complexity, arch_complexity))
y_data = (data_complexity + arch_complexity > 10).astype(np.float32)  # Example threshold
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
```

The training loop initializes the model, optimizer, and loss function. It then iterates over the training data for a specified number of epochs, performing forward and backward passes, updating the model parameters, and recording the loss and accuracy for each epoch.

```python
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    predicted_classes = (outputs >= 0.5).float()
    accuracy = (predicted_classes == y_train_tensor).float().mean()
    train_losses.append(loss.item())
    train_accuracies.append(accuracy.item())
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy.item() * 100:.2f}%')
```

After training, the model is evaluated on the test set, and the test accuracy is printed. The code also includes visualizations of the training loss and accuracy over epochs using `matplotlib`.

```python
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    predicted_classes = (predictions >= 0.5).float()
    accuracy = (predicted_classes == y_test_tensor).float().mean()
    print(f'Test Accuracy: {accuracy.item() * 100:.2f}%')
```

Finally, the code plots the predicted probabilities of emergent phenomena based on data and architectural complexity, providing a visual understanding of the model's predictions.

```python
plt.figure(figsize=(6, 5))
plt.scatter(X_test[:, 0], X_test[:, 1], c=predictions.numpy().flatten(), cmap='viridis', alpha=0.7)
plt.colorbar(label='Predicted Probability')
plt.xlabel('Data Complexity')
plt.ylabel('Architectural Complexity')
plt.title('Predicted Probabilities of Emergent Phenomena')
plt.show()
```

## Meta-learning on a sinusoidal task
The provided code is a PyTorch implementation of a simple neural network designed to perform meta-learning on a sinusoidal task. The code includes data generation, model definition, training, testing, and visualization of the results.

First, the necessary libraries are imported, including PyTorch for building and training the model, `numpy` for generating synthetic data, and `matplotlib` for visualizing the training metrics.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
```

The `SimpleNN` class defines a simple neural network with three fully connected layers. The `forward` method applies the ReLU activation function to the outputs of the first two layers and returns the output of the third layer.

```python
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
```

The `generate_sine_wave_data` function generates synthetic data for a sinusoidal task. It creates input data `x` uniformly distributed between -5 and 5 and computes the corresponding `y` values as the sine of `x` plus a phase shift.

```python
def generate_sine_wave_data(phase_shift, num_samples=100):
    x = np.random.uniform(-5, 5, size=(num_samples, 1))
    y = np.sin(x + phase_shift)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
```

The `train_model_on_task` function trains the model on a given task. It iterates over the training data for a specified number of epochs, performing forward and backward passes, updating the model parameters, and printing the loss every 100 epochs.

```python
def train_model_on_task(model, optimizer, criterion, x_train, y_train, num_epochs=1000):
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")
```

The `test_generalization` function evaluates the model's generalization to a new task. It computes the mean squared error (MSE) loss between the model's predictions and the true values for the test data.

```python
def test_generalization(model, x_test, y_test):
    with torch.no_grad():
        y_pred = model(x_test)
        loss = nn.MSELoss()(y_pred, y_test)
        return loss.item()
```

The `plot_predictions` function visualizes the model's predictions for both the training and test tasks. It plots the true and predicted values for the training data (Task 1) and the test data (Task 2) using `matplotlib`.

```python
def plot_predictions(x_train, y_train, x_test, y_test, model):
    with torch.no_grad():
        y_train_pred = model(x_train)
        y_test_pred = model(x_test)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(x_train, y_train, label='True (Task 1)')
    plt.scatter(x_train, y_train_pred, label='Predicted (Task 1)', marker='x')
    plt.title('Task 1: Training Data')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(x_test, y_test, label='True (Task 2)')
    plt.scatter(x_test, y_test_pred, label='Predicted (Task 2)', marker='x')
    plt.title('Task 2: Generalization to Unseen Data')
    plt.legend()

    plt.show()
```

In the main code block, the model is trained on a sine wave with no phase shift (Task 1) and tested on a sine wave with a phase shift (Task 2). The training and generalization losses are printed, and the results are visualized.

```python
if __name__ == "__main__":
    x_train, y_train = generate_sine_wave_data(phase_shift=0)
    x_test, y_test = generate_sine_wave_data(phase_shift=np.pi / 2)

    model = SimpleNN()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    print("Training on Task 1...")
    train_model_on_task(model, optimizer, criterion, x_train, y_train)

    print("\nTesting generalization on Task 2...")
    generalization_loss = test_generalization(model, x_test, y_test)
    print(f"Generalization Loss on Task 2: {generalization_loss:.4f}")

    plot_predictions(x_train, y_train, x_test, y_test, model)
```

## Emergent behavior detection in model predictions

The provided code is a Python script that detects and visualizes emergent behavior in a set of model predictions. The script uses `numpy` for data generation and statistical calculations, and `matplotlib` for plotting the results.

First, the script generates synthetic data representing model predictions. These predictions are normally distributed with a mean of 50 and a standard deviation of 5. The `numpy` library is used to create this data:

```python
np.random.seed(42)
predictions = np.random.normal(loc=50, scale=5, size=1000)
```

Next, the script calculates the mean and standard deviation of the generated predictions. These statistics are used to define the thresholds for detecting emergent behavior. Specifically, the thresholds are set at three standard deviations above and below the mean:

```python
mean = np.mean(predictions)
std = np.std(predictions)
threshold_upper = mean + 3 * std
threshold_lower = mean - 3 * std
```

The `detect_emergent_behavior` function identifies predictions that fall outside the defined thresholds. Predictions outside the +3σ or -3σ range are considered emergent behavior, while those within the range are considered normal behavior:

```python
def detect_emergent_behavior(predictions, mean, std, threshold_upper, threshold_lower):
    emergent_behavior = [pred for pred in predictions if pred > threshold_upper or pred < threshold_lower]
    normal_behavior = [pred for pred in predictions if threshold_lower <= pred <= threshold_upper]
    return emergent_behavior, normal_behavior
```

The script then calls this function to classify the predictions and prints a summary of the results, including the number of emergent and normal behaviors detected:

```python
emergent_behavior, normal_behavior = detect_emergent_behavior(predictions, mean, std, threshold_upper, threshold_lower)
print(f"Mean of predictions: {mean:.2f}")
print(f"Standard deviation of predictions: {std:.2f}")
print(f"Upper threshold (+3σ): {threshold_upper:.2f}")
print(f"Lower threshold (-3σ): {threshold_lower:.2f}")
print(f"Number of emergent behaviors detected: {len(emergent_behavior)}")
print(f"Number of normal behaviors detected: {len(normal_behavior)}")
```

Finally, the script visualizes the distribution of predictions and highlights the emergent behaviors using `matplotlib`. It plots a histogram of the predictions, marks the +3σ and -3σ thresholds with vertical lines, and uses scatter points to indicate the emergent behaviors:

```python
plt.figure(figsize=(10, 6))
plt.hist(predictions, bins=50, alpha=0.7, label='Predictions', color='b')
plt.axvline(threshold_upper, color='r', linestyle='--', label='+3σ Threshold')
plt.axvline(threshold_lower, color='r', linestyle='--', label='-3σ Threshold')
plt.scatter(emergent_behavior, np.zeros_like(emergent_behavior), color='r', label='Emergent Behaviors')
plt.title("Emergent Behavior Detection (+3σ Threshold)")
plt.xlabel("Prediction Values")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.show()
```

This visualization helps in understanding the distribution of predictions and identifying the points that exhibit emergent behavior.
