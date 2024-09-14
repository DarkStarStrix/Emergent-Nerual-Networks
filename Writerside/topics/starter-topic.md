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