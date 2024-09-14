import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score


# Generate random image data using numpy
def generate_random_image_data(num_sequences, seq_len, image_size, num_channels, num_classes):
    # Generate random image data and labels
    X_data = np.random.rand (num_sequences, seq_len, num_channels, image_size, image_size).astype (np.float32)
    y_data = np.random.randint (0, num_classes, num_sequences).astype (np.long)
    return X_data, y_data


# Hyperparameters for the synthetic data
num_sequences = 1000  # Number of sequences
seq_len = 5  # Sequence length (e.g., 5 images per sequence)
image_size = 32  # Image size (32x32)
num_channels = 3  # Number of channels (RGB images)
num_classes = 10  # Number of classes (for classification)

# Generate random image data and labels
X_data, y_data = generate_random_image_data (num_sequences, seq_len, image_size, num_channels, num_classes)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split (X_data, y_data, test_size=0.2, random_state=42)

# Convert numpy arrays to PyTorch tensors
X_train = torch.tensor (X_train)
y_train = torch.tensor (y_train)
X_test = torch.tensor (X_test)
y_test = torch.tensor (y_test)


# Define the CNN-RNN hybrid model
class CNN_RNN_Model (nn.Module):
    def __init__(self, input_channels, conv_out_channels, rnn_hidden_size, output_size, num_layers=1):
        super (CNN_RNN_Model, self).__init__ ()

        # CNN layers
        self.conv1 = nn.Conv2d (in_channels=input_channels, out_channels=conv_out_channels, kernel_size=3, stride=1,
                                padding=1)
        self.conv2 = nn.Conv2d (in_channels=conv_out_channels, out_channels=conv_out_channels, kernel_size=3, stride=1,
                                padding=1)
        self.pool = nn.MaxPool2d (kernel_size=2, stride=2, padding=0)

        # Calculate the size of the feature map after the convolutional and pooling layers
        self._to_linear = None
        self.convs (torch.randn (1, input_channels, 32, 32))

        # RNN layers (LSTM or GRU)
        self.rnn = nn.LSTM (input_size=self._to_linear, hidden_size=rnn_hidden_size, num_layers=num_layers,
                            batch_first=True)

        # Fully connected layer (after RNN)
        self.fc = nn.Linear (rnn_hidden_size, output_size)

    def convs(self, x):
        x = F.relu (self.conv1 (x))
        x = self.pool (F.relu (self.conv2 (x)))
        if self._to_linear is None:
            self._to_linear = x [0].numel ()
        return x

    def forward(self, x):
        # x: input tensor of shape (batch_size, sequence_len, channels, height, width)
        batch_size, seq_len, _, _, _ = x.shape

        # CNN operations for each time step
        cnn_out = []
        for t in range (seq_len):
            # Extract each time step (image frame)
            x_t = x [:, t, :, :, :]
            x_t = self.convs (x_t)
            x_t = x_t.view (batch_size, -1)  # Flatten the output for RNN
            cnn_out.append (x_t)

        # Stack the CNN outputs into a tensor of shape (batch_size, sequence_len, features)
        cnn_out = torch.stack (cnn_out, dim=1)

        # RNN operations
        rnn_out, _ = self.rnn (cnn_out)  # rnn_out shape: (batch_size, sequence_len, rnn_hidden_size)

        # Taking the output from the last time step
        out = self.fc (rnn_out [:, -1, :])  # Output layer
        return out


# Hyperparameters
input_channels = 3  # For RGB images
conv_out_channels = 32
rnn_hidden_size = 64
output_size = 10  # For classification tasks with 10 classes, for example

# Initialize the model
model = CNN_RNN_Model (input_channels=input_channels,
                       conv_out_channels=conv_out_channels,
                       rnn_hidden_size=rnn_hidden_size,
                       output_size=output_size)

# Define optimizer and loss function
optimizer = optim.Adam (model.parameters (), lr=0.001)
criterion = nn.CrossEntropyLoss ()

# Training loop
num_epochs = 5
batch_size = 16

# Create DataLoader for batches
train_data = torch.utils.data.TensorDataset (X_train, y_train)
train_loader = torch.utils.data.DataLoader (train_data, batch_size=batch_size, shuffle=True)

# Lists to store metrics
train_losses = []
train_accuracies = []
train_recalls = []

# Training the model
for epoch in range (num_epochs):
    model.train ()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    for i, (inputs, labels) in enumerate (train_loader):
        # Zero the parameter gradients
        optimizer.zero_grad ()

        # Forward pass
        outputs = model (inputs)

        # Convert labels to Long type
        labels = labels.long ()

        loss = criterion (outputs, labels)

        # Backward pass and optimization
        loss.backward ()
        optimizer.step ()

        # Calculate statistics
        running_loss += loss.item ()
        _, predicted = torch.max (outputs, 1)
        total += labels.size (0)
        correct += (predicted == labels).sum ().item ()
        all_labels.extend (labels.cpu ().numpy ())
        all_preds.extend (predicted.cpu ().numpy ())

    # Calculate metrics for the epoch
    epoch_loss = running_loss / len (train_loader)
    epoch_accuracy = correct / total
    epoch_recall = recall_score (all_labels, all_preds, average='macro')

    train_losses.append (epoch_loss)
    train_accuracies.append (epoch_accuracy)
    train_recalls.append (epoch_recall)

    print (
        f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, Recall: {epoch_recall:.4f}')

print ("Training complete.")

# Plotting the metrics
epochs = range (1, num_epochs + 1)
plt.figure (figsize=(12, 4))

plt.subplot (1, 3, 1)
plt.plot (epochs, train_losses, 'b', label='Training loss')
plt.title ('Training loss')
plt.xlabel ('Epochs')
plt.ylabel ('Loss')
plt.legend ()

plt.subplot (1, 3, 2)
plt.plot (epochs, train_accuracies, 'r', label='Training accuracy')
plt.title ('Training accuracy')
plt.xlabel ('Epochs')
plt.ylabel ('Accuracy')
plt.legend ()

plt.subplot (1, 3, 3)
plt.plot (epochs, train_recalls, 'g', label='Training recall')
plt.title ('Training recall')
plt.xlabel ('Epochs')
plt.ylabel ('Recall')
plt.legend ()

plt.tight_layout ()
plt.show ()
