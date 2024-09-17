import torch
import numpy as np


# Define the ComplexNN model class (assuming it's the same as in your other scripts)
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


# Load the pre-trained model
model_path = 'models/complex_nn_model.pkl'
model = ComplexNN ()
model.load_state_dict (torch.load (model_path))
model.eval ()


# Generate synthetic test data
def generate_test_data(num_samples=10):
    x_test = np.random.uniform (-5, 5, size=(num_samples, 1))
    return torch.tensor (x_test, dtype=torch.float32)


# Run the model on test data
def run_model_on_test_data(model, x_test):
    with torch.no_grad ():
        y_pred = model (x_test)
    return y_pred


# Main function to test the model
if __name__ == "__main__":
    # Generate test data
    x_test = generate_test_data ()

    # Run the model on test data
    y_pred = run_model_on_test_data (model, x_test)

    # Print the results
    print ("Test Data (x):", x_test.numpy ().flatten ())
    print ("Model Predictions (y):", y_pred.numpy ().flatten ())
