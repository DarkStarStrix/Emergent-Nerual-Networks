import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from Scripts.ENN import CNN_RNN_Model
from Scripts.generate_configs import generate_configs
from Scripts.synthetic_data import generate_sinusoidal_data
from Scripts.prediction_engine import LogisticRegressionModel, PredictionEngine
from Scripts.meta_learning import SimpleNN as MetaLearningEngine
from Scripts.emergent_behavior_detection import detect_emergent_behavior as EmergentBehaviorDetectionEngine


def main():
    configs = generate_configs ()
    for config in configs:
        learning_rate, batch_size, epochs = config
        model = EmergingNeuralNetwork ()
        prediction_engine = PredictionEngine (model, learning_rate)

        X, y = generate_sinusoidal_data (1000, 28 * 28)
        X_tensor = torch.tensor (X, dtype=torch.float32).view (-1, 1, 28, 28)
        y_tensor = torch.tensor (y, dtype=torch.long)
        dataset = TensorDataset (X_tensor, y_tensor)
        train_loader = DataLoader (dataset, batch_size=batch_size, shuffle=True)

        prediction_engine.train (train_loader)
        test_loss, accuracy = prediction_engine.evaluate (train_loader)
        print (f'Config: {config}, Test Loss: {test_loss}, Accuracy: {accuracy}')

        features = X
        normalized_features = prediction_engine.normalize_features (features)
        logistic_model = prediction_engine.logistic_regression (normalized_features, y)

        meta_learning_engine = MetaLearningEngine ()
        patterns = meta_learning_engine.learn_patterns (normalized_features)
        mean, std_dev = meta_learning_engine.create_probability_distribution (patterns)

        emergent_behavior_detection_engine = EmergentBehaviorDetectionEngine (mean, std_dev)
        emergent_count = 0
        for prediction in logistic_model.predict (normalized_features):
            if emergent_behavior_detection_engine.detect_emergent_behavior (prediction):
                print (f'Emergent phenomenon detected for prediction: {prediction}')
                emergent_count += 1
                if emergent_count >= 10:
                    print ("Stopping after detecting 10 emergent phenomena.")
                    break

        plt.figure (figsize=(10, 5))
        plt.plot (range (len (y)), y, label='True Labels', alpha=0.6)
        plt.plot (range (len (y)), logistic_model.predict (normalized_features), label='Predicted Labels', alpha=0.6)
        plt.legend ()
        plt.title (f'Config: {config}, Test Loss: {test_loss}, Accuracy: {accuracy}')
        plt.show ()


if __name__ == "__main__":
    main ()
