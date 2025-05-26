# parietal.py (Sensory Integration, Spatial Awareness)
import numpy as np
import json
import os

class ParietalLobeAI:
    def __init__(self, model_path="data/parietal_model.json"):
        self.input_size = 20  # Sensory data (e.g., sensor readings)
        self.output_size = 3  # Spatial coordinates
        self.weights = np.random.randn(self.input_size, self.output_size) * 0.01
        self.memory = []  # Store (input, output, error) tuples
        self.model_path = model_path
        self.load_model()

    def process_task(self, sensory_data):
        """Map sensory data to spatial coordinates."""
        input_vec = np.array(sensory_data).reshape(1, -1)
        coords = input_vec @ self.weights
        return coords.flatten().tolist()

    def learn(self, sensory_data, true_coords):
        """Update based on spatial error."""
        input_vec = np.array(sensory_data).reshape(1, -1)
        predicted = self.process_task(sensory_data)
        error = np.array(true_coords) - predicted
        self.memory.append((sensory_data, error))
        if len(self.memory) > 100:
            self.memory.pop(0)
        # Update weights (gradient descent)
        self.weights += 0.01 * input_vec.T @ error.reshape(1, -1)

    def consolidate(self):
        """Bedtime: Refine spatial model."""
        for sensory_data, error in self.memory:
            input_vec = np.array(sensory_data).reshape(1, -1)
            self.weights += 0.005 * input_vec.T @ error.reshape(1, -1)
        self.save_model()

    def save_model(self):
        with open(self.model_path, 'w') as f:
            json.dump({'weights': self.weights.tolist()}, f)

    def load_model(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, 'r') as f:
                data = json.load(f)
                self.weights = np.array(data['weights'])
