# occipital.py (Visual Processing)
import numpy as np
import json
import os

class OccipitalLobeAI:
    def __init__(self, model_path="data/occipital_model.json"):
        self.input_size = 25  # Pixel features
        self.output_size = 5  # Object labels
        self.weights = np.random.randn(self.input_size, self.output_size) * 0.01
        self.memory = []  # Store (image, label) pairs
        self.model_path = model_path
        self.load_model()

    def process_task(self, image_data):
        """Identify objects in image."""
        input_vec = np.array(image_data).reshape(1, -1)
        output = input_vec @ self.weights
        return np.argmax(output)  # Return predicted label

    def learn(self, image_data, true_label):
        """Update based on visual feedback."""
        input_vec = np.array(image_data).reshape(1, -1)
        predicted = self.process_task(image_data)
        error = np.zeros(self.output_size)
        error[true_label] = 1 - predicted
        self.memory.append((image_data, error))
        if len(self.memory) > 100:
            self.memory.pop(0)
        self.weights += 0.01 * input_vec.T @ error.reshape(1, -1)

    def consolidate(self):
        """Bedtime: Refine visual model."""
        for image_data, error in self.memory:
            input_vec = np.array(image_data).reshape(1, -1)
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
