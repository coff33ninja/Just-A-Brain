# limbic.py (Emotion, Motivation)
import numpy as np
import json
import os


class LimbicSystemAI:
    def __init__(self, model_path="data/limbic_model.json"):
        self.input_size = 10  # Tone/text features
        self.output_size = 3  # Emotion labels (e.g., happy, urgent, sad)
        self.weights = np.random.randn(self.input_size, self.output_size) * 0.01
        self.memory = []  # Store (input, emotion, reward) tuples
        self.model_path = model_path
        self.load_model()

    def process_task(self, input_data):
        """Detect emotional context."""
        input_vec = np.array(input_data).reshape(1, -1)
        output = input_vec @ self.weights
        return np.argmax(output)  # Return emotion label

    def learn(self, input_data, true_emotion, reward):
        """Update based on emotional feedback."""
        input_vec = np.array(input_data).reshape(1, -1)
        predicted = self.process_task(input_data)
        error = np.zeros(self.output_size)
        error[true_emotion] = 1 - predicted
        self.memory.append((input_data, error, reward))
        if len(self.memory) > 100:
            self.memory.pop(0)
        self.weights += 0.01 * reward * input_vec.T @ error.reshape(1, -1)

    def consolidate(self):
        """Bedtime: Refine emotional model."""
        for input_data, error, reward in self.memory:
            input_vec = np.array(input_data).reshape(1, -1)
            self.weights += 0.005 * reward * input_vec.T @ error.reshape(1, -1)
        self.save_model()

    def save_model(self):
        with open(self.model_path, "w") as f:
            json.dump({"weights": self.weights.tolist()}, f)

    def load_model(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, "r") as f:
                data = json.load(f)
                self.weights = np.array(data["weights"])
