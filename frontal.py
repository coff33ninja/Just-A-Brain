# frontal.py (Decision-Making, Planning)
import numpy as np
import json
import os

class FrontalLobeAI:
    def __init__(self, model_path="data/frontal_model.json"):
        self.input_size = 10  # Simplified input (e.g., task features)
        self.output_size = 5  # Action choices
        self.weights = np.random.randn(self.input_size, self.output_size) * 0.01  # Tiny model
        self.memory = []  # Store (input, action, reward) tuples
        self.model_path = model_path
        self.load_model()

    def process_task(self, input_data):
        """Choose an action based on input (e.g., plan a step)."""
        input_vec = np.array(input_data).reshape(1, -1)
        output = input_vec @ self.weights  # Simple forward pass
        action = np.argmax(output)  # Pick best action
        return action

    def learn(self, input_data, action, reward):
        """Update weights based on reward (simple RL)."""
        self.memory.append((input_data, action, reward))
        if len(self.memory) > 100:  # Limit memory size
            self.memory.pop(0)
        # Simple weight update (gradient approximation)
        input_vec = np.array(input_data).reshape(1, -1)
        self.weights += 0.01 * reward * input_vec.T @ np.ones((1, self.output_size))

    def consolidate(self):
        """Bedtime: Replay experiences and save model."""
        for input_data, action, reward in self.memory:
            self.learn(input_data, action, reward * 0.5)  # Replay with reduced impact
        self.save_model()

    def save_model(self):
        """Save weights to disk."""
        with open(self.model_path, 'w') as f:
            json.dump({'weights': self.weights.tolist()}, f)

    def load_model(self):
        """Load weights from disk."""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'r') as f:
                data = json.load(f)
                self.weights = np.array(data['weights'])
