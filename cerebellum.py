# cerebellum.py (Motor Control, Coordination)
import numpy as np
import json
import os


class CerebellumAI:
    def __init__(self, model_path="data/cerebellum_model.json"):
        self.input_size = 10  # Sensor feedback
        self.output_size = 3  # Motor commands
        self.weights = np.random.randn(self.input_size, self.output_size) * 0.01
        self.memory = []  # Store (input, output, error) tuples
        self.model_path = model_path
        self.load_model()

    def process_task(self, sensor_data):
        """Generate motor command."""
        input_vec = np.array(sensor_data).reshape(1, -1)
        command = input_vec @ self.weights
        return command.flatten().tolist()

    def learn(self, sensor_data, true_command):
        """Update based on motor error."""
        input_vec = np.array(sensor_data).reshape(1, -1)
        predicted = self.process_task(sensor_data)
        error = np.array(true_command) - predicted
        self.memory.append((sensor_data, error))
        if len(self.memory) > 100:
            self.memory.pop(0)
        self.weights += 0.01 * input_vec.T @ error.reshape(1, -1)

    def consolidate(self):
        """Bedtime: Optimize motor commands."""
        for sensor_data, error in self.memory:
            input_vec = np.array(sensor_data).reshape(1, -1)
            self.weights += 0.005 * input_vec.T @ error.reshape(1, -1)
        self.save_model()

    def save_model(self):
        with open(self.model_path, "w") as f:
            json.dump({"weights": self.weights.tolist()}, f)

    def load_model(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, "r") as f:
                data = json.load(f)
                self.weights = np.array(data["weights"])
