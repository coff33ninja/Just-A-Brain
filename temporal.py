# temporal.py (Memory, Language)
import numpy as np
import json
import os


class TemporalLobeAI:
    def __init__(
        self,
        model_path="data/temporal_model.json",
        memory_path="data/temporal_memory.json",
    ):
        self.input_size = 15  # Text/audio features
        self.output_size = 10  # Word embeddings
        self.weights = np.random.randn(self.input_size, self.output_size) * 0.01
        self.memory_db = []  # Store input-output pairs
        self.model_path = model_path
        self.memory_path = memory_path
        self.load_model()

    def process_task(self, input_data):
        """Process input (e.g., store or recall word)."""
        input_vec = np.array(input_data).reshape(1, -1)
        output = input_vec @ self.weights
        return output.flatten().tolist()

    def learn(self, input_data, target_output):
        """Store memory and update weights."""
        input_vec = np.array(input_data).reshape(1, -1)
        target = np.array(target_output).reshape(1, -1)
        self.memory_db.append((input_data, target_output))
        if len(self.memory_db) > 100:
            self.memory_db.pop(0)
        error = target - (input_vec @ self.weights)
        self.weights += 0.01 * input_vec.T @ error

    def consolidate(self):
        """Bedtime: Reorganize memories and refine model."""
        for input_data, target_output in self.memory_db:
            self.learn(input_data, target_output)
        self.save_model()
        self.save_memory()

    def save_model(self):
        with open(self.model_path, "w") as f:
            json.dump({"weights": self.weights.tolist()}, f)

    def save_memory(self):
        with open(self.memory_path, "w") as f:
            json.dump(self.memory_db, f)

    def load_model(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, "r") as f:
                data = json.load(f)
                self.weights = np.array(data["weights"])
        if os.path.exists(self.memory_path):
            with open(self.memory_path, "r") as f:
                self.memory_db = json.load(f)
