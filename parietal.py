# parietal.py (Sensory Integration, Spatial Awareness)
import numpy as np
import json
import os


class ParietalLobeAI:
    def __init__(self, model_path="data/parietal_model.json"):
        self.input_size = 20  # Sensory data (e.g., sensor readings)
        self.hidden_size = 25  # Size of the new hidden layer
        self.output_size = (
            3  # Spatial coordinates (e.g., x, y, z error or target position)
        )

        # Learning rates
        self.learning_rate_learn = 0.01
        self.learning_rate_consolidate = 0.005

        # New two-layer architecture weights and biases
        self.weights_input_hidden = (
            np.random.randn(self.input_size, self.hidden_size) * 0.01
        )
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.weights_hidden_output = (
            np.random.randn(self.hidden_size, self.output_size) * 0.01
        )
        self.bias_output = np.zeros((1, self.output_size))

        # Memory will store (sensory_data_list, true_coords_list) tuples.
        self.memory = []
        self.max_memory_size = 100  # Max memory size

        self.model_path = model_path
        self.load_model()

    def _prepare_input_vector(self, sensor_data):
        """Prepares a 1D numpy array from sensor_data, ensuring correct size."""
        if isinstance(sensor_data, list):
            input_vec_list = sensor_data
        elif isinstance(sensor_data, np.ndarray):
            input_vec_list = sensor_data.flatten().tolist()
        else:
            # print(f"Warning: Unexpected sensor_data type {type(sensor_data)} in _prepare_input_vector. Using zeros.")
            input_vec_list = [0.0] * self.input_size

        if len(input_vec_list) < self.input_size:
            input_vec_list.extend([0.0] * (self.input_size - len(input_vec_list)))
        elif len(input_vec_list) > self.input_size:
            input_vec_list = input_vec_list[: self.input_size]
        return np.array(input_vec_list)  # Return 1D array

    def _prepare_target_coords_vector(self, true_coords_list_or_array):
        """Prepares a 1D numpy array from true_coords, ensuring correct size."""
        if isinstance(true_coords_list_or_array, list):
            target_list = true_coords_list_or_array
        elif isinstance(true_coords_list_or_array, np.ndarray):
            target_list = true_coords_list_or_array.flatten().tolist()
        else:
            # print(f"Warning: Unexpected true_coords type {type(true_coords_list_or_array)}. Using zeros.")
            target_list = [0.0] * self.output_size

        if len(target_list) < self.output_size:
            target_list.extend([0.0] * (self.output_size - len(target_list)))
        elif len(target_list) > self.output_size:
            target_list = target_list[: self.output_size]
        return np.array(target_list)  # Return 1D array, shape (self.output_size,)

    def _forward_propagate(self, sensor_data):
        input_vec_1d = self._prepare_input_vector(sensor_data)
        if input_vec_1d.shape[0] != self.input_size:
            input_vec_1d = np.zeros(self.input_size)
        input_vec_2d = input_vec_1d.reshape(1, -1)
        hidden_layer_input = input_vec_2d @ self.weights_input_hidden + self.bias_hidden
        hidden_layer_output = np.tanh(hidden_layer_input)
        output_coords = (
            hidden_layer_output @ self.weights_hidden_output + self.bias_output
        )
        return input_vec_1d, hidden_layer_output.flatten(), output_coords.flatten()

    def process_task(self, sensory_data):
        try:
            _input_features, _hidden_activation, output_coords = (
                self._forward_propagate(sensory_data)
            )
            return output_coords.tolist()
        except Exception as e:
            return [0.0] * self.output_size

    def learn(self, sensory_data, true_coords):
        """Update based on spatial error using backpropagation."""

        # Forward propagation
        fwd_results = self._forward_propagate(sensory_data)
        input_vec_1d, hidden_output_1d, predicted_coords_1d = fwd_results

        if not np.any(input_vec_1d):  # Skip if input vector is all zeros
            # print("Warning: Skipping learning in ParietalLobeAI due to zero input vector.")
            return

        true_coords_1d = self._prepare_target_coords_vector(
            true_coords
        )  # Shape (output_size,)

        # Backward Pass
        # 1. Error at output layer (delta_output)
        # For linear output layer and Mean Squared Error loss, delta_output = predicted - true
        delta_output = predicted_coords_1d - true_coords_1d  # Shape (output_size,)

        # 2. Gradients for hidden-to-output layer
        delta_weights_ho = np.outer(
            hidden_output_1d, delta_output
        )  # Shape (hidden_size, output_size)
        delta_bias_output = delta_output  # Shape (output_size,)

        # 3. Error at hidden layer (delta_hidden)
        error_propagated_to_hidden = (
            delta_output @ self.weights_hidden_output.T
        )  # Shape (hidden_size,)
        derivative_tanh_hidden = (
            1 - hidden_output_1d**2
        )  # hidden_output_1d is tanh(z_hidden)
        delta_hidden = (
            error_propagated_to_hidden * derivative_tanh_hidden
        )  # Shape (hidden_size,)

        # 4. Gradients for input-to-hidden layer
        delta_weights_ih = np.outer(
            input_vec_1d, delta_hidden
        )  # Shape (input_size, hidden_size)
        delta_bias_hidden = delta_hidden  # Shape (hidden_size,)

        # Update Weights and Biases
        self.weights_hidden_output -= self.learning_rate_learn * delta_weights_ho
        self.bias_output -= self.learning_rate_learn * delta_bias_output.reshape(1, -1)

        self.weights_input_hidden -= self.learning_rate_learn * delta_weights_ih
        self.bias_hidden -= self.learning_rate_learn * delta_bias_hidden.reshape(1, -1)

        # Update Memory
        s_data_list = (
            sensory_data.tolist()
            if isinstance(sensory_data, np.ndarray)
            else list(sensory_data)
        )
        t_coords_list = (
            true_coords.tolist()
            if isinstance(true_coords, np.ndarray)
            else list(true_coords)
        )

        self.memory.append((s_data_list, t_coords_list))
        if len(self.memory) > self.max_memory_size:
            self.memory.pop(0)

    def consolidate(self):
        """Bedtime: Replay experiences from memory to refine model."""
        if not self.memory:
            self.save_model()
            return

        # print(f"Consolidating Parietal Lobe. Memory size: {len(self.memory)}")
        for sensory_data_to_replay, true_coords_to_replay in list(self.memory):

            fwd_results = self._forward_propagate(sensory_data_to_replay)
            input_vec_1d, hidden_output_1d, predicted_coords_1d = fwd_results

            if not np.any(input_vec_1d):
                continue

            true_coords_1d = self._prepare_target_coords_vector(true_coords_to_replay)

            delta_output = predicted_coords_1d - true_coords_1d
            delta_weights_ho = np.outer(hidden_output_1d, delta_output)
            delta_bias_output = delta_output
            error_propagated_to_hidden = delta_output @ self.weights_hidden_output.T
            derivative_tanh_hidden = 1 - hidden_output_1d**2
            delta_hidden = error_propagated_to_hidden * derivative_tanh_hidden
            delta_weights_ih = np.outer(input_vec_1d, delta_hidden)
            delta_bias_hidden = delta_hidden

            self.weights_hidden_output -= (
                self.learning_rate_consolidate * delta_weights_ho
            )
            self.bias_output -= (
                self.learning_rate_consolidate * delta_bias_output.reshape(1, -1)
            )
            self.weights_input_hidden -= (
                self.learning_rate_consolidate * delta_weights_ih
            )
            self.bias_hidden -= (
                self.learning_rate_consolidate * delta_bias_hidden.reshape(1, -1)
            )

        self.save_model()

    def _initialize_default_weights_biases(self):
        self.weights_input_hidden = (
            np.random.randn(self.input_size, self.hidden_size) * 0.01
        )
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.weights_hidden_output = (
            np.random.randn(self.hidden_size, self.output_size) * 0.01
        )
        self.bias_output = np.zeros((1, self.output_size))

    def save_model(self):
        model_data = {
            "weights_input_hidden": self.weights_input_hidden.tolist(),
            "bias_hidden": self.bias_hidden.tolist(),
            "weights_hidden_output": self.weights_hidden_output.tolist(),
            "bias_output": self.bias_output.tolist(),
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
        }
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        try:
            with open(self.model_path, "w") as f:
                json.dump(model_data, f)
            # print(f"ParietalLobeAI: Model saved to {self.model_path}") # Optional success message
        except Exception as e:
            print(f"ParietalLobeAI: Error saving model to {self.model_path}: {e}")

    def load_model(self):
        if not os.path.exists(self.model_path):
            self._initialize_default_weights_biases()
            return

        try:
            with open(self.model_path, "r") as f:
                data = json.load(f)

            # Check 1: Architectural parameters
            default_size_sentinel = -1  # A value that instance sizes should not match
            loaded_input_size = data.get("input_size", default_size_sentinel)
            loaded_hidden_size = data.get("hidden_size", default_size_sentinel)
            loaded_output_size = data.get("output_size", default_size_sentinel)

            if not (
                loaded_input_size == self.input_size
                and loaded_hidden_size == self.hidden_size
                and loaded_output_size == self.output_size
            ):
                self._initialize_default_weights_biases()
                return

            # Check 2: Presence of all required new format keys
            required_keys = [
                "weights_input_hidden",
                "bias_hidden",
                "weights_hidden_output",
                "bias_output",
            ]
            if not all(key in data for key in required_keys):
                self._initialize_default_weights_biases()
                return

            # If all keys are present, attempt to load and validate them
            w_ih = np.array(data["weights_input_hidden"])
            b_h = np.array(data["bias_hidden"])
            w_ho = np.array(data["weights_hidden_output"])
            b_o = np.array(data["bias_output"])

            # Check 3: Shape of each loaded weight/bias matrix
            expected_w_ih_shape = (self.input_size, self.hidden_size)
            expected_b_h_shape = (1, self.hidden_size)
            expected_w_ho_shape = (self.hidden_size, self.output_size)
            expected_b_o_shape = (1, self.output_size)

            if not (
                w_ih.shape == expected_w_ih_shape
                and b_h.shape == expected_b_h_shape
                and w_ho.shape == expected_w_ho_shape
                and b_o.shape == expected_b_o_shape
            ):
                self._initialize_default_weights_biases()
                return

            # If all checks passed, assign the loaded weights.
            self.weights_input_hidden = w_ih
            self.bias_hidden = b_h
            self.weights_hidden_output = w_ho
            self.bias_output = b_o

        except Exception:
            self._initialize_default_weights_biases()
            return  # Explicit return for clarity


# Example Usage
if __name__ == "__main__":
    parietal_ai = ParietalLobeAI(model_path="data/test_parietal_model_backprop.json")
    print("ParietalLobeAI initialized for backpropagation test.")

    sample_sensor_data = np.random.rand(parietal_ai.input_size).tolist()
    sample_true_coords = np.random.rand(parietal_ai.output_size).tolist()

    initial_w_ih_sample = parietal_ai.weights_input_hidden[0, 0]
    initial_w_ho_sample = parietal_ai.weights_hidden_output[0, 0]

    print(f"Initial w_ih[0,0]: {initial_w_ih_sample}, w_ho[0,0]: {initial_w_ho_sample}")

    parietal_ai.learn(sample_sensor_data, sample_true_coords)
    print(f"Memory after learn: {parietal_ai.memory}")
    print(f"w_ih[0,0] after learn: {parietal_ai.weights_input_hidden[0,0]}")
    print(f"w_ho[0,0] after learn: {parietal_ai.weights_hidden_output[0,0]}")

    if (
        initial_w_ih_sample == parietal_ai.weights_input_hidden[0, 0]
        and initial_w_ho_sample == parietal_ai.weights_hidden_output[0, 0]
    ):
        print(
            "Warning: Weights did not change after learn. This might be okay if error was zero or input was zero."
        )
    else:
        print("Weights changed after learn, as expected.")

    parietal_ai.consolidate()
    print(f"w_ih[0,0] after consolidate: {parietal_ai.weights_input_hidden[0,0]}")
    print(f"w_ho[0,0] after consolidate: {parietal_ai.weights_hidden_output[0,0]}")

    if os.path.exists("data/test_parietal_model_backprop.json"):
        os.remove("data/test_parietal_model_backprop.json")
    print("ParietalLobeAI backpropagation test finished.")
