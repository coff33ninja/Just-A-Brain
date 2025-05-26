# cerebellum.py (Motor Control, Coordination)
import numpy as np
import json
import os

class CerebellumAI:
    def __init__(self, model_path="data/cerebellum_model.json"):
        self.input_size = 10  # Sensor feedback
        self.hidden_size = 20 # Size of the new hidden layer
        self.output_size = 3  # Motor commands (e.g., scaled between -1 and 1)

        self.learning_rate_learn = 0.01
        self.learning_rate_consolidate = 0.005

        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.bias_output = np.zeros((1, self.output_size))

        self.memory = []  # Stores (sensor_data_list, true_command_list)
        self.model_path = model_path
        self.load_model()

    def _prepare_input_vector(self, sensor_data):
        """Prepares a 1D numpy array from sensor_data, ensuring correct size."""
        if isinstance(sensor_data, list):
            input_vec_list = sensor_data
        elif isinstance(sensor_data, np.ndarray):
            input_vec_list = sensor_data.flatten().tolist()
        else:
            input_vec_list = [0.0] * self.input_size

        if len(input_vec_list) < self.input_size:
            input_vec_list.extend([0.0] * (self.input_size - len(input_vec_list)))
        elif len(input_vec_list) > self.input_size:
            input_vec_list = input_vec_list[:self.input_size]
        return np.array(input_vec_list)

    def _prepare_target_command_vector(self, true_command_list):
        """Prepares a 1D numpy array from true_command_list, ensuring correct size."""
        if isinstance(true_command_list, list):
            target_list = true_command_list
        elif isinstance(true_command_list, np.ndarray):
            target_list = true_command_list.flatten().tolist()
        else:
            # print(f"Warning: Unexpected true_command_list type {type(true_command_list)}. Using zeros.")
            target_list = [0.0] * self.output_size

        if len(target_list) < self.output_size:
            target_list.extend([0.0] * (self.output_size - len(target_list)))
        elif len(target_list) > self.output_size:
            target_list = target_list[:self.output_size]
        return np.array(target_list) # Return 1D array, shape (self.output_size,)


    def _forward_propagate(self, sensor_data):
        input_vec_1d = self._prepare_input_vector(sensor_data)
        if input_vec_1d.shape[0] != self.input_size:
            input_vec_1d = np.zeros(self.input_size)
        input_vec_2d = input_vec_1d.reshape(1, -1)

        hidden_layer_input = input_vec_2d @ self.weights_input_hidden + self.bias_hidden
        hidden_layer_output = np.tanh(hidden_layer_input)

        output_layer_scores = hidden_layer_output @ self.weights_hidden_output + self.bias_output
        final_commands = np.tanh(output_layer_scores)

        return input_vec_1d, hidden_layer_output.flatten(), final_commands.flatten()

    def process_task(self, sensor_data):
        try:
            _input_features, _hidden_activation, final_commands = self._forward_propagate(sensor_data)
            return final_commands.tolist()
        except Exception as e:
            return [0.0] * self.output_size

    def learn(self, sensor_data, true_command_list_or_array):
        """Update weights based on motor error using backpropagation."""

        # Forward propagation
        input_vec_1d, hidden_output_1d, final_commands_1d = self._forward_propagate(sensor_data)

        # If input_vec_1d is all zeros (e.g., from file not found or bad sensor data), skip learning
        if not np.any(input_vec_1d):
            # print(f"Warning: Skipping learning for sensor_data due to zero input vector.")
            return

        # Prepare true_command_1d
        true_command_1d = self._prepare_target_command_vector(true_command_list_or_array) # Shape (output_size,)

        # Backward pass
        # 1. Error at output layer (delta_output)
        error_signal = true_command_1d - final_commands_1d  # Shape (output_size,)
        derivative_tanh_output = 1 - final_commands_1d**2  # Derivative of output tanh
        delta_output = error_signal * derivative_tanh_output  # Shape (output_size,)

        # 2. Gradients for hidden-to-output layer
        delta_weights_ho = np.outer(hidden_output_1d, delta_output)  # Shape (hidden_size, output_size)
        delta_bias_output = delta_output  # Shape (output_size,)

        # 3. Error at hidden layer (delta_hidden)
        error_propagated_to_hidden = delta_output @ self.weights_hidden_output.T  # Shape (hidden_size,)
        derivative_tanh_hidden = 1 - hidden_output_1d**2  # Derivative of hidden layer tanh
        delta_hidden = error_propagated_to_hidden * derivative_tanh_hidden  # Shape (hidden_size,)

        # 4. Gradients for input-to-hidden layer
        delta_weights_ih = np.outer(input_vec_1d, delta_hidden)  # Shape (input_size, hidden_size)
        delta_bias_hidden = delta_hidden  # Shape (hidden_size,)

        # Update Weights and Biases
        self.weights_hidden_output += self.learning_rate_learn * delta_weights_ho
        self.bias_output += self.learning_rate_learn * delta_bias_output.reshape(1, -1)

        self.weights_input_hidden += self.learning_rate_learn * delta_weights_ih
        self.bias_hidden += self.learning_rate_learn * delta_bias_hidden.reshape(1, -1)

        # Update Memory (store original list/array forms)
        # Ensure consistent storage format (e.g., always lists)
        s_data_list = sensor_data.tolist() if isinstance(sensor_data, np.ndarray) else list(sensor_data)
        t_cmd_list = true_command_list_or_array.tolist() if isinstance(true_command_list_or_array, np.ndarray) else list(true_command_list_or_array)

        self.memory.append((s_data_list, t_cmd_list))
        if len(self.memory) > 100:
            self.memory.pop(0)

    def consolidate(self):
        """Bedtime: Replay experiences from memory to refine model."""
        # print(f"Consolidating Cerebellum. Memory size: {len(self.memory)}")
        if not self.memory:
            self.save_model()
            return

        for sensor_data_list, true_command_list_from_mem in list(self.memory):
            # Forward propagation
            input_vec_1d, hidden_output_1d, final_commands_1d = self._forward_propagate(sensor_data_list)
            if not np.any(input_vec_1d): # Skip if image processing failed
                continue

            true_command_1d = self._prepare_target_command_vector(true_command_list_from_mem)

            # Backward pass (same logic as in learn method)
            error_signal = true_command_1d - final_commands_1d
            derivative_tanh_output = 1 - final_commands_1d**2
            delta_output = error_signal * derivative_tanh_output

            delta_weights_ho = np.outer(hidden_output_1d, delta_output)
            delta_bias_output = delta_output

            error_propagated_to_hidden = delta_output @ self.weights_hidden_output.T
            derivative_tanh_hidden = 1 - hidden_output_1d**2
            delta_hidden = error_propagated_to_hidden * derivative_tanh_hidden

            delta_weights_ih = np.outer(input_vec_1d, delta_hidden)
            delta_bias_hidden = delta_hidden

            # Update weights and biases using consolidation learning rate
            self.weights_hidden_output += self.learning_rate_consolidate * delta_weights_ho
            self.bias_output += self.learning_rate_consolidate * delta_bias_output.reshape(1, -1)

            self.weights_input_hidden += self.learning_rate_consolidate * delta_weights_ih
            self.bias_hidden += self.learning_rate_consolidate * delta_bias_hidden.reshape(1, -1)

        self.save_model()

    def _initialize_default_weights_biases(self):
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.bias_output = np.zeros((1, self.output_size))

    def save_model(self):
        model_data = {
            'weights_input_hidden': self.weights_input_hidden.tolist(),
            'bias_hidden': self.bias_hidden.tolist(),
            'weights_hidden_output': self.weights_hidden_output.tolist(),
            'bias_output': self.bias_output.tolist(),
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size
        }
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "w") as f:
            json.dump(model_data, f)

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "r") as f: data = json.load(f)
                loaded_input_size = data.get('input_size', self.input_size)
                loaded_hidden_size = data.get('hidden_size', self.hidden_size)
                loaded_output_size = data.get('output_size', self.output_size)
                if not (loaded_input_size == self.input_size and \
                        loaded_hidden_size == self.hidden_size and \
                        loaded_output_size == self.output_size):
                    self._initialize_default_weights_biases(); return
                required_keys = ['weights_input_hidden', 'bias_hidden', 'weights_hidden_output', 'bias_output']
                if all(key in data for key in required_keys):
                    w_ih = np.array(data['weights_input_hidden']); b_h = np.array(data['bias_hidden'])
                    w_ho = np.array(data['weights_hidden_output']); b_o = np.array(data['bias_output'])
                    if w_ih.shape == (self.input_size, self.hidden_size) and \
                       b_h.shape == (1, self.hidden_size) and \
                       w_ho.shape == (self.hidden_size, self.output_size) and \
                       b_o.shape == (1, self.output_size):
                        self.weights_input_hidden = w_ih; self.bias_hidden = b_h
                        self.weights_hidden_output = w_ho; self.bias_output = b_o
                    else: self._initialize_default_weights_biases()
                elif 'weights' in data: self._initialize_default_weights_biases()
                else: self._initialize_default_weights_biases()
            except Exception: self._initialize_default_weights_biases()
        else: self._initialize_default_weights_biases()

# Example Usage
if __name__ == "__main__":
    cerebellum_ai = CerebellumAI(model_path="data/test_cerebellum_model_backprop.json")
    print(f"CerebellumAI initialized for backpropagation test.")

    sample_sensor_data = np.random.rand(cerebellum_ai.input_size).tolist()
    sample_true_command = (np.random.rand(cerebellum_ai.output_size) * 2 - 1).tolist() # Commands between -1 and 1

    initial_w_ih_sample = cerebellum_ai.weights_input_hidden[0,0]
    initial_w_ho_sample = cerebellum_ai.weights_hidden_output[0,0]

    print(f"Initial w_ih[0,0]: {initial_w_ih_sample}, w_ho[0,0]: {initial_w_ho_sample}")

    # Perform learning
    cerebellum_ai.learn(sample_sensor_data, sample_true_command)
    print(f"Memory after learn: {cerebellum_ai.memory}")
    print(f"w_ih[0,0] after learn: {cerebellum_ai.weights_input_hidden[0,0]}")
    print(f"w_ho[0,0] after learn: {cerebellum_ai.weights_hidden_output[0,0]}")
    assert initial_w_ih_sample != cerebellum_ai.weights_input_hidden[0,0] or \
           initial_w_ho_sample != cerebellum_ai.weights_hidden_output[0,0], \
           "Weights did not change after learn."

    # Perform consolidation
    cerebellum_ai.consolidate()
    print(f"w_ih[0,0] after consolidate: {cerebellum_ai.weights_input_hidden[0,0]}")
    print(f"w_ho[0,0] after consolidate: {cerebellum_ai.weights_hidden_output[0,0]}")

    if os.path.exists("data/test_cerebellum_model_backprop.json"):
        os.remove("data/test_cerebellum_model_backprop.json")
    print("CerebellumAI backpropagation test finished.")
