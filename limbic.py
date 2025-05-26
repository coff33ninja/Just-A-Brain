# limbic.py (Emotion, Motivation)
import numpy as np
import json
import os


# Helper function for Softmax (module-level)
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    if x.ndim == 1:  # Vector
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)
    elif x.ndim == 2:  # Batch of vectors
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)
    else:
        raise ValueError("Input x must be 1D or 2D.")


class LimbicSystemAI:
    def __init__(self, model_path="data/limbic_model.json"):
        self.input_size = (
            10  # Now expects processed output from TemporalLobeAI (size 10)
        )
        self.hidden_size = 15  # Size of the new hidden layer
        self.output_size = 3  # Emotion labels (e.g., happy, urgent, sad)

        self.learning_rate_learn = 0.01
        self.learning_rate_consolidate = 0.005

        self.weights_input_hidden = (
            np.random.randn(self.input_size, self.hidden_size) * 0.01
        )
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.weights_hidden_output = (
            np.random.randn(self.hidden_size, self.output_size) * 0.01
        )
        self.bias_output = np.zeros((1, self.output_size))

        # Memory stores (processed_temporal_data_list, true_emotion_label, reward)
        self.memory = []
        self.max_memory_size = 100  # Max memory size for experience replay

        self.model_path = model_path
        self.load_model()

    def _ensure_input_vector_shape(self, data_list):
        """Pads or truncates data_list to match self.input_size. Returns 1D array."""
        if not isinstance(data_list, list):
            if isinstance(data_list, np.ndarray):
                data_list = data_list.flatten().tolist()
            else:
                data_list = [0.0] * self.input_size

        if len(data_list) < self.input_size:
            data_list.extend([0.0] * (self.input_size - len(data_list)))
        elif len(data_list) > self.input_size:
            data_list = data_list[: self.input_size]
        return np.array(data_list)

    def _forward_propagate(self, processed_temporal_data):
        input_vec_1d = self._ensure_input_vector_shape(processed_temporal_data)
        if input_vec_1d.shape[0] != self.input_size:
            input_vec_1d = np.zeros(self.input_size)
        input_vec_2d = input_vec_1d.reshape(1, -1)
        hidden_layer_input = input_vec_2d @ self.weights_input_hidden + self.bias_hidden
        hidden_layer_output = np.tanh(hidden_layer_input)
        output_scores = (
            hidden_layer_output @ self.weights_hidden_output + self.bias_output
        )
        return input_vec_1d, hidden_layer_output.flatten(), output_scores.flatten()

    def process_task(self, processed_temporal_data):
        try:
            _input_features, _hidden_activation, output_scores = (
                self._forward_propagate(processed_temporal_data)
            )
            return np.argmax(output_scores)
        except Exception as e:
            return np.random.randint(self.output_size)

    def learn(self, processed_temporal_data, true_emotion_label, reward):
        """Update weights based on emotional feedback using backpropagation."""

        # Forward propagation
        fwd_results = self._forward_propagate(processed_temporal_data)
        input_vec_1d, hidden_output_1d, output_scores_1d = fwd_results

        if not np.any(
            input_vec_1d
        ):  # Skip if input vector is all zeros (e.g., error in data processing)
            # print("Warning: Skipping learning in LimbicSystemAI due to zero input vector.")
            return

        # Prepare true_emotion_one_hot
        true_emotion_one_hot = np.zeros(self.output_size)
        if 0 <= true_emotion_label < self.output_size:
            true_emotion_one_hot[true_emotion_label] = 1.0
        else:
            # print(f"Warning: Invalid true_emotion_label ({true_emotion_label}) in LimbicSystemAI.learn. Using zero target.")
            # If true_emotion_label is invalid, true_emotion_one_hot remains all zeros.
            # The error will be based on the prediction vs. this zero target, which might not be ideal
            # but avoids crashing. An alternative is to return here.
            pass

        # Backward Pass
        output_probabilities = softmax(output_scores_1d)  # Shape (output_size,)

        # Error at output layer (delta_output) - Cross-entropy loss derivative with softmax
        delta_output = (
            output_probabilities - true_emotion_one_hot
        )  # Shape (output_size,)

        # Gradients for hidden-to-output layer
        delta_weights_ho = np.outer(
            hidden_output_1d, delta_output
        )  # Shape (hidden_size, output_size)
        delta_bias_output = delta_output  # Shape (output_size,)

        # Error at hidden layer (delta_hidden)
        error_propagated_to_hidden = (
            delta_output @ self.weights_hidden_output.T
        )  # Shape (hidden_size,)
        derivative_tanh_hidden = (
            1 - hidden_output_1d**2
        )  # hidden_output_1d is tanh(z_hidden)
        delta_hidden = (
            error_propagated_to_hidden * derivative_tanh_hidden
        )  # Shape (hidden_size,)

        # Gradients for input-to-hidden layer
        delta_weights_ih = np.outer(
            input_vec_1d, delta_hidden
        )  # Shape (input_size, hidden_size)
        delta_bias_hidden = delta_hidden  # Shape (hidden_size,)

        # Update Weights and Biases (scaled by reward)
        self.weights_hidden_output -= (
            self.learning_rate_learn * reward * delta_weights_ho
        )
        self.bias_output -= (
            self.learning_rate_learn * reward * delta_bias_output.reshape(1, -1)
        )

        self.weights_input_hidden -= (
            self.learning_rate_learn * reward * delta_weights_ih
        )
        self.bias_hidden -= (
            self.learning_rate_learn * reward * delta_bias_hidden.reshape(1, -1)
        )

        # Update Memory
        data_to_store = (
            processed_temporal_data.tolist()
            if isinstance(processed_temporal_data, np.ndarray)
            else list(processed_temporal_data)
        )
        self.memory.append((data_to_store, true_emotion_label, reward))
        if len(self.memory) > self.max_memory_size:
            self.memory.pop(0)

    def consolidate(self):
        """Bedtime: Replay experiences from memory to refine model."""
        if not self.memory:
            self.save_model()
            return

        # print(f"Consolidating Limbic System. Memory size: {len(self.memory)}")
        # For simplicity, re-learning on all memory items. Batching could be added.
        for data_to_relearn, true_emotion_label_from_mem, reward_from_mem in list(
            self.memory
        ):

            # Forward propagation
            fwd_results = self._forward_propagate(data_to_relearn)
            input_vec_1d, hidden_output_1d, output_scores_1d = fwd_results

            if not np.any(input_vec_1d):
                continue  # Skip if input vector is effectively empty

            # Prepare true_emotion_one_hot
            true_emotion_one_hot = np.zeros(self.output_size)
            if 0 <= true_emotion_label_from_mem < self.output_size:
                true_emotion_one_hot[true_emotion_label_from_mem] = 1.0
            else:
                continue  # Skip if label from memory is invalid

            # Backward Pass
            output_probabilities = softmax(output_scores_1d)
            delta_output = output_probabilities - true_emotion_one_hot

            delta_weights_ho = np.outer(hidden_output_1d, delta_output)
            delta_bias_output = delta_output

            error_propagated_to_hidden = delta_output @ self.weights_hidden_output.T
            derivative_tanh_hidden = 1 - hidden_output_1d**2
            delta_hidden = error_propagated_to_hidden * derivative_tanh_hidden

            delta_weights_ih = np.outer(input_vec_1d, delta_hidden)
            delta_bias_hidden = delta_hidden

            # Update Weights and Biases (using consolidation learning rate and stored reward)
            self.weights_hidden_output -= (
                self.learning_rate_consolidate * reward_from_mem * delta_weights_ho
            )
            self.bias_output -= (
                self.learning_rate_consolidate
                * reward_from_mem
                * delta_bias_output.reshape(1, -1)
            )
            self.weights_input_hidden -= (
                self.learning_rate_consolidate * reward_from_mem * delta_weights_ih
            )
            self.bias_hidden -= (
                self.learning_rate_consolidate
                * reward_from_mem
                * delta_bias_hidden.reshape(1, -1)
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
        with open(self.model_path, "w") as f:
            json.dump(model_data, f)

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "r") as f:
                    data = json.load(f)
                loaded_input_size = data.get("input_size", self.input_size)
                loaded_hidden_size = data.get("hidden_size", self.hidden_size)
                loaded_output_size = data.get("output_size", self.output_size)
                if not (
                    loaded_input_size == self.input_size
                    and loaded_hidden_size == self.hidden_size
                    and loaded_output_size == self.output_size
                ):
                    self._initialize_default_weights_biases()
                    return
                required_keys = [
                    "weights_input_hidden",
                    "bias_hidden",
                    "weights_hidden_output",
                    "bias_output",
                ]
                if all(key in data for key in required_keys):
                    w_ih = np.array(data["weights_input_hidden"])
                    b_h = np.array(data["bias_hidden"])
                    w_ho = np.array(data["weights_hidden_output"])
                    b_o = np.array(data["bias_output"])
                    if (
                        w_ih.shape == (self.input_size, self.hidden_size)
                        and b_h.shape == (1, self.hidden_size)
                        and w_ho.shape == (self.hidden_size, self.output_size)
                        and b_o.shape == (1, self.output_size)
                    ):
                        self.weights_input_hidden = w_ih
                        self.bias_hidden = b_h
                        self.weights_hidden_output = w_ho
                        self.bias_output = b_o
                    else:
                        self._initialize_default_weights_biases()
                elif "weights" in data:
                    self._initialize_default_weights_biases()  # Old format
                else:
                    self._initialize_default_weights_biases()
            except Exception:
                self._initialize_default_weights_biases()
        else:
            self._initialize_default_weights_biases()


# Example Usage
if __name__ == "__main__":
    limbic_ai = LimbicSystemAI(model_path="data/test_limbic_model_backprop.json")
    print(f"LimbicSystemAI initialized for backpropagation test.")

    sample_temporal_output = np.random.rand(limbic_ai.input_size).tolist()  # Size 10
    true_emotion = 1  # e.g. "urgent"
    reward_val = 1.0

    initial_w_ih_sample = limbic_ai.weights_input_hidden[0, 0]
    initial_w_ho_sample = limbic_ai.weights_hidden_output[0, 0]

    print(f"Initial w_ih[0,0]: {initial_w_ih_sample}, w_ho[0,0]: {initial_w_ho_sample}")

    limbic_ai.learn(sample_temporal_output, true_emotion, reward_val)
    print(f"Memory after learn: {limbic_ai.memory}")
    print(f"w_ih[0,0] after learn: {limbic_ai.weights_input_hidden[0,0]}")
    print(f"w_ho[0,0] after learn: {limbic_ai.weights_hidden_output[0,0]}")

    if (
        initial_w_ih_sample == limbic_ai.weights_input_hidden[0, 0]
        and initial_w_ho_sample == limbic_ai.weights_hidden_output[0, 0]
    ):
        print(
            "Warning: Weights did not change after learn. This might be okay if error was zero or input was zero."
        )
    else:
        print("Weights changed after learn, as expected.")

    limbic_ai.consolidate()
    print(f"w_ih[0,0] after consolidate: {limbic_ai.weights_input_hidden[0,0]}")
    print(f"w_ho[0,0] after consolidate: {limbic_ai.weights_hidden_output[0,0]}")

    if os.path.exists("data/test_limbic_model_backprop.json"):
        os.remove("data/test_limbic_model_backprop.json")
    print("LimbicSystemAI backpropagation test finished.")
