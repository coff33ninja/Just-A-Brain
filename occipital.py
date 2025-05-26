# occipital.py (Visual Processing)
import numpy as np
import json
import os
from PIL import Image

# Helper function for Softmax
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

class OccipitalLobeAI:
    def __init__(self, model_path="data/occipital_model.json"):
        self.image_size = (32, 32)
        self.grid_size = (4, 4)
        self.input_size = self.grid_size[0] * self.grid_size[1]  # 16 features
        self.hidden_size = 32
        self.output_size = 5  # Object labels

        self.learning_rate_learn = 0.01
        self.learning_rate_consolidate = 0.005

        # Initialize weights and biases - these will be set by _initialize_default_weights_biases or load_model
        self.weights_input_hidden = None
        self.bias_hidden = None
        self.weights_hidden_output = None
        self.bias_output = None

        self.memory = []  # Stores (image_path, true_label)
        self.model_path = model_path
        self.load_model() # Attempt to load, will initialize if no valid model found

    def _initialize_default_weights_biases(self):
        # print("Initializing default weights and biases for OccipitalLobeAI.")
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.bias_output = np.zeros((1, self.output_size))

    def _extract_regional_features(self, image_array):
        img_height, img_width = image_array.shape
        grid_rows, grid_cols = self.grid_size
        cell_height = img_height // grid_rows
        cell_width = img_width // grid_cols
        features = []
        for i in range(grid_rows):
            for j in range(grid_cols):
                region = image_array[i*cell_height:(i+1)*cell_height, j*cell_width:(j+1)*cell_width]
                features.append(np.mean(region))
        return np.array(features)

    def _process_image_to_vector(self, image_path):
        try:
            img = Image.open(image_path).convert('L')
            img = img.resize(self.image_size)
            img_array = np.array(img) / 255.0
            feature_vector = self._extract_regional_features(img_array)
            return feature_vector
        except FileNotFoundError:
            return np.zeros(self.input_size)
        except Exception as e:
            return np.zeros(self.input_size)

    def _forward_propagate(self, image_path):
        input_vec_1d = self._process_image_to_vector(image_path)
        if input_vec_1d.shape[0] != self.input_size:
            input_vec_1d = np.zeros(self.input_size)

        input_vec_2d = input_vec_1d.reshape(1, -1)

        hidden_layer_input = input_vec_2d @ self.weights_input_hidden + self.bias_hidden
        hidden_layer_output = np.tanh(hidden_layer_input)

        output_layer_scores = hidden_layer_output @ self.weights_hidden_output + self.bias_output

        return input_vec_1d, hidden_layer_output.flatten(), output_layer_scores.flatten()

    def process_task(self, image_path):
        try:
            _, _, output_layer_scores = self._forward_propagate(image_path)
            return np.argmax(output_layer_scores)
        except Exception as e:
            return np.random.randint(self.output_size)

    def learn(self, image_path, true_label):
        if not (0 <= true_label < self.output_size):
            return

        input_vec_1d, hidden_output_1d, output_scores_1d = self._forward_propagate(image_path)
        if not np.any(input_vec_1d):
            return

        true_one_hot = np.zeros(self.output_size)
        true_one_hot[true_label] = 1.0
        output_probabilities = softmax(output_scores_1d)
        delta_output = output_probabilities - true_one_hot

        delta_weights_ho = np.outer(hidden_output_1d, delta_output)
        delta_bias_output = delta_output

        error_propagated_to_hidden = delta_output @ self.weights_hidden_output.T
        derivative_tanh = 1 - hidden_output_1d**2
        delta_hidden = error_propagated_to_hidden * derivative_tanh

        delta_weights_ih = np.outer(input_vec_1d, delta_hidden)
        delta_bias_hidden = delta_hidden

        self.weights_hidden_output -= self.learning_rate_learn * delta_weights_ho
        self.bias_output -= self.learning_rate_learn * delta_bias_output.reshape(1, -1)
        self.weights_input_hidden -= self.learning_rate_learn * delta_weights_ih
        self.bias_hidden -= self.learning_rate_learn * delta_bias_hidden.reshape(1, -1)

        self.memory.append((image_path, true_label))
        if len(self.memory) > 100:
            self.memory.pop(0)

    def consolidate(self):
        for image_path, true_label in list(self.memory):
            if not (0 <= true_label < self.output_size):
                continue

            input_vec_1d, hidden_output_1d, output_scores_1d = self._forward_propagate(image_path)
            if not np.any(input_vec_1d):
                continue

            true_one_hot = np.zeros(self.output_size)
            true_one_hot[true_label] = 1.0
            output_probabilities = softmax(output_scores_1d)
            delta_output = output_probabilities - true_one_hot

            delta_weights_ho = np.outer(hidden_output_1d, delta_output)
            delta_bias_output = delta_output

            error_propagated_to_hidden = delta_output @ self.weights_hidden_output.T
            derivative_tanh = 1 - hidden_output_1d**2
            delta_hidden = error_propagated_to_hidden * derivative_tanh

            delta_weights_ih = np.outer(input_vec_1d, delta_hidden)
            delta_bias_hidden = delta_hidden

            self.weights_hidden_output -= self.learning_rate_consolidate * delta_weights_ho
            self.bias_output -= self.learning_rate_consolidate * delta_bias_output.reshape(1, -1)
            self.weights_input_hidden -= self.learning_rate_consolidate * delta_weights_ih
            self.bias_hidden -= self.learning_rate_consolidate * delta_bias_hidden.reshape(1, -1)

        self.save_model()

    def save_model(self):
        model_data = {
            'weights_input_hidden': self.weights_input_hidden.tolist(),
            'bias_hidden': self.bias_hidden.tolist(),
            'weights_hidden_output': self.weights_hidden_output.tolist(),
            'bias_output': self.bias_output.tolist(),
            # Adding architectural parameters for more robust loading
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size
        }
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'w') as f:
            json.dump(model_data, f)

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'r') as f:
                    data = json.load(f)

                # Check for architectural parameters if they were saved
                # If not saved, this check will fail, and we might rely on current instance's sizes
                # or decide to re-initialize if sizes are critical and not present.
                # For now, let's assume if they are present, they should match.
                # If they are not present (older model), we proceed with shape checks against current self.sizes.
                loaded_input_size = data.get('input_size', self.input_size)
                loaded_hidden_size = data.get('hidden_size', self.hidden_size)
                loaded_output_size = data.get('output_size', self.output_size)

                if loaded_input_size != self.input_size or \
                   loaded_hidden_size != self.hidden_size or \
                   loaded_output_size != self.output_size:
                    # print("Warning: Architectural mismatch. Re-initializing model.")
                    self._initialize_default_weights_biases()
                    return

                required_keys = ['weights_input_hidden', 'bias_hidden', 'weights_hidden_output', 'bias_output']
                if all(key in data for key in required_keys):
                    w_ih = np.array(data['weights_input_hidden'])
                    b_h = np.array(data['bias_hidden'])
                    w_ho = np.array(data['weights_hidden_output'])
                    b_o = np.array(data['bias_output'])

                    if w_ih.shape == (self.input_size, self.hidden_size) and \
                       b_h.shape == (1, self.hidden_size) and \
                       w_ho.shape == (self.hidden_size, self.output_size) and \
                       b_o.shape == (1, self.output_size):
                        self.weights_input_hidden = w_ih
                        self.bias_hidden = b_h
                        self.weights_hidden_output = w_ho
                        self.bias_output = b_o
                    else:
                        raise ValueError("Shape mismatch in loaded weights/biases.")
                # No explicit 'elif weights in data' for old format, as the architectural check above
                # would ideally lead to re-initialization if sizes don't match.
                # If an old model file (single 'weights' key) is encountered, and it doesn't have
                # architectural params, the all(key in data for key in required_keys) will fail.
                else:
                    # print("Warning: Model file does not contain expected two-layer architecture keys. Re-initializing.")
                    self._initialize_default_weights_biases() # If keys for new arch are missing.

            except Exception as e:
                # print(f"Error loading/parsing model from {self.model_path}: {e}. Re-initializing all weights/biases.")
                self._initialize_default_weights_biases()
        else:
            self._initialize_default_weights_biases()


# Example
if __name__ == '__main__':
    os.makedirs("data/images", exist_ok=True)
    try:
        Image.new('L', (32,32), color='gray').save('data/images/default_test_image.png')
    except Exception as e: print(f"Could not create dummy image: {e}")

    ai = OccipitalLobeAI(model_path="data/test_occipital_model_save_load.json")

    if os.path.exists('data/images/default_test_image.png'):
        print(f"Testing save/load with default_test_image.png")
        ai.learn('data/images/default_test_image.png', true_label=0)
        ai.save_model() # Save the model with architectural params

        # Create another instance and load
        ai_loaded = OccipitalLobeAI(model_path="data/test_occipital_model_save_load.json")

        # Check if weights are loaded (not re-initialized)
        # A simple check: if one weight is same, likely all are. More robust checks in unit tests.
        np.testing.assert_array_almost_equal(ai.weights_input_hidden, ai_loaded.weights_input_hidden)
        print("Model loaded successfully and weights match.")

        if os.path.exists("data/test_occipital_model_save_load.json"): os.remove("data/test_occipital_model_save_load.json")
        if os.path.exists('data/images/default_test_image.png'): os.remove('data/images/default_test_image.png')
        print("Occipital Lobe AI save/load test finished.")
    else:
        print("Skipping Occipital Lobe AI save/load test as dummy image not found.")
