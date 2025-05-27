# temporal.py (Memory, Language, and Text-to-Visual Association)
import numpy as np
import json
import os


# Helper function for Softmax (module-level)
def softmax(x):
    if x.ndim == 1:
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)
    else:
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)


class TemporalLobeAI:
    def __init__(
        self,
        model_path="data/temporal_model.json",
        memory_path="data/temporal_memory.json",
    ):
        self.input_size = 15
        self.output_size = 10  # Dimension of text embeddings
        self.visual_output_size = 5

        self.text_hidden_size = 32
        self.visual_assoc_hidden_size = 16

        self.learning_rate_learn = 0.01
        self.learning_rate_consolidate = 0.005

        # Initialize all weights and biases to None, will be set by _initialize_default_weights_biases or load_model
        self.weights_text_input_hidden = None
        self.bias_text_hidden = None
        self.weights_text_hidden_embedding = None
        self.bias_text_embedding = None
        self.weights_embed_to_visual_hidden = None
        self.bias_visual_assoc_hidden = None
        self.weights_visual_hidden_to_label = None
        self.bias_visual_assoc_label = None

        self.memory_db = []
        self.cross_modal_memory = []
        self.max_cross_modal_memory_size = 100

        self.model_path = model_path
        self.memory_path = memory_path

        self.load_model()

    def _initialize_default_weights_biases(self):
        self.weights_text_input_hidden = (
            np.random.randn(self.input_size, self.text_hidden_size) * 0.01
        )
        self.bias_text_hidden = np.zeros((1, self.text_hidden_size))
        self.weights_text_hidden_embedding = (
            np.random.randn(self.text_hidden_size, self.output_size) * 0.01
        )
        self.bias_text_embedding = np.zeros((1, self.output_size))

        self.weights_embed_to_visual_hidden = (
            np.random.randn(self.output_size, self.visual_assoc_hidden_size) * 0.01
        )
        self.bias_visual_assoc_hidden = np.zeros((1, self.visual_assoc_hidden_size))
        self.weights_visual_hidden_to_label = (
            np.random.randn(self.visual_assoc_hidden_size, self.visual_output_size)
            * 0.01
        )
        self.bias_visual_assoc_label = np.zeros((1, self.visual_output_size))

    def _to_numerical_vector(self, data, size, context="input"):
        if isinstance(data, str):
            hash_val = hash(data)
            np.random.seed(hash_val % (2**32 - 1))
            vec = np.random.rand(size)
            np.random.seed(None)
            return vec
        try:
            vec = np.array(data, dtype=float).flatten()
        except ValueError:
            return np.zeros(size)
        if vec.shape[0] == size:
            return vec
        elif vec.shape[0] < size:
            return np.concatenate((vec, np.zeros(size - vec.shape[0])))
        else:
            return vec[:size]

    def _forward_prop_text(self, input_text_1d):
        if input_text_1d.shape[0] != self.input_size:
            input_text_1d = self._to_numerical_vector(input_text_1d, self.input_size)
        input_text_2d = input_text_1d.reshape(1, -1)
        text_hidden_input = (
            input_text_2d @ self.weights_text_input_hidden + self.bias_text_hidden
        )
        text_hidden_output = np.tanh(text_hidden_input)
        text_embedding_scores = (
            text_hidden_output @ self.weights_text_hidden_embedding
            + self.bias_text_embedding
        )
        return (
            input_text_1d,
            text_hidden_output.flatten(),
            text_embedding_scores.flatten(),
        )

    def _forward_prop_visual_assoc(self, text_embedding_1d):
        if text_embedding_1d.shape[0] != self.output_size:
            text_embedding_1d = np.zeros(self.output_size)
        text_embedding_2d = text_embedding_1d.reshape(1, -1)
        visual_hidden_input = (
            text_embedding_2d @ self.weights_embed_to_visual_hidden
            + self.bias_visual_assoc_hidden
        )
        visual_hidden_output = np.tanh(visual_hidden_input)
        visual_label_scores = (
            visual_hidden_output @ self.weights_visual_hidden_to_label
            + self.bias_visual_assoc_label
        )
        return (
            text_embedding_1d,
            visual_hidden_output.flatten(),
            visual_label_scores.flatten(),
        )

    def process_task(self, input_data_item, predict_visual=False):
        actual_input_data = (
            input_data_item[-1]
            if isinstance(input_data_item, list) and input_data_item
            else input_data_item
        )
        input_text_1d = self._to_numerical_vector(actual_input_data, self.input_size)
        _, _, text_embedding = self._forward_prop_text(input_text_1d)
        if predict_visual:
            _, _, visual_label_scores = self._forward_prop_visual_assoc(text_embedding)
            return (text_embedding.tolist(), np.argmax(visual_label_scores))
        else:
            return text_embedding.tolist()

    def learn(self, sequence, visual_label_as_context=None):
        # --- A. Text Processing Learning ---
        if (
            isinstance(sequence, list)
            and sequence
            and all(isinstance(item, tuple) and len(item) == 2 for item in sequence)
        ):
            self.memory_db.append(sequence)
            if len(self.memory_db) > 100:
                self.memory_db.pop(0)

            for input_data, target_output in sequence:
                input_text_1d = self._to_numerical_vector(input_data, self.input_size)
                target_text_embedding_1d = self._to_numerical_vector(
                    target_output, self.output_size
                )

                actual_input_text_1d, text_hidden_out_1d, text_embedding_scores_1d = (
                    self._forward_prop_text(input_text_1d)
                )

                # Error at text embedding layer
                delta_text_embedding = (
                    text_embedding_scores_1d - target_text_embedding_1d
                )

                # Gradients for text_hidden-to-embedding
                delta_weights_he = np.outer(text_hidden_out_1d, delta_text_embedding)
                delta_bias_te = delta_text_embedding

                # Error at text hidden layer
                error_prop_to_text_hidden = (
                    delta_text_embedding @ self.weights_text_hidden_embedding.T
                )
                deriv_tanh_text = 1 - text_hidden_out_1d**2
                delta_text_hidden = error_prop_to_text_hidden * deriv_tanh_text

                # Gradients for text_input-to-hidden
                delta_weights_ih = np.outer(actual_input_text_1d, delta_text_hidden)
                delta_bias_th = delta_text_hidden

                # Update Text Processing Weights/Biases
                self.weights_text_hidden_embedding -= (
                    self.learning_rate_learn * delta_weights_he
                )
                self.bias_text_embedding -= (
                    self.learning_rate_learn * delta_bias_te.reshape(1, -1)
                )
                self.weights_text_input_hidden -= (
                    self.learning_rate_learn * delta_weights_ih
                )
                self.bias_text_hidden -= (
                    self.learning_rate_learn * delta_bias_th.reshape(1, -1)
                )

        # --- B. Visual Association Learning ---
        if visual_label_as_context is not None and (
            0 <= visual_label_as_context < self.visual_output_size
        ):
            if sequence and sequence[0] and isinstance(sequence[0][0], str):
                text_input_str_for_assoc = sequence[0][0]
                self.cross_modal_memory.append(
                    (text_input_str_for_assoc, visual_label_as_context)
                )
                if len(self.cross_modal_memory) > self.max_cross_modal_memory_size:
                    self.cross_modal_memory.pop(0)

                assoc_input_text_1d = self._to_numerical_vector(
                    text_input_str_for_assoc, self.input_size
                )

                # Forward prop text path to get current embedding
                _, text_hidden_out_for_assoc_1d, current_text_embedding_1d = (
                    self._forward_prop_text(assoc_input_text_1d)
                )

                # Forward prop visual association path
                _, visual_hidden_out_1d, visual_label_scores_1d = (
                    self._forward_prop_visual_assoc(current_text_embedding_1d)
                )

                true_visual_one_hot = np.zeros(self.visual_output_size)
                true_visual_one_hot[visual_label_as_context] = 1.0
                visual_output_probabilities = softmax(visual_label_scores_1d)

                delta_visual_label_output = (
                    visual_output_probabilities - true_visual_one_hot
                )

                delta_weights_vhl = np.outer(
                    visual_hidden_out_1d, delta_visual_label_output
                )
                delta_bias_vl = delta_visual_label_output

                error_prop_to_visual_hidden = (
                    delta_visual_label_output @ self.weights_visual_hidden_to_label.T
                )
                deriv_tanh_visual = 1 - visual_hidden_out_1d**2
                delta_visual_assoc_hidden = (
                    error_prop_to_visual_hidden * deriv_tanh_visual
                )

                delta_weights_evh = np.outer(
                    current_text_embedding_1d, delta_visual_assoc_hidden
                )
                delta_bias_vh = delta_visual_assoc_hidden

                self.weights_visual_hidden_to_label -= (
                    self.learning_rate_learn * delta_weights_vhl
                )
                self.bias_visual_assoc_label -= (
                    self.learning_rate_learn * delta_bias_vl.reshape(1, -1)
                )
                self.weights_embed_to_visual_hidden -= (
                    self.learning_rate_learn * delta_weights_evh
                )
                self.bias_visual_assoc_hidden -= (
                    self.learning_rate_learn * delta_bias_vh.reshape(1, -1)
                )

                # --- C. Error Backpropagation from Visual Path to Text Embedding Layer ---
                error_from_visual_to_embedding = (
                    delta_visual_assoc_hidden @ self.weights_embed_to_visual_hidden.T
                )

                # This error (delta_text_embedding_from_visual) applies to current_text_embedding_1d
                delta_text_embedding_from_visual = (
                    error_from_visual_to_embedding  # Shape (output_size,)
                )

                # Update weights_text_hidden_embedding
                # text_hidden_out_for_assoc_1d is the output of the text hidden layer for this specific input
                delta_weights_he_from_visual = np.outer(
                    text_hidden_out_for_assoc_1d, delta_text_embedding_from_visual
                )
                delta_bias_te_from_visual = delta_text_embedding_from_visual
                self.weights_text_hidden_embedding -= (
                    self.learning_rate_learn * delta_weights_he_from_visual
                )
                self.bias_text_embedding -= (
                    self.learning_rate_learn * delta_bias_te_from_visual.reshape(1, -1)
                )

                # Propagate this error further back to text_input_hidden weights
                error_prop_to_text_hidden_from_visual = (
                    delta_text_embedding_from_visual
                    @ self.weights_text_hidden_embedding.T
                )
                deriv_tanh_text_for_assoc = 1 - text_hidden_out_for_assoc_1d**2
                delta_text_hidden_from_visual = (
                    error_prop_to_text_hidden_from_visual * deriv_tanh_text_for_assoc
                )

                delta_weights_ih_from_visual = np.outer(
                    assoc_input_text_1d, delta_text_hidden_from_visual
                )
                delta_bias_th_from_visual = delta_text_hidden_from_visual

                self.weights_text_input_hidden -= (
                    self.learning_rate_learn * delta_weights_ih_from_visual
                )
                self.bias_text_hidden -= (
                    self.learning_rate_learn * delta_bias_th_from_visual.reshape(1, -1)
                )

    def consolidate(self):
        lr = self.learning_rate_consolidate
        # Consolidate text processing (memory_db)
        for sequence_item in list(self.memory_db):
            if not (
                isinstance(sequence_item, list)
                and sequence_item
                and all(isinstance(p, tuple) and len(p) == 2 for p in sequence_item)
            ):
                continue
            for input_data, target_output in sequence_item:
                input_text_1d = self._to_numerical_vector(input_data, self.input_size)
                target_text_embedding_1d = self._to_numerical_vector(
                    target_output, self.output_size
                )
                actual_input_text_1d, text_hidden_out_1d, text_embedding_scores_1d = (
                    self._forward_prop_text(input_text_1d)
                )
                delta_text_embedding = (
                    text_embedding_scores_1d - target_text_embedding_1d
                )
                delta_weights_he = np.outer(text_hidden_out_1d, delta_text_embedding)
                delta_bias_te = delta_text_embedding
                error_prop_to_text_hidden = (
                    delta_text_embedding @ self.weights_text_hidden_embedding.T
                )
                deriv_tanh_text = 1 - text_hidden_out_1d**2
                delta_text_hidden = error_prop_to_text_hidden * deriv_tanh_text
                delta_weights_ih = np.outer(actual_input_text_1d, delta_text_hidden)
                delta_bias_th = delta_text_hidden
                self.weights_text_hidden_embedding -= lr * delta_weights_he
                self.bias_text_embedding -= lr * delta_bias_te.reshape(1, -1)
                self.weights_text_input_hidden -= lr * delta_weights_ih
                self.bias_text_hidden -= lr * delta_bias_th.reshape(1, -1)

        # Consolidate visual association (cross_modal_memory)
        for text_input_str, true_visual_label in list(self.cross_modal_memory):
            if not (0 <= true_visual_label < self.visual_output_size):
                continue

            assoc_input_text_1d = self._to_numerical_vector(
                text_input_str, self.input_size
            )
            _, text_hidden_out_for_assoc_1d, current_text_embedding_1d = (
                self._forward_prop_text(assoc_input_text_1d)
            )
            _, visual_hidden_out_1d, visual_label_scores_1d = (
                self._forward_prop_visual_assoc(current_text_embedding_1d)
            )

            true_visual_one_hot = np.zeros(self.visual_output_size)
            true_visual_one_hot[true_visual_label] = 1.0
            visual_output_probabilities = softmax(visual_label_scores_1d)
            delta_visual_label_output = (
                visual_output_probabilities - true_visual_one_hot
            )
            delta_weights_vhl = np.outer(
                visual_hidden_out_1d, delta_visual_label_output
            )
            delta_bias_vl = delta_visual_label_output
            error_prop_to_visual_hidden = (
                delta_visual_label_output @ self.weights_visual_hidden_to_label.T
            )
            deriv_tanh_visual = 1 - visual_hidden_out_1d**2
            delta_visual_assoc_hidden = error_prop_to_visual_hidden * deriv_tanh_visual
            delta_weights_evh = np.outer(
                current_text_embedding_1d, delta_visual_assoc_hidden
            )
            delta_bias_vh = delta_visual_assoc_hidden
            self.weights_visual_hidden_to_label -= lr * delta_weights_vhl
            self.bias_visual_assoc_label -= lr * delta_bias_vl.reshape(1, -1)
            self.weights_embed_to_visual_hidden -= lr * delta_weights_evh
            self.bias_visual_assoc_hidden -= lr * delta_bias_vh.reshape(1, -1)

            error_from_visual_to_embedding = (
                delta_visual_assoc_hidden @ self.weights_embed_to_visual_hidden.T
            )
            delta_text_embedding_from_visual = error_from_visual_to_embedding
            delta_weights_he_from_visual = np.outer(
                text_hidden_out_for_assoc_1d, delta_text_embedding_from_visual
            )
            delta_bias_te_from_visual = delta_text_embedding_from_visual
            self.weights_text_hidden_embedding -= lr * delta_weights_he_from_visual
            self.bias_text_embedding -= lr * delta_bias_te_from_visual.reshape(1, -1)
            error_prop_to_text_hidden_from_visual = (
                delta_text_embedding_from_visual @ self.weights_text_hidden_embedding.T
            )
            deriv_tanh_text_for_assoc = 1 - text_hidden_out_for_assoc_1d**2
            delta_text_hidden_from_visual = (
                error_prop_to_text_hidden_from_visual * deriv_tanh_text_for_assoc
            )
            delta_weights_ih_from_visual = np.outer(
                assoc_input_text_1d, delta_text_hidden_from_visual
            )
            delta_bias_th_from_visual = delta_text_hidden_from_visual
            self.weights_text_input_hidden -= lr * delta_weights_ih_from_visual
            self.bias_text_hidden -= lr * delta_bias_th_from_visual.reshape(1, -1)

        self.save_model()
        self.save_memory()

    def save_model(self):
        model_data_to_save = {
            "weights_text_input_hidden": self.weights_text_input_hidden.tolist(),
            "bias_text_hidden": self.bias_text_hidden.tolist(),
            "weights_text_hidden_embedding": self.weights_text_hidden_embedding.tolist(),
            "bias_text_embedding": self.bias_text_embedding.tolist(),
            "weights_embed_to_visual_hidden": self.weights_embed_to_visual_hidden.tolist(),
            "bias_visual_assoc_hidden": self.bias_visual_assoc_hidden.tolist(),
            "weights_visual_hidden_to_label": self.weights_visual_hidden_to_label.tolist(),
            "bias_visual_assoc_label": self.bias_visual_assoc_label.tolist(),
            "input_size": self.input_size,
            "text_hidden_size": self.text_hidden_size,
            "output_size": self.output_size,
            "visual_assoc_hidden_size": self.visual_assoc_hidden_size,
            "visual_output_size": self.visual_output_size,
        }
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "w") as f:
            json.dump(model_data_to_save, f)

    def save_memory(self):
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        with open(self.memory_path, "w") as f:
            json.dump(
                {
                    "memory_db": self.memory_db,
                    "cross_modal_memory": self.cross_modal_memory,
                },
                f,
            )

    def load_model(self):
        self._initialize_default_weights_biases()
        self.memory_db = []
        self.cross_modal_memory = []

        model_loaded_successfully = False # Initialize the flag
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "r") as f:
                    data = json.load(f)

                # Validate architectural parameters
                arch_params_ok = all([
                    data.get("input_size") == self.input_size,
                    data.get("text_hidden_size") == self.text_hidden_size,
                    data.get("output_size") == self.output_size,
                    data.get("visual_assoc_hidden_size") == self.visual_assoc_hidden_size,
                    data.get("visual_output_size") == self.visual_output_size
                ])
                if not arch_params_ok:
                    print("TemporalLobeAI: Model architecture mismatch. Using default weights.")
                    # self._initialize_default_weights_biases() # Already called
                    # Memory will be loaded separately

                else:
                    # Validate presence of all weight keys
                    required_keys = [
                        "weights_text_input_hidden", "bias_text_hidden",
                        "weights_text_hidden_embedding", "bias_text_embedding",
                        "weights_embed_to_visual_hidden", "bias_visual_assoc_hidden",
                        "weights_visual_hidden_to_label", "bias_visual_assoc_label"
                    ]
                    if not all(key in data for key in required_keys):
                        print("TemporalLobeAI: Model file missing weight keys. Using default weights.")
                        # self._initialize_default_weights_biases() # Already called
                    else:
                        # Attempt to load weights and check shapes
                        try:
                            w_t_ih = np.array(data["weights_text_input_hidden"])
                            b_t_h = np.array(data["bias_text_hidden"])
                            # ... (load all other weights and biases) ...
                            w_v_hl = np.array(data["weights_visual_hidden_to_label"])
                            b_v_al = np.array(data["bias_visual_assoc_label"])

                            # Example shape check (do for all)
                            if w_t_ih.shape != (self.input_size, self.text_hidden_size):
                                raise ValueError("Shape mismatch for weights_text_input_hidden")
                            # ... (check all other shapes) ...

                            # If all checks pass, assign weights
                            self.weights_text_input_hidden = w_t_ih
                            self.bias_text_hidden = b_t_h
                            self.weights_text_hidden_embedding = np.array(data["weights_text_hidden_embedding"])
                            self.bias_text_embedding = np.array(data["bias_text_embedding"])
                            self.weights_embed_to_visual_hidden = np.array(data["weights_embed_to_visual_hidden"])
                            self.bias_visual_assoc_hidden = np.array(data["bias_visual_assoc_hidden"])
                            self.weights_visual_hidden_to_label = w_v_hl
                            self.bias_visual_assoc_label = b_v_al
                            model_loaded_successfully = True
                            # Success message will be printed based on the flag later
                        except (ValueError, TypeError) as e_shape:
                            print(f"TemporalLobeAI: Error validating model weights/shapes: {e_shape}. Using default weights.")
                            self._initialize_default_weights_biases() # Re-initialize if loading specific weights failed

            except Exception as e_load_model:
                print(f"TemporalLobeAI: Error loading model file {self.model_path}: {e_load_model}. Using default weights.")
                # self._initialize_default_weights_biases() # Already called
        else:
            print(f"TemporalLobeAI: No model file found at {self.model_path}. Using default weights.")
            # self._initialize_default_weights_biases() # Already called

        # Report final model loading status based on the flag
        if model_loaded_successfully:
            print("TemporalLobeAI: Model weights were successfully loaded from file.")
        else:
            print("TemporalLobeAI: Model is using default or re-initialized weights due to missing file or loading issues.")

        # --- Load Memory ---
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, "r") as f_mem:
                    loaded_memory_data = json.load(f_mem)
                if isinstance(loaded_memory_data, dict):
                    self.memory_db = loaded_memory_data.get("memory_db", [])
                    self.cross_modal_memory = loaded_memory_data.get("cross_modal_memory", [])
                elif isinstance(loaded_memory_data, list): # Backward compatibility
                    self.memory_db = loaded_memory_data
                    self.cross_modal_memory = []

                # Apply filters for data integrity
                self.memory_db = [
                    item for item in self.memory_db if isinstance(item, list) and
                    all(isinstance(p, tuple) and len(p) == 2 for p in item)
                ] if isinstance(self.memory_db, list) else []
                self.cross_modal_memory = [
                    item for item in self.cross_modal_memory if isinstance(item, tuple) and len(item) == 2
                ] if isinstance(self.cross_modal_memory, list) else []
                print("TemporalLobeAI: Memory loaded.")
            except Exception as e_load_mem:
                print(f"TemporalLobeAI: Error loading memory file {self.memory_path}: {e_load_mem}. Initializing empty memory.")
                self.memory_db = []
                self.cross_modal_memory = []
        else:
            print(f"TemporalLobeAI: No memory file found at {self.memory_path}. Initializing empty memory.")
            self.memory_db = []
            self.cross_modal_memory = []

# Example Usage
if __name__ == "__main__":
    ai = TemporalLobeAI(
        model_path="data/test_temporal_model_full_bp.json",
        memory_path="data/test_temporal_memory_full_bp.json",
    )
    print("TemporalLobeAI initialized for full backpropagation.")

    text_seq = [
        ("a red block", "a red block")
    ]  # Target is itself for auto-encoding like embedding
    visual_label = 0  # e.g. "red things"

    print(
        "Initial weights (sample from text_input_hidden):",
        ai.weights_text_input_hidden[0, 0],
    )
    print(
        "Initial weights (sample from embed_to_visual_label):",
        ai.weights_visual_hidden_to_label[0, 0],
    )

    ai.learn(text_seq, visual_label_as_context=visual_label)
    print("After learn:")
    print("  weights_text_input_hidden (sample):", ai.weights_text_input_hidden[0, 0])
    print(
        "  weights_visual_hidden_to_label (sample):",
        ai.weights_visual_hidden_to_label[0, 0],
    )

    ai.consolidate()
    print("After consolidate:")
    print("  weights_text_input_hidden (sample):", ai.weights_text_input_hidden[0, 0])
    print(
        "  weights_visual_hidden_to_label (sample):",
        ai.weights_visual_hidden_to_label[0, 0],
    )

    if os.path.exists("data/test_temporal_model_full_bp.json"):
        os.remove("data/test_temporal_model_full_bp.json")
    if os.path.exists("data/test_temporal_memory_full_bp.json"):
        os.remove("data/test_temporal_memory_full_bp.json")
    print("Temporal Lobe AI full backpropagation test finished.")
