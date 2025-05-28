# limbic.py (Emotion, Motivation using Keras)
import numpy as np
import json  # For memory persistence
import os
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Input # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore


class LimbicSystemAI:
    def __init__(self, model_path="data/limbic_model.weights.h5"):  # Updated extension
        self.input_size = 10  # Processed output from TemporalLobeAI (size 10)
        self.output_size = 3  # Emotion labels (e.g., happy, urgent, sad)

        # Learning rate for the Keras optimizer
        self.learning_rate_learn = 0.001  # Standard Keras learning rate

        self.model_path = model_path
        self.memory_path = self.model_path.replace(
            ".weights.h5", "_memory.json"
        )  # Path for memory

        # Build and compile the Keras model
        self.model = self._build_model()

        # Load model weights if they exist
        self.load_model()

        # Memory stores (processed_temporal_data_list, true_emotion_label, reward)
        self.memory = []
        self.max_memory_size = 100
        self._load_memory()  # Load memory if it exists

    def _build_model(self):
        model = Sequential(
            [
                Input(shape=(self.input_size,), name="input_layer"),
                Dense(16, activation="relu", name="dense_hidden1"),
                # Dense(8, activation='relu', name='dense_hidden2'), # Optional
                Dense(
                    self.output_size, activation="softmax", name="dense_output_softmax"
                ),  # Softmax for classification
            ]
        )
        optimizer = Adam(learning_rate=self.learning_rate_learn)
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        # model.summary() # Uncomment to print summary when an instance is created
        return model

    def _ensure_input_vector_shape(self, data_list_or_array):
        """Pads or truncates data_list to match self.input_size. Returns 1D array."""
        if isinstance(data_list_or_array, np.ndarray):
            data_list = data_list_or_array.flatten().tolist()
        elif isinstance(data_list_or_array, list):
            data_list = data_list_or_array
        else:
            print(
                f"Limbic System: Warning: Unexpected input data type {type(data_list_or_array)}. Using zeros."
            )
            data_list = [0.0] * self.input_size

        current_len = len(data_list)
        if current_len == self.input_size:
            return np.array(data_list, dtype=float)
        elif current_len < self.input_size:
            return np.array(
                data_list + [0.0] * (self.input_size - current_len), dtype=float
            )
        else:
            return np.array(data_list[: self.input_size], dtype=float)

    def process_task(self, processed_temporal_data):
        # print("Limbic System: Processing task with temporal data...") # Optional: reduce print verbosity
        input_vec_1d = self._ensure_input_vector_shape(processed_temporal_data)

        default_probabilities = [1.0 / self.output_size] * self.output_size
        default_label = np.random.randint(0, self.output_size)

        if not np.any(input_vec_1d) and not np.all(
            np.array(processed_temporal_data, dtype=float) == 0
        ):
            # print( # Converted F541
            #     "Limbic System: Input vector became all zeros after preparation. Returning default emotion."
            # )
            return {"label": default_label, "probabilities": default_probabilities}

        input_batch = np.reshape(input_vec_1d, [1, self.input_size])

        try:
            predictions_batch = self.model.predict(input_batch, verbose=0)
            predicted_emotion_label = np.argmax(predictions_batch[0])
            probabilities = predictions_batch[0].tolist()
            # print(
            #     f"Limbic System: Predicted emotion label: {int(predicted_emotion_label)}, Probs: {probabilities}"
            # )
            return {"label": int(predicted_emotion_label), "probabilities": probabilities}
        except Exception as e:
            print(f"Limbic System: Error during model prediction: {e}")
            return {"label": default_label, "probabilities": default_probabilities}

    def learn(self, processed_temporal_data, true_emotion_label, reward):
        print(
            f"Limbic System: Learning with temporal data, emotion label {true_emotion_label}, reward {reward}..."
        )
        input_vec_1d = self._ensure_input_vector_shape(processed_temporal_data)

        if not np.any(input_vec_1d) and not np.all(
            np.array(processed_temporal_data, dtype=float) == 0
        ):
            print(
                "Limbic System: Input vector for learning is all zeros after preparation. Skipping learning."
            )
            return

        try:
            valid_emotion_label = int(true_emotion_label)
        except ValueError:
            print(
                f"Limbic System: Invalid true_emotion_label '{true_emotion_label}'. Must be an integer. Skipping learning."
            )
            return

        if not (0 <= valid_emotion_label < self.output_size):
            print(
                f"Limbic System: true_emotion_label {valid_emotion_label} is out of range (0-{self.output_size-1}). Skipping learning."
            )
            return

        input_batch = np.reshape(input_vec_1d, [1, self.input_size])
        target_label_array = np.array([valid_emotion_label])

        # Simple reward application: Adjust sample weight.
        # Keras `fit` expects sample_weight to be a 1D array with one weight per sample.
        # A positive reward could mean higher weight, negative lower or zero.
        # This is a basic interpretation; more complex reward schemes exist.
        sample_weight_value = 1.0  # Neutral
        if reward > 0:
            sample_weight_value = 1.5  # Emphasize positive experiences
        elif reward < 0:
            sample_weight_value = (
                0.5  # De-emphasize negative experiences (or even 0 to ignore)
            )

        sample_weights_for_fit = np.array([sample_weight_value])

        try:
            history = self.model.fit(
                input_batch,
                target_label_array,
                epochs=1,
                verbose=0,
                sample_weight=sample_weights_for_fit,  # Apply reward as sample weight
            )
            loss = history.history.get("loss", [float("nan")])[0]
            accuracy = history.history.get("accuracy", [float("nan")])[0]
            print(
                f"Limbic System: Training complete. Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Sample Weight: {sample_weight_value}"
            )

            data_to_store = (
                processed_temporal_data.tolist()
                if isinstance(processed_temporal_data, np.ndarray)
                else list(processed_temporal_data)
            )
            # Ensure standard Python types are stored for JSON serialization
            self.memory.append(
                (data_to_store, int(valid_emotion_label), float(reward))
            )
            if len(self.memory) > self.max_memory_size:
                self.memory.pop(0)
        except Exception as e:
            print(f"Limbic System: Error during model training: {e}")

    def save_model(self):
        print(f"Limbic System: Saving model weights to {self.model_path}...")
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        try:
            if not self.model_path.endswith(".weights.h5"):
                print(f"Limbic System: Warning: model_path '{self.model_path}' does not end with '.weights.h5'. Keras save_weights prefers this extension.")
            self.model.save_weights(self.model_path)
            print("Limbic System: Model weights saved.")
        except Exception as e:
            print(f"Limbic System: Error saving model weights: {e}")

    def load_model(self):
        if os.path.exists(self.model_path):
            print(f"Limbic System: Loading model weights from {self.model_path}...")
            try:
                if not hasattr(self, "model") or self.model is None:
                    self.model = self._build_model()
                self.model.load_weights(self.model_path)
                print("Limbic System: Model weights loaded successfully.")
            except Exception as e:
                print(
                    f"Limbic System: Error loading model weights: {e}. Model remains initialized."
                )
        else:
            print(
                f"Limbic System: No pre-trained weights found at {self.model_path}. Model is newly initialized."
            )

    def _save_memory(self):
        print("Limbic System: Saving memory...")
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        try:
            with open(self.memory_path, "w") as f:
                json.dump({"memory": self.memory}, f)
            print("Limbic System: Memory saved.")
        except Exception as e:
            print(f"Limbic System: Error saving memory: {e}")

    def _load_memory(self):
        if os.path.exists(self.memory_path):
            print("Limbic System: Loading memory...")
            try:
                with open(self.memory_path, "r") as f:
                    loaded_json_content = json.load(f) # This can fail if JSON is malformed
            except json.JSONDecodeError as e_json:
                print(f"Limbic System: Error decoding JSON from memory file {self.memory_path}: {e_json}. Initializing empty memory.")
                self.memory = []
                return # Exit after handling bad JSON

                raw_memory_list = loaded_json_content.get("memory", [])
                processed_memory = []
                for item in raw_memory_list:
                    if isinstance(item, (list, tuple)) and len(item) == 3:
                        try:
                            # Ensure data_list, label, and reward are correct types
                            data_list = list(item[0])
                            label = int(item[1])
                            reward_val = float(item[2])
                            processed_memory.append((data_list, label, reward_val))
                        except (TypeError, ValueError) as e_conv:
                            print(f"Limbic System: Skipping invalid memory item during load: {item}, error: {e_conv}")
                    else:
                        print(f"Limbic System: Skipping malformed memory item during load: {item}")
                self.memory = processed_memory
                if self.memory:
                    print("Limbic System: Memory loaded.")
                else:
                    print("Limbic System: Memory file was empty or contained no valid items.")
            except Exception as e:
                print(
                    f"Limbic System: Error loading memory: {e}. Initializing empty memory."
                )
                self.memory = []
        else:
            print("Limbic System: No memory file found. Initializing empty memory.")
            self.memory = []

    def consolidate(self):
        print("Limbic System: Starting consolidation...")
        if not self.memory:
            print("Limbic System: Memory is empty. Nothing to consolidate.")
        else:
            print(
                f"Limbic System: Consolidating {len(self.memory)} experiences from memory."
            )

            input_data_list = []
            target_labels_list = []
            sample_weights_list = []  # For reward-weighted consolidation

            for p_data, label, reward_val in self.memory:
                input_data_list.append(self._ensure_input_vector_shape(p_data))
                target_labels_list.append(int(label))

                # Apply reward weighting similar to learn method
                s_weight = 1.0
                if reward_val > 0:
                    s_weight = 1.5
                elif reward_val < 0:
                    s_weight = 0.5
                sample_weights_list.append(s_weight)

            if input_data_list:
                input_data_np = np.array(input_data_list)
                target_labels_np = np.array(target_labels_list)
                sample_weights_np = np.array(sample_weights_list)

                print(
                    f"Limbic System: Training consolidation batch of size {input_data_np.shape[0]}..."
                )
                try:
                    self.model.fit(
                        input_data_np,
                        target_labels_np,
                        epochs=1,  # Could be more for consolidation
                        verbose=0,
                        batch_size=16,
                        sample_weight=sample_weights_np,  # Use stored rewards as sample weights
                    )
                    print("Limbic System: Consolidation training complete.")
                except Exception as e:
                    print(f"Limbic System: Error during consolidation training: {e}")
            else:
                print(
                    "Limbic System: No valid data prepared for consolidation training."
                )

        self.save_model()
        self._save_memory()
        print("Limbic System: Consolidation complete.")

    def reset_training_data(self):
        """Clears all learned data, resets model, and deletes saved files."""
        print("Limbic System: Resetting all training data and model state...")
        # Clear memory
        self.memory = []
        print("Limbic System: Memory cleared.")

        # Delete saved files
        for path in [self.model_path, self.memory_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"Limbic System: Deleted file {path}")
                except Exception as e:
                    print(f"Limbic System: Error deleting file {path}: {e}")
            else:
                print(f"Limbic System: File {path} not found, skipping deletion.")

        # Re-initialize model
        self.model = self._build_model() # _build_model already compiles
        # self.model.compile(optimizer=Adam(learning_rate=self.learning_rate_learn), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        print("Limbic System: Reset complete. Model re-initialized.")



# Example Usage
if __name__ == "__main__":
    print("\n--- Testing LimbicSystemAI (Keras FFN) ---")
    test_model_path = "data/test_limbic_keras.weights.h5"

    # Clean up old test files
    if os.path.exists(test_model_path):
        os.remove(test_model_path)
    memory_file = test_model_path.replace(".weights.h5", "_memory.json")
    if os.path.exists(memory_file):
        os.remove(memory_file)

    limbic_ai = LimbicSystemAI(model_path=test_model_path)
    limbic_ai.model.summary()

    print("\n--- Testing Data Preparation ---")
    sample_temporal_data_short = [0.1] * (limbic_ai.input_size - 3)
    prepared_input = limbic_ai._ensure_input_vector_shape(sample_temporal_data_short)
    print(f"Prepared input for short data: shape {prepared_input.shape}")
    if prepared_input.shape != (limbic_ai.input_size,):
        raise AssertionError(f"Prepared input shape mismatch: Expected {(limbic_ai.input_size,)}, got {prepared_input.shape}")

    print("\n--- Testing process_task ---")
    temporal_input = np.random.rand(limbic_ai.input_size).tolist()
    predicted_emotion_dict = limbic_ai.process_task(temporal_input)
    predicted_emotion_label_from_dict = predicted_emotion_dict["label"]
    print(f"Predicted emotion for random input: {predicted_emotion_dict}")
    if not (0 <= predicted_emotion_label_from_dict < limbic_ai.output_size):
        raise AssertionError(f"Predicted emotion label {predicted_emotion_label_from_dict} out of expected range [0, {limbic_ai.output_size-1}]")

    print("\n--- Testing learn ---")
    true_emotion_label = 1
    reward_value_positive = 1.0
    reward_value_negative = -1.0

    print("Learning with positive reward:")
    limbic_ai.learn(temporal_input, true_emotion_label, reward_value_positive)
    print(f"Memory size after one learn call (positive): {len(limbic_ai.memory)}")

    print("Learning with negative reward:")
    limbic_ai.learn(temporal_input, true_emotion_label, reward_value_negative)
    print(f"Memory size after one learn call (negative): {len(limbic_ai.memory)}")

    print("Learning with invalid label:")
    limbic_ai.learn(temporal_input, limbic_ai.output_size + 1, reward_value_positive)

    for _ in range(5):
        t_data = np.random.rand(limbic_ai.input_size).tolist()
        e_label = np.random.randint(0, limbic_ai.output_size)
        r_val = np.random.choice([-1, 0, 1])
        limbic_ai.learn(t_data, e_label, r_val)
    print(f"Memory size after several learn calls: {len(limbic_ai.memory)}")

    print("\n--- Testing Consolidation ---")
    limbic_ai.consolidate()

    print("\n--- Testing Save/Load ---")
    limbic_ai.save_model()

    print("\nCreating new LimbicSystemAI instance for loading test...")
    limbic_ai_loaded = LimbicSystemAI(model_path=test_model_path)

    if hasattr(limbic_ai.model.get_layer("dense_output_softmax"), "kernel") and hasattr(
        limbic_ai_loaded.model.get_layer("dense_output_softmax"), "kernel"
    ):
        original_weights_output = limbic_ai.model.get_layer(
            "dense_output_softmax"
        ).kernel.numpy()
        loaded_weights_output = limbic_ai_loaded.model.get_layer(
            "dense_output_softmax"
        ).kernel.numpy()
        if np.array_equal(original_weights_output, loaded_weights_output):
            print("Model weights loaded successfully.")
        else:
            print("Model weights loading failed or mismatch.")
    else:
        print(
            "Could not access kernel weights for 'dense_output_softmax' layer to compare."
        )

    if len(limbic_ai_loaded.memory) == len(limbic_ai.memory):
        print(f"Memory loaded successfully with {len(limbic_ai_loaded.memory)} items.")
    else:
        print("Memory loading failed or mismatch.")

    # Clean up
    if os.path.exists(test_model_path):
        os.remove(test_model_path)
    if os.path.exists(memory_file):
        os.remove(memory_file)
    print("\nCleaned up test files.")

    print("\nLimbic System AI (Keras FFN) test script finished.")
