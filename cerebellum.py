# cerebellum.py (Motor Control, Coordination using Keras)
import numpy as np
import json  # For memory persistence
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers.legacy import Adam as LegacyAdam  # For compatibility


class CerebellumAI:
    def __init__(
        self, model_path="data/cerebellum_model.weights.h5"
    ):  # Updated extension
        self.input_size = 10  # Sensor feedback
        self.output_size = 3  # Motor commands (e.g., scaled between -1 and 1)

        # Learning rate for the Keras optimizer
        self.learning_rate_learn = 0.001  # Adam default is often 0.001

        self.model_path = model_path
        self.memory_path = self.model_path.replace(
            ".weights.h5", "_memory.json"
        )  # Path for memory

        # Build and compile the Keras model
        self.model = self._build_model()

        # Load model weights if they exist
        self.load_model()

        # Memory will store (sensor_data_list, true_command_list) tuples.
        self.memory = []
        self.max_memory_size = 100
        self._load_memory()  # Load memory if it exists

    def _build_model(self):
        model = Sequential(
            [
                Input(shape=(self.input_size,), name="input_layer"),
                Dense(20, activation="relu", name="dense_hidden1"),  # Or 'tanh'
                # Dense(15, activation='relu', name='dense_hidden2'), # Optional second hidden layer
                Dense(
                    self.output_size, activation="tanh", name="dense_output"
                ),  # tanh for motor commands in [-1, 1]
            ]
        )
        optimizer = LegacyAdam(learning_rate=self.learning_rate_learn)
        model.compile(
            optimizer=optimizer, loss="mse"
        )  # Mean Squared Error for regression
        # model.summary() # Uncomment to print summary when an instance is created
        return model

    def _prepare_input_vector(self, sensor_data):
        """Prepares a 1D numpy array from sensor_data, ensuring correct size."""
        if isinstance(sensor_data, list):
            input_vec_list = sensor_data
        elif isinstance(sensor_data, np.ndarray):
            input_vec_list = sensor_data.flatten().tolist()
        else:
            print(
                f"Cerebellum Lobe: Warning: Unexpected sensor_data type {type(sensor_data)}. Using zeros."
            )
            input_vec_list = [0.0] * self.input_size

        current_len = len(input_vec_list)
        if current_len == self.input_size:
            return np.array(input_vec_list, dtype=float)
        elif current_len < self.input_size:
            return np.array(
                input_vec_list + [0.0] * (self.input_size - current_len), dtype=float
            )
        else:
            return np.array(input_vec_list[: self.input_size], dtype=float)

    def _prepare_target_command_vector(self, true_command_list):
        """Prepares a 1D numpy array from true_command_list, ensuring correct size and range [-1, 1]."""
        if isinstance(true_command_list, list):
            target_list = true_command_list
        elif isinstance(true_command_list, np.ndarray):
            target_list = true_command_list.flatten().tolist()
        else:
            print(
                f"Cerebellum Lobe: Warning: Unexpected true_command_list type {type(true_command_list)}. Using zeros."
            )
            target_list = [0.0] * self.output_size

        current_len = len(target_list)
        if current_len < self.output_size:
            target_list.extend([0.0] * (self.output_size - current_len))
        elif current_len > self.output_size:
            target_list = target_list[: self.output_size]

        # Ensure values are within [-1, 1] for tanh compatibility
        # This is a simple clip, more sophisticated scaling might be needed depending on source of true_command_list
        prepared_array = np.array(target_list, dtype=float)
        prepared_array = np.clip(prepared_array, -1.0, 1.0)
        return prepared_array

    def process_task(self, sensor_data):
        print(f"Cerebellum Lobe: Processing task with sensor data...")
        input_vec_1d = self._prepare_input_vector(sensor_data)

        if not np.any(input_vec_1d) and not np.all(
            np.array(sensor_data, dtype=float) == 0
        ):
            print(
                "Cerebellum Lobe: Input vector became all zeros after preparation. Returning default command."
            )
            return [0.0] * self.output_size

        input_batch = np.reshape(input_vec_1d, [1, self.input_size])

        try:
            predicted_commands_batch = self.model.predict(input_batch, verbose=0)
            predicted_commands_list = predicted_commands_batch[0].tolist()
            print(
                f"Cerebellum Lobe: Predicted motor commands: {predicted_commands_list}"
            )
            return predicted_commands_list
        except Exception as e:
            print(f"Cerebellum Lobe: Error during model prediction: {e}")
            return [0.0] * self.output_size

    def learn(self, sensor_data, target_motor_commands):
        print(
            f"Cerebellum Lobe: Learning with sensor data and target motor commands..."
        )
        input_vec_1d = self._prepare_input_vector(sensor_data)
        target_commands_1d = self._prepare_target_command_vector(target_motor_commands)

        if not np.any(input_vec_1d) and not np.all(
            np.array(sensor_data, dtype=float) == 0
        ):
            print(
                "Cerebellum Lobe: Input vector for learning is all zeros after preparation. Skipping learning."
            )
            return

        input_batch = np.reshape(input_vec_1d, [1, self.input_size])
        target_batch = np.reshape(target_commands_1d, [1, self.output_size])

        try:
            history = self.model.fit(input_batch, target_batch, epochs=1, verbose=0)
            loss = history.history.get("loss", [float("nan")])[0]
            print(f"Cerebellum Lobe: Training complete. Loss: {loss:.4f}")

            s_data_to_store = (
                sensor_data.tolist()
                if isinstance(sensor_data, np.ndarray)
                else list(sensor_data)
            )
            t_cmds_to_store = (
                target_motor_commands.tolist()
                if isinstance(target_motor_commands, np.ndarray)
                else list(target_motor_commands)
            )
            self.memory.append((s_data_to_store, t_cmds_to_store))
            if len(self.memory) > self.max_memory_size:
                self.memory.pop(0)
        except Exception as e:
            print(f"Cerebellum Lobe: Error during model training: {e}")

    def save_model(self):
        print(f"Cerebellum Lobe: Saving model weights to {self.model_path}...")
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        try:
            self.model.save_weights(self.model_path)
            print("Cerebellum Lobe: Model weights saved.")
        except Exception as e:
            print(f"Cerebellum Lobe: Error saving model weights: {e}")

    def load_model(self):
        if os.path.exists(self.model_path):
            print(f"Cerebellum Lobe: Loading model weights from {self.model_path}...")
            try:
                if not hasattr(self, "model") or self.model is None:
                    self.model = self._build_model()
                self.model.load_weights(self.model_path)
                print("Cerebellum Lobe: Model weights loaded successfully.")
            except Exception as e:
                print(
                    f"Cerebellum Lobe: Error loading model weights: {e}. Model remains initialized."
                )
        else:
            print(
                f"Cerebellum Lobe: No pre-trained weights found at {self.model_path}. Model is newly initialized."
            )

    def _save_memory(self):
        print("Cerebellum Lobe: Saving memory...")
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        try:
            with open(self.memory_path, "w") as f:
                json.dump({"memory": self.memory}, f)
            print("Cerebellum Lobe: Memory saved.")
        except Exception as e:
            print(f"Cerebellum Lobe: Error saving memory: {e}")

    def _load_memory(self):
        if os.path.exists(self.memory_path):
            print("Cerebellum Lobe: Loading memory...")
            try:
                with open(self.memory_path, "r") as f:
                    loaded_data = json.load(f)
                    self.memory = loaded_data.get("memory", [])
                self.memory = [
                    (list(item[0]), list(item[1]))
                    for item in self.memory
                    if isinstance(item, (list, tuple)) and len(item) == 2
                ]
                print("Cerebellum Lobe: Memory loaded.")
            except Exception as e:
                print(
                    f"Cerebellum Lobe: Error loading memory: {e}. Initializing empty memory."
                )
                self.memory = []
        else:
            print("Cerebellum Lobe: No memory file found. Initializing empty memory.")
            self.memory = []

    def consolidate(self):
        print("Cerebellum Lobe: Starting consolidation...")
        if not self.memory:
            print("Cerebellum Lobe: Memory is empty. Nothing to consolidate.")
        else:
            print(
                f"Cerebellum Lobe: Consolidating {len(self.memory)} experiences from memory."
            )

            sensor_data_batch_list = []
            target_commands_batch_list = []

            for s_data, t_cmds in self.memory:
                sensor_data_batch_list.append(self._prepare_input_vector(s_data))
                target_commands_batch_list.append(
                    self._prepare_target_command_vector(t_cmds)
                )

            if sensor_data_batch_list:
                input_data_np = np.array(sensor_data_batch_list)
                target_data_np = np.array(target_commands_batch_list)
                print(
                    f"Cerebellum Lobe: Training consolidation batch of size {input_data_np.shape[0]}..."
                )
                try:
                    # Using learning_rate_learn for consolidation for simplicity
                    self.model.fit(
                        input_data_np,
                        target_data_np,
                        epochs=1,
                        verbose=0,
                        batch_size=16,
                    )
                    print("Cerebellum Lobe: Consolidation training complete.")
                except Exception as e:
                    print(f"Cerebellum Lobe: Error during consolidation training: {e}")
            else:
                print(
                    "Cerebellum Lobe: No valid data prepared for consolidation training."
                )

        self.save_model()
        self._save_memory()
        print("Cerebellum Lobe: Consolidation complete.")


# Example Usage
if __name__ == "__main__":
    print("\n--- Testing CerebellumAI (Keras FFN) ---")
    test_model_path = "data/test_cerebellum_keras.weights.h5"

    # Clean up old test files
    if os.path.exists(test_model_path):
        os.remove(test_model_path)
    memory_file = test_model_path.replace(".weights.h5", "_memory.json")
    if os.path.exists(memory_file):
        os.remove(memory_file)

    cerebellum_ai = CerebellumAI(model_path=test_model_path)
    cerebellum_ai.model.summary()

    print("\n--- Testing Data Preparation ---")
    sample_sensor_data_short = [0.1] * (
        cerebellum_ai.input_size - 5
    )  # Shorter than input_size
    prepared_input = cerebellum_ai._prepare_input_vector(sample_sensor_data_short)
    print(f"Prepared input for short data: shape {prepared_input.shape}")
    assert prepared_input.shape == (cerebellum_ai.input_size,)

    sample_commands_long = [0.5, -0.5, 0.0, 0.9, -0.9]  # Longer than output_size
    prepared_target = cerebellum_ai._prepare_target_command_vector(sample_commands_long)
    print(
        f"Prepared target for long data: {prepared_target}, shape {prepared_target.shape}"
    )
    assert prepared_target.shape == (cerebellum_ai.output_size,)
    assert np.all(prepared_target >= -1) and np.all(prepared_target <= 1)

    print("\n--- Testing process_task ---")
    sensor_input = np.random.rand(cerebellum_ai.input_size).tolist()
    predicted_commands = cerebellum_ai.process_task(sensor_input)
    print(f"Predicted commands for random input: {predicted_commands}")
    assert len(predicted_commands) == cerebellum_ai.output_size
    assert all(-1 <= cmd <= 1 for cmd in predicted_commands)  # Check tanh output range

    print("\n--- Testing learn ---")
    target_motor_cmds = (
        np.random.rand(cerebellum_ai.output_size) * 2 - 1
    ).tolist()  # Random in [-1, 1]
    cerebellum_ai.learn(sensor_input, target_motor_cmds)
    print(f"Memory size after one learn call: {len(cerebellum_ai.memory)}")

    for i in range(5):
        s_data = np.random.rand(cerebellum_ai.input_size).tolist()
        t_cmds = (np.random.rand(cerebellum_ai.output_size) * 2 - 1).tolist()
        cerebellum_ai.learn(s_data, t_cmds)
    print(f"Memory size after several learn calls: {len(cerebellum_ai.memory)}")

    print("\n--- Testing Consolidation ---")
    cerebellum_ai.consolidate()

    print("\n--- Testing Save/Load ---")
    cerebellum_ai.save_model()

    print("\nCreating new CerebellumAI instance for loading test...")
    cerebellum_ai_loaded = CerebellumAI(model_path=test_model_path)

    if hasattr(cerebellum_ai.model.get_layer("dense_output"), "kernel") and hasattr(
        cerebellum_ai_loaded.model.get_layer("dense_output"), "kernel"
    ):
        original_weights_output = cerebellum_ai.model.get_layer(
            "dense_output"
        ).kernel.numpy()
        loaded_weights_output = cerebellum_ai_loaded.model.get_layer(
            "dense_output"
        ).kernel.numpy()
        if np.array_equal(original_weights_output, loaded_weights_output):
            print(
                "Model weights loaded successfully (output layer kernel weights match)."
            )
        else:
            print("Model weights loading failed or mismatch.")
    else:
        print("Could not access kernel weights for 'dense_output' layer to compare.")

    if len(cerebellum_ai_loaded.memory) == len(cerebellum_ai.memory):
        print(
            f"Memory loaded successfully with {len(cerebellum_ai_loaded.memory)} items."
        )
    else:
        print("Memory loading failed or mismatch.")

    # Clean up
    if os.path.exists(test_model_path):
        os.remove(test_model_path)
    if os.path.exists(memory_file):
        os.remove(memory_file)
    print("\nCleaned up test files.")

    print("\nCerebellum Lobe AI (Keras FFN) test script finished.")
