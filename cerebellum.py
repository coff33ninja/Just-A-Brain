# cerebellum.py (Motor Control, Coordination)
import numpy as np
import os
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Input # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore


class CerebellumAI:
    def __init__(self, model_path="data/cerebellum_model.weights.h5"):
        self.input_size = 10  # Sensor feedback
        self.hidden_size = 20  # Size of the new hidden layer
        self.output_size = 3  # Motor commands (e.g., scaled between -1 and 1)

        self.learning_rate_learn = 0.01
        # self.learning_rate_consolidate = 0.005 # Less directly applicable with Keras single optimizer

        self.memory = []  # Stores (sensor_data_list, true_command_list)
        self.max_memory_size = 100 # Max memory size
        self.model_path = model_path

        self.model = self._build_model()
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate_learn), loss='mse')

        self.load_model()

    def _build_model(self):
        model = Sequential(
            [
                Input(shape=(self.input_size,), name="input_layer"),
                Dense(self.hidden_size, activation='tanh', name='hidden_layer'),
                Dense(self.output_size, activation='tanh', name='output_layer'),
            ]
        )
        return model

    def _prepare_input_vector(self, sensor_data):
        """Prepares a 1D numpy array from sensor_data, ensuring correct size and dtype."""
        if isinstance(sensor_data, list):
            input_vec_list = sensor_data
        elif isinstance(sensor_data, np.ndarray):
            input_vec_list = sensor_data.flatten().tolist()
        else:
            input_vec_list = [0.0] * self.input_size

        if len(input_vec_list) < self.input_size:
            input_vec_list.extend([0.0] * (self.input_size - len(input_vec_list)))
        elif len(input_vec_list) > self.input_size:
            input_vec_list = input_vec_list[: self.input_size]
        return np.array(input_vec_list, dtype=np.float32)

    def _prepare_target_command_vector(self, true_command_list_or_array):
        """Prepares a 1D numpy array from true_command_list, ensuring correct size and dtype."""
        if isinstance(true_command_list_or_array, list):
            target_list = true_command_list_or_array
        elif isinstance(true_command_list_or_array, np.ndarray):
            target_list = true_command_list_or_array.flatten().tolist()
        else:
            target_list = [0.0] * self.output_size

        if len(target_list) < self.output_size:
            target_list.extend([0.0] * (self.output_size - len(target_list)))
        elif len(target_list) > self.output_size:
            target_list = target_list[: self.output_size]
        return np.array(target_list, dtype=np.float32)

    def process_task(self, sensor_data):
        try:
            input_vec_1d = self._prepare_input_vector(sensor_data)
            input_batch = np.reshape(input_vec_1d, [1, self.input_size])
            predicted_commands_batch = self.model.predict(input_batch, verbose=0)
            return predicted_commands_batch[0].tolist()
        except Exception as e:
            print(f"Error in CerebellumAI process_task: {e}")
            return [0.0] * self.output_size

    def learn(self, sensor_data, true_command_list_or_array):
        """Update model based on sensor data and true command."""
        try:
            input_vec_1d = self._prepare_input_vector(sensor_data)
            target_command_1d = self._prepare_target_command_vector(true_command_list_or_array)

            if not np.any(input_vec_1d): # Skip if input is all zeros
                return

            input_batch = np.reshape(input_vec_1d, [1, self.input_size])
            target_batch = np.reshape(target_command_1d, [1, self.output_size])

            self.model.train_on_batch(input_batch, target_batch)

            # Update Memory
            s_data_list = (
                sensor_data.tolist()
                if isinstance(sensor_data, np.ndarray)
                else list(sensor_data)
            )
            t_cmd_list = (
                true_command_list_or_array.tolist()
                if isinstance(true_command_list_or_array, np.ndarray)
                else list(true_command_list_or_array)
            )
            self.memory.append((s_data_list, t_cmd_list))
            if len(self.memory) > self.max_memory_size:
                self.memory.pop(0)
        except Exception as e:
            print(f"Error in CerebellumAI learn: {e}")


    def consolidate(self):
        """Bedtime: Replay experiences from memory to refine model."""
        if not self.memory:
            self.save_model() # Save even if memory is empty, in case model was loaded and not changed
            return

        # print(f"Consolidating Cerebellum. Memory size: {len(self.memory)}")
        try:
            sensor_data_list_for_batch = [s_data for s_data, _ in list(self.memory)]
            true_commands_list_for_batch = [t_cmd for _, t_cmd in list(self.memory)]

            sensor_data_batch = np.array(
                [self._prepare_input_vector(s_data) for s_data in sensor_data_list_for_batch],
                dtype=np.float32
            )
            true_commands_batch = np.array(
                [self._prepare_target_command_vector(t_cmd) for t_cmd in true_commands_list_for_batch],
                dtype=np.float32
            )

            if sensor_data_batch.shape[0] > 0: # Ensure there's data to train on
                self.model.fit(
                    sensor_data_batch,
                    true_commands_batch,
                    epochs=1,
                    batch_size=min(len(self.memory), 32), # Use a reasonable batch size
                    verbose=0
                )
        except Exception as e:
            print(f"Error during CerebellumAI consolidation training: {e}")

        self.save_model()

    def save_model(self):
        print(f"CerebellumAI: Saving model weights to {self.model_path}")
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        try:
            if not self.model_path.endswith(".weights.h5"):
                print(f"CerebellumAI: Warning: model_path '{self.model_path}' does not end with '.weights.h5'. Keras save_weights prefers this.")
            self.model.save_weights(self.model_path)
            # print(f"CerebellumAI: Model saved to {self.model_path}")
        except Exception as e:
            print(f"CerebellumAI: Error saving model to {self.model_path}: {e}")

    def load_model(self):
        print(f"CerebellumAI: Attempting to load model weights from {self.model_path}")
        if os.path.exists(self.model_path):
            try:
                self.model.load_weights(self.model_path)
                print(f"CerebellumAI: Model weights loaded successfully from {self.model_path}")
            except Exception as e:
                print(f"CerebellumAI: Error loading weights from {self.model_path}: {e}. Model continues with initial Keras weights.")
        else:
            print(f"CerebellumAI: No weights file found at {self.model_path}. Model is using new Keras initializations.")

    def reset_training_data(self):
        """Clears all learned data, resets model, and deletes saved files."""
        print("CerebellumAI: Resetting all training data and model state...")
        # Clear memory
        self.memory = []
        print("CerebellumAI: Memory cleared.")

        # Delete saved files
        if os.path.exists(self.model_path):
            try:
                os.remove(self.model_path)
                print(f"CerebellumAI: Deleted file {self.model_path}")
            except Exception as e:
                print(f"CerebellumAI: Error deleting file {self.model_path}: {e}")
        else:
            print(f"CerebellumAI: File {self.model_path} not found, skipping deletion.")

        # Re-initialize model
        self.model = self._build_model()
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate_learn), loss='mse')
        print("CerebellumAI: Reset complete. Model re-initialized.")

# Example Usage
if __name__ == "__main__":
    # Use a .weights.h5 extension for Keras
    test_model_file = "data/test_cerebellum_model_keras.weights.h5"
    cerebellum_ai = CerebellumAI(model_path=test_model_file)
    print("CerebellumAI (Keras) initialized.")

    sample_sensor_data = np.random.rand(cerebellum_ai.input_size).tolist()
    sample_true_command = (
        np.random.rand(cerebellum_ai.output_size) * 2 - 1 # Commands between -1 and 1
    ).tolist()

    # Get initial weights of a layer for comparison (e.g., first dense layer's kernel)
    initial_weights_sample = None
    if cerebellum_ai.model.layers: # Ensure model has layers
        weights_list = cerebellum_ai.model.layers[0].get_weights()
        if weights_list and len(weights_list[0]) > 0: # Check if kernel exists and is not empty
             initial_weights_sample = weights_list[0][0,0]
             print(f"Initial weight sample (hidden_layer kernel [0,0]): {initial_weights_sample}")

    print("\n--- Testing process_task ---")
    predicted_commands = cerebellum_ai.process_task(sample_sensor_data)
    print(f"Predicted commands: {predicted_commands}")
    if not (len(predicted_commands) == cerebellum_ai.output_size):
        raise AssertionError("process_task output length mismatch")

    print("\n--- Testing learn method ---")
    cerebellum_ai.learn(sample_sensor_data, sample_true_command)
    print(f"Memory after learn: {cerebellum_ai.memory}")

    weights_after_learn_sample = None
    if cerebellum_ai.model.layers:
        weights_list_after_learn = cerebellum_ai.model.layers[0].get_weights()
        if weights_list_after_learn and len(weights_list_after_learn[0]) > 0:
            weights_after_learn_sample = weights_list_after_learn[0][0,0]
            print(f"Weight sample after learn (hidden_layer kernel [0,0]): {weights_after_learn_sample}")
            if initial_weights_sample is not None:
                if np.isclose(initial_weights_sample, weights_after_learn_sample):
                    print("Warning: Weights did not change significantly after learn. This might be okay for a single step or if error was small.")
                else:
                    print("Weights changed after learn, as expected.")

    print("\n--- Testing consolidate method ---")
    # Add more diverse data to memory for better consolidation test
    for _ in range(5):
        cerebellum_ai.learn(np.random.rand(cerebellum_ai.input_size).tolist(),
                            (np.random.rand(cerebellum_ai.output_size) * 2 - 1).tolist())

    cerebellum_ai.consolidate() # This also saves the model

    weights_after_consolidate_sample = None
    if cerebellum_ai.model.layers:
        weights_list_after_consolidate = cerebellum_ai.model.layers[0].get_weights()
        if weights_list_after_consolidate and len(weights_list_after_consolidate[0]) > 0:
            weights_after_consolidate_sample = weights_list_after_consolidate[0][0,0]
            print(f"Weight sample after consolidate (hidden_layer kernel [0,0]): {weights_after_consolidate_sample}")
            if weights_after_learn_sample is not None:
                if np.isclose(weights_after_learn_sample, weights_after_consolidate_sample):
                    print("Warning: Weights did not change significantly after consolidate. Check learning rate or data variance.")
                else:
                    print("Weights changed after consolidate, as expected.")

    print("\n--- Testing model load (after save in consolidate) ---")
    cerebellum_ai_loaded = CerebellumAI(model_path=test_model_file)
    loaded_weights_sample = None
    if cerebellum_ai_loaded.model.layers:
        weights_list_loaded = cerebellum_ai_loaded.model.layers[0].get_weights()
        if weights_list_loaded and len(weights_list_loaded[0]) > 0:
            loaded_weights_sample = weights_list_loaded[0][0,0]
            print(f"Loaded model weight sample (hidden_layer kernel [0,0]): {loaded_weights_sample}")
            if weights_after_consolidate_sample is not None:
                if not np.isclose(weights_after_consolidate_sample, loaded_weights_sample):
                    raise AssertionError("Loaded weights do not match saved weights.")
                print("Model loading confirmed: weights match.")

    # Clean up dummy model file
    if os.path.exists(test_model_file):
        os.remove(test_model_file)
        print(f"\nCleaned up test model file: {test_model_file}")

    print("\nCerebellumAI (Keras) test script finished.")
