# parietal.py (Sensory Integration, Spatial Awareness)
import numpy as np
import os
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Input # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore


class ParietalLobeAI:
    def __init__(self, model_path="data/parietal_model.weights.h5"):
        self.input_size = 20  # Sensory data (e.g., sensor readings)
        self.hidden_size = 25  # Size of the new hidden layer
        self.output_size = (
            3  # Spatial coordinates (e.g., x, y, z error or target position)
        )

        # Learning rates
        self.learning_rate_learn = 0.01
        # self.learning_rate_consolidate = 0.005 # Less directly applicable with Keras single optimizer

        # Memory will store (sensory_data_list, true_coords_list) tuples.
        self.memory = []
        self.max_memory_size = 100  # Max memory size
        self.model_path = model_path

        self.model = self._build_model()
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate_learn), loss='mse')

        self.load_model()

    def _build_model(self):
        model = Sequential(
            [
                Input(shape=(self.input_size,), name="input_layer"),
                Dense(self.hidden_size, activation='tanh', name='hidden_layer'),
                Dense(self.output_size, activation='linear', name='output_layer'),
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

    def _prepare_target_coords_vector(self, true_coords_list_or_array):
        """Prepares a 1D numpy array from true_coords, ensuring correct size and dtype."""
        if isinstance(true_coords_list_or_array, list):
            target_list = true_coords_list_or_array
        elif isinstance(true_coords_list_or_array, np.ndarray):
            target_list = true_coords_list_or_array.flatten().tolist()
        else:
            target_list = [0.0] * self.output_size

        if len(target_list) < self.output_size:
            target_list.extend([0.0] * (self.output_size - len(target_list)))
        elif len(target_list) > self.output_size:
            target_list = target_list[: self.output_size]
        return np.array(target_list, dtype=np.float32)

    def process_task(self, sensory_data):
        try:
            input_vec_1d = self._prepare_input_vector(sensory_data)
            input_batch = np.reshape(input_vec_1d, [1, self.input_size])
            predicted_coords_batch = self.model.predict(input_batch, verbose=0)
            return predicted_coords_batch[0].tolist()
        except Exception as e:
            print(f"Error in ParietalLobeAI process_task: {e}")
            return [0.0] * self.output_size

    def learn(self, sensory_data, true_coords):
        """Update based on spatial error."""
        try:
            input_vec_1d = self._prepare_input_vector(sensory_data)
            target_coords_1d = self._prepare_target_coords_vector(true_coords)

            if not np.any(input_vec_1d):
                return

            input_batch = np.reshape(input_vec_1d, [1, self.input_size])
            target_batch = np.reshape(target_coords_1d, [1, self.output_size])

            self.model.train_on_batch(input_batch, target_batch)

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
        except Exception as e:
            print(f"Error in ParietalLobeAI learn: {e}")


    def consolidate(self):
        """Bedtime: Replay experiences from memory to refine model."""
        if not self.memory:
            self.save_model()
            return

        # print(f"Consolidating Parietal Lobe. Memory size: {len(self.memory)}")

        try:
            sensory_data_list_for_batch = [s_data for s_data, _ in list(self.memory)]
            true_coords_list_for_batch = [t_coords for _, t_coords in list(self.memory)]

            sensory_data_batch = np.array(
                [self._prepare_input_vector(s_data) for s_data in sensory_data_list_for_batch],
                dtype=np.float32
            )
            true_coords_batch = np.array(
                [self._prepare_target_coords_vector(t_coords) for t_coords in true_coords_list_for_batch],
                dtype=np.float32
            )

            if sensory_data_batch.shape[0] > 0: # Ensure there's data to train on
                self.model.fit(
                    sensory_data_batch,
                    true_coords_batch,
                    epochs=1,
                    batch_size=min(len(self.memory), 32),
                    verbose=0
                )
        except Exception as e:
            print(f"Error during ParietalLobeAI consolidation training: {e}")

        self.save_model()

    def save_model(self):
        print(f"ParietalLobeAI: Saving model weights to {self.model_path}")
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        try:
            if not self.model_path.endswith(".weights.h5"):
                print(f"ParietalLobeAI: Warning: model_path '{self.model_path}' does not end with '.weights.h5'. Keras save_weights prefers this.")
            self.model.save_weights(self.model_path)
            # print(f"ParietalLobeAI: Model saved to {self.model_path}")
        except Exception as e:
            print(f"ParietalLobeAI: Error saving model to {self.model_path}: {e}")

    def load_model(self):
        print(f"ParietalLobeAI: Attempting to load model weights from {self.model_path}")
        if os.path.exists(self.model_path):
            try:
                self.model.load_weights(self.model_path)
                print(f"ParietalLobeAI: Model weights loaded successfully from {self.model_path}")
            except Exception as e:
                print(f"ParietalLobeAI: Error loading weights from {self.model_path}: {e}. Model continues with initial Keras weights.")
        else:
            print(f"ParietalLobeAI: No weights file found at {self.model_path}. Model is using new Keras initializations.")


# Example Usage
if __name__ == "__main__":
    # Use a .weights.h5 extension for Keras
    test_model_file = "data/test_parietal_model_keras.weights.h5"
    parietal_ai = ParietalLobeAI(model_path=test_model_file)
    print("ParietalLobeAI (Keras) initialized.")

    sample_sensor_data = np.random.rand(parietal_ai.input_size).tolist()
    sample_true_coords = np.random.rand(parietal_ai.output_size).tolist()

    # Get initial weights of a layer for comparison (e.g., first dense layer's kernel)
    initial_weights_sample = None
    if parietal_ai.model.layers:
        initial_weights_sample = parietal_ai.model.layers[0].get_weights()[0][0,0] # kernel of first dense layer
        print(f"Initial weight sample (hidden_layer kernel [0,0]): {initial_weights_sample}")


    print("\n--- Testing process_task ---")
    predicted_coords = parietal_ai.process_task(sample_sensor_data)
    print(f"Predicted coords: {predicted_coords}")
    if not (len(predicted_coords) == parietal_ai.output_size):
        raise AssertionError("process_task output length mismatch")

    print("\n--- Testing learn method ---")
    parietal_ai.learn(sample_sensor_data, sample_true_coords)
    print(f"Memory after learn: {parietal_ai.memory}")

    weights_after_learn_sample = None
    if parietal_ai.model.layers:
        weights_after_learn_sample = parietal_ai.model.layers[0].get_weights()[0][0,0]
        print(f"Weight sample after learn (hidden_layer kernel [0,0]): {weights_after_learn_sample}")
        if initial_weights_sample is not None:
            if np.isclose(initial_weights_sample, weights_after_learn_sample):
                 print("Warning: Weights did not change significantly after learn. This might be okay for a single step or if error was small.")
            else:
                print("Weights changed after learn, as expected.")


    print("\n--- Testing consolidate method ---")
    # Add more diverse data to memory for better consolidation test
    for _ in range(5):
        parietal_ai.learn(np.random.rand(parietal_ai.input_size).tolist(),
                          np.random.rand(parietal_ai.output_size).tolist())

    parietal_ai.consolidate() # This also saves the model

    weights_after_consolidate_sample = None
    if parietal_ai.model.layers:
        weights_after_consolidate_sample = parietal_ai.model.layers[0].get_weights()[0][0,0]
        print(f"Weight sample after consolidate (hidden_layer kernel [0,0]): {weights_after_consolidate_sample}")
        if weights_after_learn_sample is not None:
            if np.isclose(weights_after_learn_sample, weights_after_consolidate_sample):
                print("Warning: Weights did not change significantly after consolidate. Check learning rate or data variance.")
            else:
                print("Weights changed after consolidate, as expected.")

    print("\n--- Testing model load (after save in consolidate) ---")
    parietal_ai_loaded = ParietalLobeAI(model_path=test_model_file)
    loaded_weights_sample = None
    if parietal_ai_loaded.model.layers:
        loaded_weights_sample = parietal_ai_loaded.model.layers[0].get_weights()[0][0,0]
        print(f"Loaded model weight sample (hidden_layer kernel [0,0]): {loaded_weights_sample}")
        if weights_after_consolidate_sample is not None:
            if not np.isclose(weights_after_consolidate_sample, loaded_weights_sample):
                raise AssertionError("Loaded weights do not match saved weights.")
            print("Model loading confirmed: weights match.")


    # Clean up dummy model file
    if os.path.exists(test_model_file):
        os.remove(test_model_file)
        print(f"\nCleaned up test model file: {test_model_file}")

    print("\nParietalLobeAI (Keras) test script finished.")
