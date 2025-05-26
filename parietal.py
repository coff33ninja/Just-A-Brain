# parietal.py (Sensory Integration, Spatial Awareness using Keras)
import numpy as np
import json # Still needed for memory persistence if desired, though Keras handles model weights
import os
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Input # type: ignore
from tensorflow.keras.optimizers.legacy import Adam as LegacyAdam # type: ignore # For compatibility

class ParietalLobeAI:
    def __init__(self, model_path="data/parietal_model.weights.h5"): # Updated extension
        self.input_size = 20  # Sensory data (e.g., sensor readings)
        self.output_size = 3  # Spatial coordinates (e.g., x, y, z) - regression target

        # Learning rate for the Keras optimizer
        self.learning_rate_learn = 0.001 # Standard Keras learning rate
        # learning_rate_consolidate is not directly used unless model is recompiled or optimizer LR is changed.
        # For simplicity, consolidation will use the same LR as initial training.

        self.model_path = model_path
        self.memory_path = self.model_path.replace(".weights.h5", "_memory.json") # Path for memory

        # Build and compile the Keras model
        self.model = self._build_model()
        
        # Load model weights if they exist
        self.load_model()

        # Memory will store (sensory_data_list, true_coords_list) tuples.
        self.memory = []
        self.max_memory_size = 200 # Increased memory size
        self._load_memory() # Load memory if it exists

    def _build_model(self):
        model = Sequential([
            Input(shape=(self.input_size,), name="input_layer"), # Explicit Input layer
            Dense(32, activation='relu', name='dense_hidden1'),
            # Dense(16, activation='relu', name='dense_hidden2'), # Optional second hidden layer
            Dense(self.output_size, activation='linear', name='dense_output') # Linear activation for regression
        ])
        optimizer = LegacyAdam(learning_rate=self.learning_rate_learn)
        model.compile(optimizer=optimizer, loss='mse') # Mean Squared Error for regression
        # model.summary() # Uncomment to print summary when an instance is created
        return model

    def _prepare_input_vector(self, sensor_data):
        """Prepares a 1D numpy array from sensor_data, ensuring correct size."""
        if isinstance(sensor_data, list):
            input_vec_list = sensor_data
        elif isinstance(sensor_data, np.ndarray):
            input_vec_list = sensor_data.flatten().tolist()
        else:
            print(f"Parietal Lobe: Warning: Unexpected sensor_data type {type(sensor_data)}. Using zeros.")
            input_vec_list = [0.0] * self.input_size

        current_len = len(input_vec_list)
        if current_len == self.input_size:
            return np.array(input_vec_list, dtype=float)
        elif current_len < self.input_size:
            # Pad with zeros
            return np.array(input_vec_list + [0.0] * (self.input_size - current_len), dtype=float)
        else:
            # Truncate
            return np.array(input_vec_list[:self.input_size], dtype=float)

    def _prepare_target_coords_vector(self, true_coords):
        """Prepares a 1D numpy array from true_coords, ensuring correct size."""
        if isinstance(true_coords, list):
            target_list = true_coords
        elif isinstance(true_coords, np.ndarray):
            target_list = true_coords.flatten().tolist()
        else:
            print(f"Parietal Lobe: Warning: Unexpected true_coords type {type(true_coords)}. Using zeros.")
            target_list = [0.0] * self.output_size
        
        current_len = len(target_list)
        if current_len == self.output_size:
            return np.array(target_list, dtype=float)
        elif current_len < self.output_size:
            return np.array(target_list + [0.0] * (self.output_size - current_len), dtype=float)
        else:
            return np.array(target_list[:self.output_size], dtype=float)

    def process_task(self, sensory_data):
        print("Parietal Lobe: Processing task with sensory data...")
        input_vec_1d = self._prepare_input_vector(sensory_data)
        
        # Check if input_vec_1d is all zeros, which might indicate a preparation error or actual zero input
        if not np.any(input_vec_1d) and not np.all(np.array(sensory_data, dtype=float) == 0): # If not all zeros originally
            print("Parietal Lobe: Input vector became all zeros after preparation, possibly due to invalid input type. Returning default.")
            return [0.0] * self.output_size

        input_batch = np.reshape(input_vec_1d, [1, self.input_size])
        
        try:
            predicted_coords_batch = self.model.predict(input_batch, verbose=0)
            predicted_coords_list = predicted_coords_batch[0].tolist()
            print(f"Parietal Lobe: Predicted coordinates: {predicted_coords_list}")
            return predicted_coords_list
        except Exception as e:
            print(f"Parietal Lobe: Error during model prediction: {e}")
            return [0.0] * self.output_size

    def learn(self, sensory_data, true_coords): # Assuming true_coords, not spatial_error
        print("Parietal Lobe: Learning with sensory data and true coordinates...")
        input_vec_1d = self._prepare_input_vector(sensory_data)
        target_coords_1d = self._prepare_target_coords_vector(true_coords)

        # Check if input_vec_1d is all zeros (similar to process_task)
        if not np.any(input_vec_1d) and not np.all(np.array(sensory_data, dtype=float) == 0):
            print("Parietal Lobe: Input vector for learning is all zeros after preparation. Skipping learning.")
            return

        input_batch = np.reshape(input_vec_1d, [1, self.input_size])
        target_batch = np.reshape(target_coords_1d, [1, self.output_size])

        try:
            history = self.model.fit(input_batch, target_batch, epochs=1, verbose=0)
            loss = history.history.get('loss', [float('nan')])[0]
            print(f"Parietal Lobe: Training complete. Loss: {loss:.4f}")

            # Store raw forms in memory
            s_data_to_store = sensory_data.tolist() if isinstance(sensory_data, np.ndarray) else list(sensory_data)
            t_coords_to_store = true_coords.tolist() if isinstance(true_coords, np.ndarray) else list(true_coords)
            self.memory.append((s_data_to_store, t_coords_to_store))
            if len(self.memory) > self.max_memory_size:
                self.memory.pop(0)
        except Exception as e:
            print(f"Parietal Lobe: Error during model training: {e}")


    def save_model(self):
        print(f"Parietal Lobe: Saving model weights to {self.model_path}...")
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        try:
            self.model.save_weights(self.model_path)
            print("Parietal Lobe: Model weights saved.")
        except Exception as e:
            print(f"Parietal Lobe: Error saving model weights: {e}")

    def load_model(self):
        if os.path.exists(self.model_path):
            print(f"Parietal Lobe: Loading model weights from {self.model_path}...")
            try:
                # Model must be built before loading weights
                if not hasattr(self, 'model') or self.model is None:
                     self.model = self._build_model() # Should already be called in __init__
                self.model.load_weights(self.model_path)
                print("Parietal Lobe: Model weights loaded successfully.")
            except Exception as e:
                print(f"Parietal Lobe: Error loading model weights: {e}. Model remains initialized.")
        else:
            print(f"Parietal Lobe: No pre-trained weights found at {self.model_path}. Model is newly initialized.")

    def _save_memory(self):
        print("Parietal Lobe: Saving memory...")
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        try:
            with open(self.memory_path, "w") as f:
                json.dump({"memory": self.memory}, f)
            print("Parietal Lobe: Memory saved.")
        except Exception as e:
            print(f"Parietal Lobe: Error saving memory: {e}")

    def _load_memory(self):
        if os.path.exists(self.memory_path):
            print("Parietal Lobe: Loading memory...")
            try:
                with open(self.memory_path, "r") as f:
                    loaded_data = json.load(f)
                    self.memory = loaded_data.get("memory", [])
                # Ensure memory items are tuples of lists (or convert if necessary)
                self.memory = [
                    (list(item[0]), list(item[1])) for item in self.memory 
                    if isinstance(item, (list, tuple)) and len(item) == 2
                ]
                print("Parietal Lobe: Memory loaded.")
            except Exception as e:
                print(f"Parietal Lobe: Error loading memory: {e}. Initializing empty memory.")
                self.memory = []
        else:
            print("Parietal Lobe: No memory file found. Initializing empty memory.")
            self.memory = []


    def consolidate(self):
        print("Parietal Lobe: Starting consolidation...")
        if not self.memory:
            print("Parietal Lobe: Memory is empty. Nothing to consolidate.")
        else:
            print(f"Parietal Lobe: Consolidating {len(self.memory)} experiences from memory.")
            # Example: Batch processing for consolidation
            sensory_data_batch = []
            true_coords_batch = []
            
            # Simple batching: process all memory items if not too large, or a sample
            # For Keras, it's often better to fit on the entire dataset (or large batches)
            # if memory is reasonably sized.
            
            for s_data, t_coords in self.memory:
                sensory_data_batch.append(self._prepare_input_vector(s_data))
                true_coords_batch.append(self._prepare_target_coords_vector(t_coords))
            
            if sensory_data_batch: # If any valid data was prepared
                input_data_np = np.array(sensory_data_batch)
                target_data_np = np.array(true_coords_batch)
                print(f"Parietal Lobe: Training consolidation batch of size {input_data_np.shape[0]}...")
                try:
                    self.model.fit(input_data_np, target_data_np, epochs=1, verbose=0, batch_size=16) # Example batch_size
                    print("Parietal Lobe: Consolidation training complete.")
                except Exception as e:
                    print(f"Parietal Lobe: Error during consolidation training: {e}")
            else:
                print("Parietal Lobe: No valid data prepared for consolidation training.")

        self.save_model() # Save updated model weights
        self._save_memory() # Save memory state (e.g., if it was pruned or modified)
        print("Parietal Lobe: Consolidation complete.")

# Example Usage
if __name__ == "__main__":
    print("\n--- Testing ParietalLobeAI (Keras FFN) ---")
    test_model_path = "data/test_parietal_keras.weights.h5"
    
    # Clean up old test files
    if os.path.exists(test_model_path):
        os.remove(test_model_path)
    if os.path.exists(test_model_path.replace(".weights.h5", "_memory.json")):
        os.remove(test_model_path.replace(".weights.h5", "_memory.json"))

    parietal_ai = ParietalLobeAI(model_path=test_model_path)
    parietal_ai.model.summary()

    print("\n--- Testing Data Preparation ---")
    sample_sensor_data_short = [1.0, 2.0]
    prepared_input = parietal_ai._prepare_input_vector(sample_sensor_data_short)
    print(f"Prepared input for short data ({sample_sensor_data_short}): {prepared_input}, shape: {prepared_input.shape}")
    assert prepared_input.shape == (parietal_ai.input_size,)

    sample_coords_long = [0.1, 0.2, 0.3, 0.4]
    prepared_target = parietal_ai._prepare_target_coords_vector(sample_coords_long)
    print(f"Prepared target for long data ({sample_coords_long}): {prepared_target}, shape: {prepared_target.shape}")
    assert prepared_target.shape == (parietal_ai.output_size,)
    
    print("\n--- Testing process_task ---")
    sensor_input = np.random.rand(parietal_ai.input_size).tolist()
    predicted_coords = parietal_ai.process_task(sensor_input)
    print(f"Predicted coords for random input: {predicted_coords}")
    assert len(predicted_coords) == parietal_ai.output_size

    print("\n--- Testing learn ---")
    true_spatial_coords = np.random.rand(parietal_ai.output_size).tolist()
    parietal_ai.learn(sensor_input, true_spatial_coords)
    print(f"Memory size after one learn call: {len(parietal_ai.memory)}")

    # Simulate a few more learning steps
    for i in range(5):
        s_data = np.random.rand(parietal_ai.input_size).tolist()
        t_coords = (np.random.rand(parietal_ai.output_size) * (i+1)).tolist() # Make targets somewhat different
        parietal_ai.learn(s_data, t_coords)
    print(f"Memory size after several learn calls: {len(parietal_ai.memory)}")


    print("\n--- Testing Consolidation ---")
    parietal_ai.consolidate()

    print("\n--- Testing Save/Load ---")
    parietal_ai.save_model() # Includes saving memory
    
    # Create a new instance to test loading
    print("\nCreating new ParietalLobeAI instance for loading test...")
    parietal_ai_loaded = ParietalLobeAI(model_path=test_model_path)
    
    # Test if model weights were loaded (simple check on one layer's weights)
    # Need to access weights of a specific layer, e.g., the output layer
    if hasattr(parietal_ai.model.get_layer("dense_output"), "kernel") and \
       hasattr(parietal_ai_loaded.model.get_layer("dense_output"), "kernel"):
        original_weights_output = parietal_ai.model.get_layer("dense_output").kernel.numpy()
        loaded_weights_output = parietal_ai_loaded.model.get_layer("dense_output").kernel.numpy()
        if np.array_equal(original_weights_output, loaded_weights_output):
            print("Model weights loaded successfully (output layer kernel weights match).")
        else:
            print("Model weights loading failed or mismatch. This might happen if consolidation changed weights after save and before this check.")
    else:
         print("Could not access kernel weights for 'dense_output' layer to compare.")

    if len(parietal_ai_loaded.memory) == len(parietal_ai.memory):
        print(f"Memory loaded successfully with {len(parietal_ai_loaded.memory)} items.")
    else:
        print("Memory loading failed or mismatch.")

    # Clean up
    if os.path.exists(test_model_path):
        os.remove(test_model_path)
    if os.path.exists(test_model_path.replace(".weights.h5", "_memory.json")):
        os.remove(test_model_path.replace(".weights.h5", "_memory.json"))
    print("\nCleaned up test files.")
    
    print("\nParietal Lobe AI (Keras FFN) test script finished.")
