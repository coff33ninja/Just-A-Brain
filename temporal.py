# temporal.py (Memory, Language, and Text-to-Visual Association)
import numpy as np
import json
import os
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Dense, Input 
from tensorflow.keras.optimizers import Adam


class TemporalLobeAI:
    def __init__(
        self,
        model_path="data/temporal_combined_model.weights.h5", 
        memory_path="data/temporal_memory.json",
    ):
        self.input_size = 15
        self.output_size = 10  # Dimension of text embeddings (embedding_output_name)
        self.text_hidden_size = 32
        
        self.visual_output_size = 5  # Number of visual classes (visual_label_output_name)
        self.visual_assoc_hidden_size = 16

        self.learning_rate_learn = 0.01
        
        self.model = self._build_combined_model()
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate_learn),
            loss={
                'text_embedding_output': 'mse', 
                'visual_label_output': 'sparse_categorical_crossentropy' 
            },
            loss_weights={'text_embedding_output': 0.5, 'visual_label_output': 0.5}
        )

        self.memory_db = [] 
        self.cross_modal_memory = [] 
        self.max_cross_modal_memory_size = 100 

        self.model_path = model_path 
        self.arch_path = os.path.splitext(model_path)[0] + "_arch.json" 
        self.memory_path = memory_path
        
        self.load_model() 

    def _build_combined_model(self):
        text_input = Input(shape=(self.input_size,), name="text_input_vector")
        shared_hidden_text = Dense(self.text_hidden_size, activation='tanh', name='text_hidden_layer')(text_input)
        text_embedding_output = Dense(self.output_size, activation='linear', name='text_embedding_output')(shared_hidden_text)
        
        visual_assoc_hidden = Dense(self.visual_assoc_hidden_size, activation='tanh', name='visual_assoc_hidden')(text_embedding_output)
        visual_label_output = Dense(self.visual_output_size, activation='softmax', name='visual_label_output')(visual_assoc_hidden)
        
        model = Model(inputs=text_input, outputs=[text_embedding_output, visual_label_output])
        return model

    def _to_numerical_vector(self, data, size, context="input"):
        if isinstance(data, str):
            hash_val = hash(data)
            np.random.seed(hash_val % (2**32 - 1))
            vec = np.random.rand(size)
            np.random.seed(None)
            return vec.astype(np.float32) 
        try:
            vec = np.array(data, dtype=np.float32).flatten() 
        except ValueError:
            return np.zeros(size, dtype=np.float32) 
        
        if vec.shape[0] == size:
            return vec
        elif vec.shape[0] < size:
            return np.concatenate((vec, np.zeros(size - vec.shape[0], dtype=np.float32)))
        else:
            return vec[:size]

    def process_task(self, input_data_item, predict_visual=False):
        actual_input_data = (
            input_data_item[-1]
            if isinstance(input_data_item, list) and input_data_item
            else input_data_item
        )
        input_text_1d = self._to_numerical_vector(actual_input_data, self.input_size)
        input_batch = np.reshape(input_text_1d, [1, self.input_size])
        
        try:
            predictions = self.model.predict(input_batch, verbose=0)
            text_embedding = predictions[0][0] 

            if predict_visual:
                visual_label_probs = predictions[1][0] 
                predicted_visual_label = np.argmax(visual_label_probs)
                return (text_embedding.tolist(), int(predicted_visual_label))
            else:
                return text_embedding.tolist()
        except Exception as e:
            print(f"Error during TemporalLobeAI self.model.predict: {e}")
            if predict_visual:
                return ([0.0] * self.output_size, -1) 
            else:
                return [0.0] * self.output_size


    def learn(self, sequence, visual_label_as_context=None):
        if visual_label_as_context is None:
            # Text Sequence Learning Path
            if (
                isinstance(sequence, list)
                and sequence
                and all(isinstance(item, tuple) and len(item) == 2 for item in sequence)
            ):
                self.memory_db.append(sequence)
                if len(self.memory_db) > 100: 
                    self.memory_db.pop(0)

                for input_data, target_output in sequence:
                    try:
                        input_text_1d = self._to_numerical_vector(input_data, self.input_size)
                        target_text_embedding_1d = self._to_numerical_vector(target_output, self.output_size)

                        input_batch = np.reshape(input_text_1d, [1, self.input_size])
                        target_embedding_batch = np.reshape(target_text_embedding_1d, [1, self.output_size])
                        
                        y_targets = {
                            'text_embedding_output': target_embedding_batch,
                            'visual_label_output': np.array([[0]], dtype='int32') # Dummy visual target
                        }
                        sample_weights = {'text_embedding_output': 1.0, 'visual_label_output': 0.0}
                        
                        self.model.train_on_batch(input_batch, y_targets, sample_weight=sample_weights)
                    except Exception as e:
                        print(f"Error during TemporalLobeAI learn (text sequence path): {e}")
                        continue
        else:
            # Visual Association Learning Path
            if (0 <= visual_label_as_context < self.visual_output_size):
                if sequence and sequence[0] and isinstance(sequence[0][0], str):
                    text_input_str_for_assoc = sequence[0][0]
                    
                    self.cross_modal_memory.append(
                        (text_input_str_for_assoc, visual_label_as_context)
                    )
                    if len(self.cross_modal_memory) > self.max_cross_modal_memory_size:
                        self.cross_modal_memory.pop(0)

                    try:
                        input_text_1d = self._to_numerical_vector(text_input_str_for_assoc, self.input_size)
                        input_batch = np.reshape(input_text_1d, [1, self.input_size])

                        current_outputs = self.model.predict(input_batch, verbose=0)
                        self_target_embedding_batch = current_outputs[0] 
                        
                        visual_label_batch = np.array([[visual_label_as_context]], dtype='int32')
                        
                        y_targets = {
                            'text_embedding_output': self_target_embedding_batch,
                            'visual_label_output': visual_label_batch
                        }
                        self.model.train_on_batch(input_batch, y_targets)
                    except Exception as e:
                         print(f"Error during TemporalLobeAI learn (visual association path): {e}")


    def consolidate(self):
        print("TemporalLobeAI: Starting consolidation...")

        # Consolidate text processing (memory_db)
        if self.memory_db:
            inputs_for_text_fit = []
            targets_for_text_embedding_fit = []
            for sequence_item in list(self.memory_db):
                if not (
                    isinstance(sequence_item, list)
                    and sequence_item
                    and all(isinstance(p, tuple) and len(p) == 2 for p in sequence_item)
                ):
                    continue
                for input_data, target_output in sequence_item:
                    try:
                        input_text_1d = self._to_numerical_vector(input_data, self.input_size)
                        target_text_embedding_1d = self._to_numerical_vector(target_output, self.output_size)
                        inputs_for_text_fit.append(input_text_1d)
                        targets_for_text_embedding_fit.append(target_text_embedding_1d)
                    except Exception as e:
                        print(f"Error preparing text data for consolidation: {e}")
                        continue
            
            if inputs_for_text_fit and targets_for_text_embedding_fit:
                try:
                    input_vectors_batch = np.array(inputs_for_text_fit, dtype=np.float32)
                    target_embeddings_batch = np.array(targets_for_text_embedding_fit, dtype=np.float32)
                    
                    dummy_visual_targets = np.zeros((len(target_embeddings_batch), 1), dtype='int32')
                    y_targets_text = {
                        'text_embedding_output': target_embeddings_batch,
                        'visual_label_output': dummy_visual_targets
                    }
                    sample_weights_text = {'text_embedding_output': 1.0, 'visual_label_output': 0.0}
                    
                    print(f"Consolidating {len(input_vectors_batch)} text experiences...")
                    self.model.fit(
                        input_vectors_batch, 
                        y_targets_text, 
                        sample_weight=sample_weights_text,
                        epochs=1, 
                        batch_size=min(len(input_vectors_batch), 32), 
                        verbose=0
                    )
                except Exception as e:
                    print(f"Error during TemporalLobeAI text_model.fit in consolidation (memory_db): {e}")
        else:
            print("TemporalLobeAI: memory_db is empty. Skipping text consolidation.")

        # Consolidate visual association (cross_modal_memory)
        if self.cross_modal_memory:
            inputs_for_visual_fit = []
            targets_for_embedding_in_visual_fit = []
            targets_for_visual_label_fit = []

            for text_input_str, true_visual_label in list(self.cross_modal_memory):
                try:
                    if not (0 <= true_visual_label < self.visual_output_size):
                        print(f"Skipping invalid visual label {true_visual_label} in cross_modal_memory.")
                        continue

                    assoc_input_text_1d = self._to_numerical_vector(text_input_str, self.input_size)
                    assoc_input_batch = np.reshape(assoc_input_text_1d, [1, self.input_size])
                    
                    current_outputs = self.model.predict(assoc_input_batch, verbose=0)
                    self_target_embedding = current_outputs[0][0] # Get the first (and only) embedding from the batch

                    inputs_for_visual_fit.append(assoc_input_text_1d)
                    targets_for_embedding_in_visual_fit.append(self_target_embedding)
                    targets_for_visual_label_fit.append(true_visual_label) # Store as single int for sparse crossentropy
                except Exception as e:
                    print(f"Error preparing visual association data for consolidation: {e}")
                    continue
            
            if inputs_for_visual_fit:
                try:
                    input_visual_batch = np.array(inputs_for_visual_fit, dtype=np.float32)
                    target_embedding_visual_batch = np.array(targets_for_embedding_in_visual_fit, dtype=np.float32)
                    target_visual_label_batch = np.array(targets_for_visual_label_fit, dtype='int32').reshape(-1,1) # Ensure correct shape for sparse

                    y_targets_visual = {
                        'text_embedding_output': target_embedding_visual_batch,
                        'visual_label_output': target_visual_label_batch
                    }
                    
                    print(f"Consolidating {len(input_visual_batch)} visual association experiences...")
                    self.model.fit(
                        input_visual_batch, 
                        y_targets_visual, 
                        epochs=1, 
                        batch_size=min(len(input_visual_batch), 32), 
                        verbose=0
                    )
                except Exception as e:
                    print(f"Error during TemporalLobeAI model.fit in consolidation (cross_modal_memory): {e}")
        else:
            print("TemporalLobeAI: cross_modal_memory is empty. Skipping visual association consolidation.")
        
        self.save_model()
        self.save_memory()
        print("TemporalLobeAI: Consolidation finished.")


    def save_model(self):
        print(f"TemporalLobeAI: Saving combined model weights to {self.model_path}")
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        try:
            if not self.model_path.endswith(".weights.h5"):
                 print(f"TemporalLobeAI: Warning: model_path '{self.model_path}' does not end with '.weights.h5'. Keras save_weights prefers this.")
            self.model.save_weights(self.model_path) 
        except Exception as e:
            print(f"TemporalLobeAI: Error saving Keras combined model weights: {e}")

        arch_data = {
            "input_size": self.input_size,
            "text_hidden_size": self.text_hidden_size,
            "output_size": self.output_size, 
            "visual_assoc_hidden_size": self.visual_assoc_hidden_size, 
            "visual_output_size": self.visual_output_size,
        }
        try:
            with open(self.arch_path, "w") as f_arch:
                json.dump(arch_data, f_arch)
        except Exception as e:
            print(f"TemporalLobeAI: Error saving architecture JSON to {self.arch_path}: {e}")


    def save_memory(self):
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        memory_to_save = {
            "memory_db": self.memory_db,
            "cross_modal_memory": self.cross_modal_memory 
        }
        try:
            with open(self.memory_path, "w") as f:
                json.dump(memory_to_save, f)
        except Exception as e:
            print(f"TemporalLobeAI: Error saving memory to {self.memory_path}: {e}")


    def load_model(self):
        self.memory_db = [] 
        self.cross_modal_memory = [] 

        if os.path.exists(self.arch_path):
            try:
                with open(self.arch_path, "r") as f_arch:
                    arch_data = json.load(f_arch)
                if not (arch_data.get("input_size") == self.input_size and
                        arch_data.get("text_hidden_size") == self.text_hidden_size and
                        arch_data.get("output_size") == self.output_size and
                        arch_data.get("visual_assoc_hidden_size") == self.visual_assoc_hidden_size and
                        arch_data.get("visual_output_size") == self.visual_output_size):
                    print("TemporalLobeAI: Architecture parameters in JSON do not match current model. Re-initializing model.")
                else:
                    print(f"TemporalLobeAI: Architecture parameters loaded from {self.arch_path} and match current config.")
            except Exception as e:
                print(f"TemporalLobeAI: Error loading or verifying architecture JSON from {self.arch_path}: {e}.")
        
        if os.path.exists(self.model_path):
            try:
                self.model.load_weights(self.model_path) 
                print(f"TemporalLobeAI: Keras combined model weights loaded successfully from {self.model_path}.")
            except Exception as e:
                print(f"TemporalLobeAI: Error loading Keras combined model weights from {self.model_path}: {e}. Model uses new initializations.")
        else:
            print(f"TemporalLobeAI: No Keras combined model weights file found at {self.model_path}. Model uses new initializations.")

        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, "r") as f_mem:
                    loaded_memory_data = json.load(f_mem)
                if isinstance(loaded_memory_data, dict):
                    self.memory_db = loaded_memory_data.get("memory_db", [])
                    self.cross_modal_memory = loaded_memory_data.get("cross_modal_memory", []) 
                elif isinstance(loaded_memory_data, list): 
                    self.memory_db = loaded_memory_data
                    self.cross_modal_memory = [] 
                
                temp_memory_db = []
                if isinstance(self.memory_db, list):
                    for item_sequence in self.memory_db:
                        if isinstance(item_sequence, list) and all(
                            isinstance(pair, list) and len(pair) == 2
                            for pair in item_sequence
                        ):
                            temp_memory_db.append(
                                [tuple(p) for p in item_sequence]
                            ) 
                self.memory_db = temp_memory_db
                
                temp_cross_modal_memory = []
                if isinstance(self.cross_modal_memory, list):
                    for item_pair in self.cross_modal_memory:
                        if isinstance(item_pair, list) and len(item_pair) == 2:
                            try: 
                                temp_cross_modal_memory.append(
                                    (item_pair[0], int(item_pair[1]))
                                )
                            except (ValueError, TypeError):
                                print(
                                    f"TemporalLobeAI: Skipping malformed cross_modal_memory item during load: {item_pair}"
                                )
                self.cross_modal_memory = temp_cross_modal_memory
                print("TemporalLobeAI: Memory loaded.")
            except Exception as e_json_mem: 
                print(
                    f"TemporalLobeAI: Error loading/processing memory file {self.memory_path}: {e_json_mem}. Initializing empty memory."
                )
                self.memory_db = []
                self.cross_modal_memory = []
        else:
            print(f"TemporalLobeAI: No memory file found at {self.memory_path}. Initializing empty memory.")
            self.memory_db = []
            self.cross_modal_memory = []


# Example Usage
if __name__ == "__main__":
    test_model_weights_path = "data/test_temporal_combined.weights.h5"
    test_arch_path = "data/test_temporal_combined_arch.json" 
    test_memory_path = "data/test_temporal_combined_memory.json"

    ai = TemporalLobeAI(
        model_path=test_model_weights_path,
        memory_path=test_memory_path,
    )
    print("TemporalLobeAI (Keras Combined Model) initialized.")
    ai.model.summary() 

    text_input_example = "a descriptive sentence"
    
    embedding = ai.process_task(text_input_example, predict_visual=False)
    print(f"\nText Embedding for '{text_input_example}': {embedding} (length: {len(embedding)})")
    assert len(embedding) == ai.output_size

    embedding, visual_label = ai.process_task(text_input_example, predict_visual=True)
    print(f"Text Embedding: {embedding} (length: {len(embedding)})")
    print(f"Predicted Visual Label: {visual_label}")
    assert len(embedding) == ai.output_size
    assert isinstance(visual_label, int)

    print("\n--- Simulating learn ---")
    ai.learn([("text q1", "text a1")], visual_label_as_context=None) 
    ai.learn([("text for visual", "related text")], visual_label_as_context=0) 
    print(f"Memory DB size: {len(ai.memory_db)}")
    print(f"Cross-modal Memory size: {len(ai.cross_modal_memory)}")

    print("\n--- Simulating consolidate ---")
    ai.consolidate() # Now includes training logic

    print("\n--- Testing model load ---")
    ai_loaded = TemporalLobeAI(model_path=test_model_weights_path, memory_path=test_memory_path)
    print(f"Loaded Memory DB size: {len(ai_loaded.memory_db)}")
    print(f"Loaded Cross-modal Memory size: {len(ai_loaded.cross_modal_memory)}")
    assert len(ai_loaded.memory_db) == len(ai.memory_db)
    assert len(ai_loaded.cross_modal_memory) == len(ai.cross_modal_memory)


    for p in [test_model_weights_path, test_arch_path, test_memory_path]:
        if os.path.exists(p):
            os.remove(p)
    print("\nTemporal Lobe AI (Keras Combined Model) test finished and files cleaned up.")
