# temporal.py (Memory, Language, and Text-to-Visual Association with RNN/LSTM)
import numpy as np
import json
import os
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense # Using LSTM # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.optimizers.legacy import Adam as LegacyAdam # type: ignore # For compatibility

class TemporalLobeAI:
    def __init__(
        self,
        model_path="data/temporal_model.weights.h5", # Updated extension
        memory_path="data/temporal_memory.json", # For cross-modal memory
        tokenizer_path="data/temporal_tokenizer.json"
    ):
        # RNN/LSTM Parameters
        self.vocab_size = 2000  # Max words in vocabulary
        self.max_sequence_length = 20  # Max length of text sequences
        self.embedding_dim = 50  # Dimension of word embeddings
        self.lstm_units = 64 # Number of units in LSTM layer
        
        # Output sizes
        self.output_size = 10 # Dimension of the text embedding/context vector from RNN
        self.visual_output_size = 5 # Number of visual labels (for association)

        # Learning rate (can be part of optimizer, but good to have as attribute)
        self.learning_rate_learn = 0.001 # Used in compile

        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<unk>")
        
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.memory_path = memory_path # Stores cross-modal associations

        # Memory for cross-modal associations (text_input_str, visual_label)
        self.cross_modal_memory = []
        self.max_cross_modal_memory_size = 200 # Increased size

        # Build and compile the model
        self.model = self._build_model() # This also compiles the model
        
        # Load tokenizer and model weights
        self.load_model() # Handles both tokenizer and weights

    def _build_model(self):
        # Text processing pathway
        text_input = Input(shape=(self.max_sequence_length,), name="text_input")
        embedding_layer = Embedding(input_dim=self.vocab_size, 
                                    output_dim=self.embedding_dim, 
                                    input_length=self.max_sequence_length,
                                    name="embedding")(text_input)
        lstm_layer = LSTM(self.lstm_units, name="lstm")(embedding_layer)
        
        # Output 1: Text Embedding (context vector)
        text_embedding_output = Dense(self.output_size, activation='tanh', name="text_embedding")(lstm_layer)

        # Output 2: Visual Label Prediction (from text embedding)
        visual_pred_hidden = Dense(32, activation='relu', name="visual_pred_hidden")(text_embedding_output) 
        visual_label_output = Dense(self.visual_output_size, activation='softmax', name="visual_label_pred")(visual_pred_hidden)

        model = Model(inputs=text_input, outputs=[text_embedding_output, visual_label_output], name="temporal_lobe_model")
        
        optimizer = LegacyAdam(learning_rate=self.learning_rate_learn)
        model.compile(optimizer=optimizer,
                      loss={'visual_label_pred': 'sparse_categorical_crossentropy', 
                            'text_embedding': None}, # No direct loss on embedding for now
                      metrics={'visual_label_pred': 'accuracy'})
        # model.summary() # Uncomment to print summary when an instance is created
        return model

    def _fit_tokenizer_if_needed(self, texts):
        if not self.tokenizer.word_index: # Check if tokenizer has been fit (word_index is empty)
            print("Temporal Lobe: Fitting tokenizer for the first time on provided texts...")
            self.tokenizer.fit_on_texts(texts)
            # Optional: Save tokenizer immediately after fitting, though save_model handles this too.
            # self._save_tokenizer() 
            print("Temporal Lobe: Tokenizer fitting complete.")

    def _preprocess_text(self, text_input_list):
        if not text_input_list or not isinstance(text_input_list, list) or not all(isinstance(t, str) for t in text_input_list):
            print("Temporal Lobe: Error - _preprocess_text expects a list of strings.")
            # Return a dummy sequence of the correct shape if input is invalid.
            # The first dimension is the batch size.
            if isinstance(text_input_list, list):
                first_dim = len(text_input_list)
            else:
                first_dim = 1 # Default batch size for non-list input
            return np.zeros((first_dim, self.max_sequence_length))

        self._fit_tokenizer_if_needed(text_input_list) # Ensure tokenizer is fit
        
        sequences = self.tokenizer.texts_to_sequences(text_input_list)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_sequence_length, padding='post', truncating='post')
        return padded_sequences

    def process_task(self, input_data_item, predict_visual=False):
        # Determine the actual text string from various possible input formats
        if isinstance(input_data_item, list) and input_data_item:
            actual_input_data = input_data_item[-1] # Assume last item if list
        else:
            actual_input_data = input_data_item

        if not isinstance(actual_input_data, str) or not actual_input_data.strip():
            print(f"Temporal Lobe: Invalid or empty text input for process_task ('{actual_input_data}'). Using default.")
            actual_input_data = "default empty text" # Default text

        print(f"Temporal Lobe: Processing task with text: '{actual_input_data}'")
        processed_sequence = self._preprocess_text([actual_input_data]) # Expects a list
        
        try:
            embedding_output, visual_prediction_scores = self.model.predict(processed_sequence, verbose=0)
        except Exception as e:
            print(f"Temporal Lobe: Error during model prediction: {e}")
            # Return default values matching the expected output structure
            default_embedding = np.zeros(self.output_size).tolist()
            if predict_visual:
                return (default_embedding, -1) # -1 for error in visual prediction
            else:
                return default_embedding

        embedding_vector = embedding_output[0] # predict returns a batch, get the first item

        if predict_visual:
            predicted_visual_label = np.argmax(visual_prediction_scores[0])
            print(f"Temporal Lobe: Text embedding: {embedding_vector.tolist()}, Predicted visual label: {int(predicted_visual_label)}")
            return (embedding_vector.tolist(), int(predicted_visual_label))
        else:
            print(f"Temporal Lobe: Text embedding: {embedding_vector.tolist()}")
            return embedding_vector.tolist()

    def learn(self, sequence, visual_label_as_context=None):
        texts_for_tokenizer_fitting = []
        if isinstance(sequence, list) and sequence:
            for item in sequence:
                if isinstance(item, tuple) and len(item) == 2:
                    texts_for_tokenizer_fitting.append(str(item[0])) # Input text
                    texts_for_tokenizer_fitting.append(str(item[1])) # Target text (if used for embedding learning)
                elif isinstance(item, str): # If sequence is just a list of texts
                     texts_for_tokenizer_fitting.append(item)


        # --- Visual Association Learning (Primary Training Path) ---
        if visual_label_as_context is not None:
            text_input_str_for_assoc = None
            if isinstance(sequence, list) and sequence and isinstance(sequence[0], tuple) and len(sequence[0]) > 0 and isinstance(sequence[0][0], str):
                text_input_str_for_assoc = sequence[0][0]
            elif isinstance(sequence, str): # If sequence itself is the input string
                 text_input_str_for_assoc = sequence
            
            if text_input_str_for_assoc:
                if not texts_for_tokenizer_fitting: # If not already populated
                    texts_for_tokenizer_fitting.append(text_input_str_for_assoc)
                
                self._fit_tokenizer_if_needed(list(set(texts_for_tokenizer_fitting))) # Fit with unique texts

                print(f"Temporal Lobe: Learning visual association for text '{text_input_str_for_assoc}' with label {visual_label_as_context}")
                input_sequence_for_assoc = self._preprocess_text([text_input_str_for_assoc])
                
                try:
                    target_visual_label_val = int(visual_label_as_context)
                except ValueError:
                    print(f"Temporal Lobe: Invalid visual label '{visual_label_as_context}'. Must be an integer. Skipping.")
                    return

                if not (0 <= target_visual_label_val < self.visual_output_size):
                    print(f"Temporal Lobe: Visual label {target_visual_label_val} out of range (0-{self.visual_output_size-1}). Skipping.")
                    return
                
                target_visual_label_array = np.array([target_visual_label_val])

                try:
                    history = self.model.fit(
                        input_sequence_for_assoc,
                        {'visual_label_pred': target_visual_label_array}, # Training only the visual prediction output
                        epochs=1,
                        verbose=0
                    )
                    loss = history.history.get('visual_label_pred_loss', [float('nan')])[0]
                    accuracy = history.history.get('visual_label_pred_accuracy', [float('nan')])[0] # Key might vary
                    print(f"Temporal Lobe: Visual association training complete. Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
                    
                    # Store this experience
                    self.cross_modal_memory.append((text_input_str_for_assoc, visual_label_as_context))
                    if len(self.cross_modal_memory) > self.max_cross_modal_memory_size:
                        self.cross_modal_memory.pop(0)
                except Exception as e:
                    print(f"Temporal Lobe: Error during visual association training: {e}")
            else:
                print("Temporal Lobe: No valid text input found in sequence for visual association learning.")
        
        elif texts_for_tokenizer_fitting: # If only text sequence provided (no visual label), just fit tokenizer
             self._fit_tokenizer_if_needed(list(set(texts_for_tokenizer_fitting)))
             print("Temporal Lobe: Tokenizer fitted with provided text sequence. No visual label for training.")


    def _save_tokenizer(self):
        print("Temporal Lobe: Saving tokenizer...")
        os.makedirs(os.path.dirname(self.tokenizer_path), exist_ok=True)
        try:
            tokenizer_json = self.tokenizer.to_json()
            with open(self.tokenizer_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(tokenizer_json, ensure_ascii=False))
            print("Temporal Lobe: Tokenizer saved.")
        except Exception as e:
            print(f"Temporal Lobe: Error saving tokenizer: {e}")

    def _load_tokenizer(self):
        if os.path.exists(self.tokenizer_path):
            print("Temporal Lobe: Loading tokenizer...")
            try:
                with open(self.tokenizer_path, 'r', encoding='utf-8') as f:
                    tokenizer_json_str = f.read() # Read the whole file
                    # Use the directly imported tokenizer_from_json
                    self.tokenizer = tokenizer_from_json(tokenizer_json_str)
                print("Temporal Lobe: Tokenizer loaded successfully.")
            except Exception as e:
                print(f"Temporal Lobe: Error loading tokenizer: {e}. Using new tokenizer.")
                self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<unk>")
        else:
            print("Temporal Lobe: No saved tokenizer found. Using new tokenizer.")
            self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<unk>")


    def save_model(self):
        print(f"Temporal Lobe: Saving model weights to {self.model_path}...")
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        try:
            self.model.save_weights(self.model_path)
            self._save_tokenizer() # Save tokenizer along with model weights
            self._save_memory() # Save cross-modal memory
            print("Temporal Lobe: Model weights, tokenizer, and memory saved.")
        except Exception as e:
            print(f"Temporal Lobe: Error saving model/tokenizer/memory: {e}")

    def load_model(self):
        # Tokenizer must be loaded before model if vocab_size in _build_model depends on it.
        # Here, vocab_size is fixed, but it's good practice.
        self._load_tokenizer() 
        if os.path.exists(self.model_path):
            print(f"Temporal Lobe: Loading model weights from {self.model_path}...")
            try:
                # Ensure model is built before loading weights (already done in __init__)
                self.model.load_weights(self.model_path)
                print("Temporal Lobe: Model weights loaded successfully.")
            except Exception as e:
                print(f"Temporal Lobe: Error loading model weights: {e}. Model remains initialized.")
        else:
            print(f"Temporal Lobe: No pre-trained weights found at {self.model_path}. Model is newly initialized.")
        self._load_memory() # Load cross-modal memory

    def _save_memory(self):
        print("Temporal Lobe: Saving cross-modal memory...")
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        try:
            with open(self.memory_path, "w") as f:
                json.dump({"cross_modal_memory": self.cross_modal_memory}, f)
            print("Temporal Lobe: Cross-modal memory saved.")
        except Exception as e:
            print(f"Temporal Lobe: Error saving cross-modal memory: {e}")
            
    def _load_memory(self):
        if os.path.exists(self.memory_path):
            print("Temporal Lobe: Loading cross-modal memory...")
            try:
                with open(self.memory_path, "r") as f:
                    loaded_data = json.load(f)
                    self.cross_modal_memory = loaded_data.get("cross_modal_memory", [])
                # Ensure loaded memory is a list of tuples
                self.cross_modal_memory = [tuple(item) for item in self.cross_modal_memory if isinstance(item, list) and len(item) == 2]
                print("Temporal Lobe: Cross-modal memory loaded.")
            except Exception as e:
                print(f"Temporal Lobe: Error loading cross-modal memory: {e}. Initializing empty memory.")
                self.cross_modal_memory = []
        else:
            print("Temporal Lobe: No cross-modal memory file found. Initializing empty memory.")
            self.cross_modal_memory = []


    def consolidate(self):
        print("Temporal Lobe: Starting consolidation (replaying cross-modal memory)...")
        if not self.cross_modal_memory:
            print("Temporal Lobe: No experiences in cross-modal memory to consolidate.")
        else:
            # Replay from cross_modal_memory
            num_replays = min(len(self.cross_modal_memory), 32) # Replay up to 32 samples or all if fewer
            print(f"Temporal Lobe: Replaying {num_replays} experiences for consolidation.")
            
            # Create batches for more efficient training if many samples
            texts_to_train = [item[0] for item in self.cross_modal_memory[:num_replays]]
            labels_to_train = np.array([int(item[1]) for item in self.cross_modal_memory[:num_replays]])
            
            input_sequences = self._preprocess_text(texts_to_train)

            if input_sequences.shape[0] > 0 : # Check if any valid sequences produced
                try:
                    self.model.fit(
                        input_sequences,
                        {'visual_label_pred': labels_to_train},
                        epochs=1, # Could be more for consolidation
                        verbose=0,
                        batch_size=min(input_sequences.shape[0], 16) # Smaller batch size for consolidation
                    )
                    print("Temporal Lobe: Consolidation replay training complete.")
                except Exception as e:
                    print(f"Temporal Lobe: Error during consolidation replay training: {e}")
            else:
                print("Temporal Lobe: No valid sequences to train on during consolidation after preprocessing.")

        self.save_model() # Saves weights, tokenizer, and memory
        print("Temporal Lobe: Consolidation complete.")


# Example Usage
if __name__ == '__main__':
    print("\n--- Testing TemporalLobeAI (RNN/LSTM) ---")
    test_model_path = "data/test_temporal_rnn.weights.h5"
    test_tokenizer_path = "data/test_temporal_rnn_tokenizer.json"
    test_memory_path = "data/test_temporal_rnn_memory.json"

    # Clean up old test files
    for f_path in [test_model_path, test_tokenizer_path, test_memory_path]:
        if os.path.exists(f_path):
            os.remove(f_path)

    temporal_ai = TemporalLobeAI(
        model_path=test_model_path,
        tokenizer_path=test_tokenizer_path,
        memory_path=test_memory_path
    )
    temporal_ai.model.summary()

    print("\n--- Testing Tokenizer Fitting (Implicit via learn/process) ---")
    sample_texts_for_fitting = [
        "hello world", "this is a test", "another sample text", "world of AI"
    ]
    temporal_ai._fit_tokenizer_if_needed(sample_texts_for_fitting)
    if temporal_ai.tokenizer.word_index:
        print(f"Tokenizer fitted. Vocab size (actual): {len(temporal_ai.tokenizer.word_index)}")
        print(f"Word for 'world': {temporal_ai.tokenizer.word_index.get('world')}")
    else:
        print("Tokenizer fitting failed or no texts provided.")

    print("\n--- Testing Text Preprocessing ---")
    test_text = ["hello world of AI"]
    preprocessed = temporal_ai._preprocess_text(test_text)
    print(f"Original: '{test_text[0]}', Preprocessed: {preprocessed}")
    
    print("\n--- Testing process_task (Embedding only) ---")
    embedding = temporal_ai.process_task(test_text[0], predict_visual=False)
    print(f"Embedding for '{test_text[0]}': shape {np.array(embedding).shape}")

    print("\n--- Testing process_task (Embedding + Visual Prediction) ---")
    embedding_pv, visual_pred_pv = temporal_ai.process_task(test_text[0], predict_visual=True)
    print(f"For '{test_text[0]}': Embedding shape {np.array(embedding_pv).shape}, Visual Pred: {visual_pred_pv}")

    print("\n--- Testing learn (Visual Association) ---")
    # Example sequence for learn: list of (input_text, target_text_for_embedding)
    # For now, we simplify learn to focus on visual association using the first text.
    learn_text_sequence = [("the quick brown fox", "the quick brown fox")] # Dummy target text for embedding
    visual_label_for_learn = 3 
    temporal_ai.learn(learn_text_sequence, visual_label_as_context=visual_label_for_learn)
    
    learn_text_sequence_2 = [("a blue sphere", "a blue sphere")]
    visual_label_for_learn_2 = 1
    temporal_ai.learn(learn_text_sequence_2, visual_label_as_context=visual_label_for_learn_2)
    
    print(f"Cross-modal memory size after learning: {len(temporal_ai.cross_modal_memory)}")

    print("\n--- Testing Consolidation ---")
    temporal_ai.consolidate()

    print("\n--- Testing Save/Load ---")
    temporal_ai.save_model() # Should save weights, tokenizer, memory
    
    # Create new instance to test loading
    print("\nCreating new TemporalLobeAI instance for loading test...")
    temporal_ai_loaded = TemporalLobeAI(
        model_path=test_model_path,
        tokenizer_path=test_tokenizer_path,
        memory_path=test_memory_path
    )
    
    if temporal_ai_loaded.tokenizer.word_index and \
       temporal_ai_loaded.tokenizer.word_index.get('world') == temporal_ai.tokenizer.word_index.get('world'):
        print("Tokenizer loaded successfully.")
    else:
        print("Tokenizer loading failed or mismatch.")
        
    if len(temporal_ai_loaded.cross_modal_memory) == len(temporal_ai.cross_modal_memory):
        print(f"Cross-modal memory loaded successfully with {len(temporal_ai_loaded.cross_modal_memory)} items.")
    else:
        print("Cross-modal memory loading failed or mismatch.")

    # Test if model weights were loaded (simple check on one layer's weights)
    original_weights_lstm = temporal_ai.model.get_layer("lstm").get_weights()[0]
    loaded_weights_lstm = temporal_ai_loaded.model.get_layer("lstm").get_weights()[0]
    if np.array_equal(original_weights_lstm, loaded_weights_lstm):
        print("Model weights loaded successfully (LSTM layer sample weights match).")
    else:
        print("Model weights loading failed or mismatch.")

    # Clean up
    for f_path in [test_model_path, test_tokenizer_path, test_memory_path]:
        if os.path.exists(f_path):
            os.remove(f_path)
    print("\nCleaned up test files.")
    
    print("\nTemporal Lobe AI (RNN/LSTM) test script finished.")
