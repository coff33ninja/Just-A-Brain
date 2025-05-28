# frontal.py (Decision-Making, Planning using DQN)
import numpy as np
import json
import os
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Input, LSTM # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import random
from collections import deque  # For replay buffer


class FrontalLobeAI:
    def __init__(
        self, input_size=18, output_size=5, model_path="data/frontal_model.weights.h5", replay_batch_size=32
    ):
        # input_size now refers to the number of features in each step of a sequence
        self.feature_size_per_step = input_size
        self.output_size = output_size  # Action size
        self.sequence_length = 5  # Number of recent states to consider as working memory

        # LSTM layer units
        self.lstm_units = 64

        # --- DQN (Deep Q-Network) Parameters ---
        # DQN Parameters
        self.learning_rate_dqn = 0.001
        self.discount_factor_gamma = 0.95
        self.replay_buffer_size = 10000
        self.replay_batch_size = replay_batch_size
        self.target_update_frequency = 100  # Steps
        self.learn_step_counter = 0

        # --- Exploration Parameters ---
        self.exploration_rate_epsilon = 1.0
        self.epsilon_decay_rate = (
            0.001  # Make this smaller for longer exploration, e.g., 0.0001 or 0.00001
        )
        self.min_epsilon = 0.01

        # --- Memory Components ---
        # Short-Term Memory (STM): Replay buffer for recent experiences
        # This will now store (state_sequence, action, reward, next_state_sequence, done)
        self.working_memory_buffer = deque(maxlen=self.sequence_length) # Stores recent feature_vectors for current decision making

        self.memory_stm = deque(maxlen=self.replay_buffer_size)

        # Long-Term Episodic Memory (LTM): Stores significant past experiences (e.g., episode ends)
        self.ltm_size = 50000 # Configurable maximum size for LTM
        self.long_term_episodic_memory = deque(maxlen=self.ltm_size)

        # --- File Paths for Persistence ---
        self.model_path = model_path
        base_path = self.model_path.replace(".weights.h5", "")
        self.epsilon_path = base_path + "_epsilon.json"
        self.ltm_path = base_path + "_ltm.json" # Path for Long-Term Episodic Memory

        # --- Model Initialization ---
        # Main Q-network (learned LTM policy) and Target Q-network
        self.model = self._build_model()  # Main Q-network
        self.target_model = self._build_model()  # Target Q-network
        self.update_target_model()  # Initialize target model weights to match main model

        self.load_model()  # Load weights and epsilon if they exist

    def _build_model(self):
        model = Sequential(
            [
                Input(shape=(self.sequence_length, self.feature_size_per_step), name="input_layer"),
                LSTM(self.lstm_units, name="lstm_layer"), # Input shape is (batch_size, timesteps, features)
                Dense(
                    32, activation="relu", name="dense_hidden_1"
                ),
                Dense(self.output_size, activation="linear", name="dense_output"),  # Q-values
            ]
        )
        optimizer = Adam(learning_rate=self.learning_rate_dqn)
        model.compile(
            optimizer=optimizer, loss="mse")  # Mean Squared Error for Q-learning
        return model

    def _prepare_state_vector(self, state_data):
        """
        Prepares raw state data into a flat numpy vector of self.feature_size_per_step.
        This vector represents one time step in a sequence.
        """
        try:
            state_vector_1d = np.array(state_data, dtype=float).flatten()
            current_len = state_vector_1d.shape[0]

            if current_len == self.feature_size_per_step:
                return state_vector_1d
            elif current_len < self.feature_size_per_step:
                padding = np.zeros(self.feature_size_per_step - current_len)
                state_vector_1d = np.concatenate((state_vector_1d, padding))
            else:
                state_vector_1d = state_vector_1d[: self.feature_size_per_step]
            return state_vector_1d
        except Exception as e:
            print(f"Error preparing state vector from data '{state_data}': {e}. Using zero vector.")
            return np.zeros(self.feature_size_per_step)

    def remember(self, state_sequence, action, reward, next_state_sequence, done):
        """Stores an experience in STM (replay buffer) and potentially LTM (episodic memory)."""
        # state_sequence and next_state_sequence are lists of feature vectors (each of size self.feature_size_per_step)
        experience = (list(state_sequence), action, reward, list(next_state_sequence), done)
        self.memory_stm.append(experience)

        if done: # If the episode ended, store this significant experience in LTM
            self.long_term_episodic_memory.append(experience)

    def process_task(
        self, current_raw_input_data
    ):
        """
        Processes the current raw input, updates working memory, and chooses an action.
        Returns the chosen action and the state sequence used for the decision.
        """
        current_feature_vector = self._prepare_state_vector(current_raw_input_data)
        self.working_memory_buffer.append(current_feature_vector)

        # Prepare the sequence for the model
        current_sequence_list = list(self.working_memory_buffer)
        if len(current_sequence_list) < self.sequence_length:
            padding_needed = self.sequence_length - len(current_sequence_list)
            padding = [np.zeros(self.feature_size_per_step) for _ in range(padding_needed)]
            padded_sequence_list = padding + current_sequence_list
        else:
            padded_sequence_list = current_sequence_list

        sequence_for_model_np = np.array(padded_sequence_list) # Shape: (sequence_length, feature_size_per_step)
        # Reshape for Keras model prediction (expects batch dimension: 1 sample, sequence_length timesteps, feature_size_per_step features)
        state_batch = np.reshape(sequence_for_model_np, [1, self.sequence_length, self.feature_size_per_step])

        if np.random.rand() <= self.exploration_rate_epsilon:
            action = np.random.randint(0, self.output_size)  # Explore
        else:
            q_values = self.model.predict(state_batch, verbose=0)[0]  # Exploit
            action = np.argmax(q_values)

        return int(action), padded_sequence_list # Return action and the sequence used

    def learn(self, state_sequence_t, action_t, reward_t, next_state_sequence_t_plus_1, done):
        """
        Stores the experience and triggers replay if buffer is large enough.
        """
        self.remember(state_sequence_t, action_t, reward_t, next_state_sequence_t_plus_1, done)

        if (
            len(self.memory_stm) >= self.replay_batch_size
        ):  # Start replay only when enough samples
            self.replay(memory_source=self.memory_stm) # Replay from STM

    def _calculate_sequence_similarity(self, seq1_list, seq2_list):
        """Calculates cosine similarity between two state sequences (lists of feature vectors)."""
        # Flatten sequences and compute cosine similarity
        vec1 = np.array(seq1_list).flatten()
        vec2 = np.array(seq2_list).flatten()
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0 # Avoid division by zero
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return similarity

    def _find_similar_ltm_experiences(self, query_sequence_list, k=3):
        """Finds k most similar experiences from LTM based on state sequence similarity."""
        if not self.long_term_episodic_memory:
            return []

        similarities = []
        for ltm_exp in self.long_term_episodic_memory:
            ltm_state_sequence = ltm_exp[0] # state_sequence is the first element
            sim = self._calculate_sequence_similarity(query_sequence_list, ltm_state_sequence)
            similarities.append((sim, ltm_exp))

        similarities.sort(key=lambda x: x[0], reverse=True)
        return [exp for sim, exp in similarities[:k]]

    def replay(self, memory_source, num_samples=None):
        """Trains the DQN by replaying experiences from the given memory_source."""
        batch_size_to_sample = num_samples if num_samples is not None else self.replay_batch_size
        if len(memory_source) < batch_size_to_sample:
            return  # Not enough memory to sample a batch

        minibatch = random.sample(list(memory_source), batch_size_to_sample)  # nosec B311

        # Active LTM Retrieval: If replaying from STM, augment with similar LTM experiences
        if memory_source == self.memory_stm and len(self.long_term_episodic_memory) > 0:
            num_ltm_to_add = batch_size_to_sample // 4  # e.g., add 25% LTM experiences
            if num_ltm_to_add > 0 and minibatch:
                # Use the state sequence from a random experience in the STM minibatch as a query
                query_stm_experience = random.choice(minibatch)  # nosec B311
                query_sequence = query_stm_experience[0] # state_sequence_t

                similar_ltm_exps = self._find_similar_ltm_experiences(query_sequence, k=num_ltm_to_add)

                # Replace some STM samples with LTM samples to keep batch size roughly constant
                if similar_ltm_exps:
                    num_to_replace = min(len(similar_ltm_exps), len(minibatch))
                    # Ensure we don't try to sample more than available if minibatch is small
                    if num_to_replace > 0 :
                        minibatch = random.sample(minibatch, len(minibatch) - num_to_replace) + similar_ltm_exps[:num_to_replace]
                        random.shuffle(minibatch) # Shuffle again after adding LTM experiences

        # Experiences are (state_sequence, action, reward, next_state_sequence, done)
        # state_sequences are already lists of prepared feature vectors.
        current_state_sequences = np.array([experience[0] for experience in minibatch]) # Shape: (batch, seq_len, features)
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_state_sequences = np.array([experience[3] for experience in minibatch]) # Shape: (batch, seq_len, features)
        dones = np.array([experience[4] for experience in minibatch])

        # 3. Make one call to self.model.predict() using current_states_prepared_np
        current_q_values_batch = self.model.predict(current_state_sequences, verbose=0)

        # 4. Make one call to self.target_model.predict() using next_states_prepared_np
        next_q_values_from_target_model_batch = self.target_model.predict(next_state_sequences, verbose=0)

        # 5. Calculate the targets for the entire batch
        targets_batch = np.copy(current_q_values_batch) # Initialize targets as current Q-values

        # Calculate target values: reward + gamma * max_next_q for non-done states
        # For done states, target is just the reward
        for i in range(batch_size_to_sample):
            if dones[i]:
                targets_batch[i, actions[i]] = rewards[i]
            else:
                max_next_q = np.max(next_q_values_from_target_model_batch[i]) # Changed from amax
                targets_batch[i, actions[i]] = rewards[i] + self.discount_factor_gamma * max_next_q

        # 6. Finally, call self.model.fit() once with current_states_prepared_np and the calculated batch of targets
        try:
            self.model.fit(
                current_state_sequences,
                targets_batch,
                epochs=1,
                verbose=0,
            )
        except Exception as e:
            print(f"Frontal Lobe: Error during batch model training in replay: {e}")
            return # Exit replay if training fails

        if memory_source == self.memory_stm: # Only update counters and decay for STM replay
            self.learn_step_counter += 1
            if self.learn_step_counter % self.target_update_frequency == 0:
                self.update_target_model()

        # Decay epsilon after a learning step (replay)
        if self.exploration_rate_epsilon > self.min_epsilon:
            self.exploration_rate_epsilon -= self.epsilon_decay_rate
            self.exploration_rate_epsilon = max(self.min_epsilon, self.exploration_rate_epsilon)


    def update_target_model(self):
        """Copies weights from the main model to the target model."""
        self.target_model.set_weights(self.model.get_weights())
        print("Frontal Lobe: Target network updated.")

    def _save_ltm(self):
        """Saves the Long-Term Episodic Memory to a JSON file."""
        print(f"Frontal Lobe: Saving Long-Term Episodic Memory to {self.ltm_path}...")
        try:
            # Convert numpy arrays in experiences to lists for JSON serialization
            # Experiences are (state_sequence, action, reward, next_state_sequence, done)
            # state_sequence and next_state_sequence are lists of feature vectors (which might be numpy arrays)
            serializable_ltm = []
            for s_seq, action, reward, ns_seq, done in list(self.long_term_episodic_memory):
                s_seq_list = [s.tolist() if isinstance(s, np.ndarray) else list(s) for s in s_seq]
                ns_seq_list = [ns.tolist() if isinstance(ns, np.ndarray) else list(ns) for ns in ns_seq]
                serializable_ltm.append((s_seq_list, int(action), float(reward), ns_seq_list, bool(done)))

            with open(self.ltm_path, "w") as f:
                json.dump({"long_term_episodic_memory": serializable_ltm}, f)
            print(f"Frontal Lobe: Long-Term Episodic Memory saved. Size: {len(serializable_ltm)}")
        except Exception as e:
            print(f"Frontal Lobe: Error saving Long-Term Episodic Memory: {e}")

    def save_model(self):
        print(
            f"Saving Frontal Lobe model to {self.model_path} and epsilon to {self.epsilon_path}"
        )
        # This method now saves:
        # 1. Model weights (LTM - learned policy)
        # 2. Epsilon state (exploration parameter)
        # 3. Long-Term Episodic Memory (LTM - significant experiences)
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        try:
            if not self.model_path.endswith(".weights.h5"):
                print(f"Warning: FrontalLobeAI model_path '{self.model_path}' does not end with '.weights.h5'. Keras save_weights might prefer this.")
            # Ensure the directory for model_path exists
            self.model.save_weights(self.model_path) # Keras saves in HDF5 format
            epsilon_data = {"exploration_rate_epsilon": self.exploration_rate_epsilon}
            with open(self.epsilon_path, "w") as f:
                json.dump(epsilon_data, f)

            self._save_ltm() # Save the Long-Term Episodic Memory
            print("Frontal Lobe: Model weights, epsilon, and LTM saved.")
        except Exception as e:
            print(f"Error saving Frontal Lobe model/epsilon: {e}")

    def _load_ltm(self):
        """Loads the Long-Term Episodic Memory from a JSON file."""
        if os.path.exists(self.ltm_path):
            print(f"Frontal Lobe: Loading Long-Term Episodic Memory from {self.ltm_path}...")
            try:
                with open(self.ltm_path, "r") as f:
                    data = json.load(f)
                    loaded_ltm_serializable = data.get("long_term_episodic_memory", [])
                    # Convert inner lists back to numpy arrays if they were stored as lists of numbers
                    loaded_ltm_raw = []
                    for s_seq_list, action, reward, ns_seq_list, done in loaded_ltm_serializable:
                        s_seq_np = [np.array(s, dtype=float) for s in s_seq_list]
                        ns_seq_np = [np.array(ns, dtype=float) for ns in ns_seq_list]
                        loaded_ltm_raw.append((s_seq_np, action, reward, ns_seq_np, done))

                    self.long_term_episodic_memory = deque(loaded_ltm_raw, maxlen=self.ltm_size)
                print(f"Frontal Lobe: Long-Term Episodic Memory loaded. Size: {len(self.long_term_episodic_memory)}")
            except Exception as e:
                print(f"Frontal Lobe: Error loading Long-Term Episodic Memory: {e}. Initializing empty LTM.")
                self.long_term_episodic_memory = deque(maxlen=self.ltm_size)

    def load_model(self):
        if os.path.exists(self.model_path):
            print(f"Loading Frontal Lobe model weights from {self.model_path}")
            try:
                # Ensure the model is built before loading weights
                if self.model is None: # Should not happen if __init__ is correct
                    self.model = self._build_model()
                self.model.load_weights(self.model_path) # Keras loads from HDF5 format
                # Important: After loading weights into main model, also update target model
                self.update_target_model()
                print("Frontal Lobe model weights loaded and target model updated.")
            except Exception as e:
                print(
                    f"Error loading Frontal Lobe model weights: {e}. Model remains initialized."
                )
        else:
            # This case means the .weights.h5 file was not found.
            # If model_path was originally .json and we are trying to load it as .weights.h5,
            # this path will be hit. The model remains newly initialized.
            print(
                f"No pre-trained weights found for Frontal Lobe at {self.model_path} (expected .weights.h5). Model is newly initialized."
            )

        if os.path.exists(self.epsilon_path):
            print(f"Loading Frontal Lobe epsilon from {self.epsilon_path}")
            try:
                with open(self.epsilon_path, "r") as f:
                    epsilon_data = json.load(f)
                    self.exploration_rate_epsilon = epsilon_data.get(
                        "exploration_rate_epsilon", 1.0
                    )
                print(
                    f"Frontal Lobe epsilon loaded: {self.exploration_rate_epsilon:.3f}"
                )
            except Exception as e:
                print(
                    f"Error loading Frontal Lobe epsilon: {e}. Using default epsilon."
                )
        else:
            print(
                f"No epsilon file found at {self.epsilon_path}. Using default epsilon."
            )

        self._load_ltm() # Load the Long-Term Episodic Memory

    def consolidate(self):
        """Bedtime: Perform more replay steps and save the model."""
        print("Frontal Lobe: Starting consolidation...")

        consolidation_replay_batches = 10  # Example: number of replay batches for consolidation

        # Consolidate from Short-Term Memory (STM - replay buffer)
        if len(self.memory_stm) >= self.replay_batch_size:
            print(
                f"Frontal Lobe: Performing {consolidation_replay_batches} replay batches from STM for consolidation."
            )
            for _ in range(consolidation_replay_batches):
                self.replay(memory_source=self.memory_stm)
        else:
            print(
                "Frontal Lobe: Not enough experiences in STM to consolidate with extensive replay."
            )

        # Consolidate from Long-Term Episodic Memory (LTM)
        if len(self.long_term_episodic_memory) >= self.replay_batch_size:
            print(
                f"Frontal Lobe: Performing {consolidation_replay_batches} replay batches from LTM for consolidation."
            )
            for _ in range(consolidation_replay_batches):
                # Sample from LTM; note that LTM replay doesn't affect epsilon decay or target net update counter here
                self.replay(memory_source=self.long_term_episodic_memory)
        else:
            print(
                "Frontal Lobe: Not enough experiences in LTM to consolidate with extensive replay."
            )

        self.update_target_model()  # Ensure target model is up-to-date
        self.save_model()
        print(
            "Frontal Lobe: Consolidation complete (replayed experiences and saved model)."
        )

# Example Usage
if __name__ == "__main__":
    print("\n--- Testing FrontalLobeAI (DQN) ---")
    # Use temporary paths for testing
    test_model_path = "data/test_frontal_dqn.weights.h5"

    # Clean up old test files if they exist
    if os.path.exists(test_model_path):
        os.remove(test_model_path)
    if os.path.exists(test_model_path + "_epsilon.json"):
        os.remove(test_model_path + "_epsilon.json")
    if os.path.exists(test_model_path.replace(".weights.h5", "") + "_ltm.json"):
        os.remove(test_model_path.replace(".weights.h5", "") + "_ltm.json")

    frontal_ai = FrontalLobeAI(model_path=test_model_path)
    print(f"Frontal AI initialized. Epsilon: {frontal_ai.exploration_rate_epsilon:.3f}")
    frontal_ai.model.summary()

    # Simulate some learning steps
    num_test_steps = frontal_ai.replay_batch_size * 2  # Ensure enough for a few replays
    print(f"\n--- Simulating {num_test_steps} learning steps ---")
    for i in range(num_test_steps):
        # Example state: concatenation of vision (5), parietal (3), temporal (10) outputs
        # For testing, use random data that matches input_size
        state_raw = np.random.rand(frontal_ai.input_size).tolist()

        action = frontal_ai.process_task(state_raw)  # This also decays epsilon

        reward = np.random.choice([-1, 0, 1])
        next_state_raw = np.random.rand(frontal_ai.input_size).tolist()
        done = i % 10 == 9  # Episode ends every 10 steps

        frontal_ai.learn(state_raw, action, reward, next_state_raw, done)

        if (i + 1) % (frontal_ai.replay_batch_size // 2) == 0:  # Log progress
            print(
                f"Step {i+1}: Epsilon: {frontal_ai.exploration_rate_epsilon:.3f}, STM size: {len(frontal_ai.memory_stm)}, LTM size: {len(frontal_ai.long_term_episodic_memory)}"
            )

    print(
        f"\n--- Finished simulation. Final Epsilon: {frontal_ai.exploration_rate_epsilon:.3f} ---"
    )

    print("\n--- Testing Consolidation ---")
    frontal_ai.consolidate()

    print("\n--- Testing Save/Load ---")
    # Save current state (if consolidate didn't already)
    frontal_ai.save_model()

    # Store current epsilon for comparison after load
    epsilon_before_load = frontal_ai.exploration_rate_epsilon
    ltm_size_before_load = len(frontal_ai.long_term_episodic_memory)

    # Create a new instance and load
    print("\nCreating new FrontalLobeAI instance for loading test...")
    frontal_ai_loaded = FrontalLobeAI(model_path=test_model_path)

    # Check if epsilon was loaded correctly
    print(
        f"Epsilon before load: {epsilon_before_load:.3f}, Epsilon after load: {frontal_ai_loaded.exploration_rate_epsilon:.3f}"
    )
    # Due to floating point precision, exact match might be tricky if decay is very small.
    # For this test, it should be reasonably close or exact if decay step is large enough.
    if (
        abs(frontal_ai_loaded.exploration_rate_epsilon - epsilon_before_load) < 1e-5
    ):  # Allow small tolerance
        print("Epsilon loaded correctly.")
    else:
        print(
            "Epsilon loading discrepancy. Check save/load logic or decay interaction."
        )

    # Check if LTM was loaded correctly
    print(
        f"LTM size before load: {ltm_size_before_load}, LTM size after load: {len(frontal_ai_loaded.long_term_episodic_memory)}"
    )
    if len(frontal_ai_loaded.long_term_episodic_memory) == ltm_size_before_load:
        print("LTM loaded correctly (size matches).")
    else:
        print("LTM loading discrepancy (size mismatch).")
    # A simple check: compare a bias term from the output layer of the main model
    if hasattr(frontal_ai.model.get_layer("dense_output"), "bias") and hasattr(
        frontal_ai_loaded.model.get_layer("dense_output"), "bias"
    ):
        original_bias = frontal_ai.model.get_layer("dense_output").bias.numpy()
        loaded_bias = frontal_ai_loaded.model.get_layer("dense_output").bias.numpy()
        if np.array_equal(original_bias, loaded_bias):
            print("Model weights loaded successfully (output layer bias term matches).")
        else:
            print(
                "Model weights loaded, but output layer bias term does not match. Investigate."
            )
    else:
        print("Could not compare bias terms for layers.")

    # Clean up dummy model file
    if os.path.exists(test_model_path):
        os.remove(test_model_path)
    if os.path.exists(test_model_path + "_epsilon.json"):
        os.remove(test_model_path + "_epsilon.json")
    if os.path.exists(test_model_path.replace(".weights.h5", "") + "_ltm.json"):
        os.remove(test_model_path.replace(".weights.h5", "") + "_ltm.json")
    print("\nCleaned up test model files.")

    print("\nFrontal Lobe AI (DQN) test script finished.")
