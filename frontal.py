# frontal.py (Decision-Making, Planning using DQN)
import numpy as np
import json
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# from tensorflow.keras.optimizers import Adam # Standard Adam
from tensorflow.keras.optimizers.legacy import (
    Adam as LegacyAdam,
)  # For M1/M2 Mac compatibility if needed
import random
from collections import deque  # For replay buffer


class FrontalLobeAI:
    def __init__(
        self, input_size=18, output_size=5, model_path="data/frontal_model.weights.h5"
    ):
        self.input_size = input_size  # State size
        self.output_size = output_size  # Action size

        # DQN Parameters
        self.learning_rate_dqn = 0.001
        self.discount_factor_gamma = 0.95
        self.replay_buffer_size = 10000
        self.replay_batch_size = 32
        self.target_update_frequency = 100  # Steps
        self.learn_step_counter = 0

        # Exploration Parameters
        self.exploration_rate_epsilon = 1.0
        self.epsilon_decay_rate = (
            0.001  # Make this smaller for longer exploration, e.g., 0.0001 or 0.00001
        )
        self.min_epsilon = 0.01

        self.memory = deque(maxlen=self.replay_buffer_size)

        self.model_path = model_path
        self.epsilon_path = self.model_path + "_epsilon.json"  # Path for epsilon

        # Build and compile models
        self.model = self._build_model()  # Main Q-network
        self.target_model = self._build_model()  # Target Q-network
        self.update_target_model()  # Initialize target model weights to match main model

        self.load_model()  # Load weights and epsilon if they exist

    def _build_model(self):
        model = Sequential(
            [
                Dense(
                    32, input_dim=self.input_size, activation="relu", name="dense_input"
                ),
                Dense(32, activation="relu", name="dense_hidden_1"),
                Dense(
                    self.output_size, activation="linear", name="dense_output"
                ),  # Q-values for each action
            ]
        )
        # Using LegacyAdam for broader compatibility, especially on M1/M2 Macs.
        # For other systems, tf.keras.optimizers.Adam can be used.
        optimizer = LegacyAdam(learning_rate=self.learning_rate_dqn)
        model.compile(
            optimizer=optimizer, loss="mse"
        )  # Mean Squared Error for Q-learning
        return model

    def _prepare_state_vector(self, state_data):
        """
        Prepares raw state data into a flat numpy vector of the correct input size.
        """
        try:
            # Attempt to convert to numpy array and flatten
            state_vector_1d = np.array(state_data, dtype=float).flatten()

            current_len = state_vector_1d.shape[0]
            if current_len == self.input_size:
                return state_vector_1d
            elif current_len < self.input_size:
                # Pad with zeros if too short
                padding = np.zeros(self.input_size - current_len)
                state_vector_1d = np.concatenate((state_vector_1d, padding))
                # print(f"Warning: State data was shorter than input_size. Padded. Original len: {current_len}")
            else:
                # Truncate if too long
                state_vector_1d = state_vector_1d[: self.input_size]
                # print(f"Warning: State data was longer than input_size. Truncated. Original len: {current_len}")
            return state_vector_1d
        except Exception as e:  # Catch broader exceptions during conversion/shaping
            # print(f"Error preparing state vector from data '{state_data}': {e}. Using zero vector.")
            return np.zeros(self.input_size)

    def remember(self, state, action, reward, next_state, done):
        """Stores an experience in the replay buffer."""
        # State and next_state are stored in their raw/original format from the environment
        self.memory.append((state, action, reward, next_state, done))

    def process_task(
        self, current_input_data
    ):  # Renamed from input_data to avoid confusion
        """Choose an action using epsilon-greedy policy."""
        state_vector_1d = self._prepare_state_vector(current_input_data)
        # Reshape for Keras model prediction (expects batch dimension)
        state_batch = np.reshape(state_vector_1d, [1, self.input_size])

        if np.random.rand() <= self.exploration_rate_epsilon:
            action = np.random.randint(0, self.output_size)  # Explore
            # print(f"Frontal Lobe: Exploring - Chose action {action} randomly.")
        else:
            q_values = self.model.predict(state_batch, verbose=0)[0]  # Exploit
            action = np.argmax(q_values)
            # print(f"Frontal Lobe: Exploiting - Q-values: {q_values}, Chose action {action}.")

        # Epsilon decay is usually done after a learning step, not every action.
        # However, if 'learn' is not called frequently, this placement might be okay for slow decay.
        # Consider moving decay to after 'replay' if 'learn' is called often.
        self.exploration_rate_epsilon = max(
            self.min_epsilon, self.exploration_rate_epsilon - self.epsilon_decay_rate
        )
        return int(action)

    def learn(self, state, action, reward, next_state, done):
        """
        Stores the experience and triggers replay if buffer is large enough.
        This method is called by the main loop after an action is taken.
        """
        self.remember(state, action, reward, next_state, done)

        if (
            len(self.memory) >= self.replay_batch_size
        ):  # Start replay only when enough samples
            # print("Frontal Lobe: Replay buffer filled enough, starting replay.")
            self.replay()
        # else:
        # print(f"Frontal Lobe: Memory size {len(self.memory)} < batch size {self.replay_batch_size}. Not replaying yet.")

    def replay(self):
        """Trains the DQN by replaying experiences from the memory buffer."""
        if len(self.memory) < self.replay_batch_size:
            return  # Not enough memory to sample a batch

        minibatch = random.sample(self.memory, self.replay_batch_size)

        states_prepared = []
        q_values_updated = []

        for state_raw, action, reward, next_state_raw, done in minibatch:
            state_prepared = self._prepare_state_vector(state_raw)
            state_batch_for_predict = np.reshape(state_prepared, [1, self.input_size])

            target = reward
            if not done:
                next_state_prepared = self._prepare_state_vector(next_state_raw)
                next_state_batch_for_predict = np.reshape(
                    next_state_prepared, [1, self.input_size]
                )
                # Use target_model for predicting next Q-values (stabilizes learning)
                target = reward + self.discount_factor_gamma * np.amax(
                    self.target_model.predict(next_state_batch_for_predict, verbose=0)[
                        0
                    ]
                )

            # Get current Q-values for the original state from the main model
            current_q_values_for_state = self.model.predict(
                state_batch_for_predict, verbose=0
            )
            current_q_values_for_state[0][
                action
            ] = target  # Update Q-value for the action taken

            states_prepared.append(state_prepared)
            q_values_updated.append(
                current_q_values_for_state[0]
            )  # Append the full array of Q-values for this state

        # Train the main model on the batch of experiences
        # This is more efficient than training one by one in the loop
        try:
            self.model.fit(
                np.array(states_prepared),
                np.array(q_values_updated),
                epochs=1,
                verbose=0,
            )
        except Exception as e:
            print(f"Frontal Lobe: Error during batch model training in replay: {e}")
            return  # Exit replay if training fails

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_frequency == 0:
            self.update_target_model()

    def update_target_model(self):
        """Copies weights from the main model to the target model."""
        self.target_model.set_weights(self.model.get_weights())
        print("Frontal Lobe: Target network updated.")

    def save_model(self):
        print(
            f"Saving Frontal Lobe model to {self.model_path} and epsilon to {self.epsilon_path}"
        )
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        try:
            self.model.save_weights(self.model_path)
            epsilon_data = {"exploration_rate_epsilon": self.exploration_rate_epsilon}
            with open(self.epsilon_path, "w") as f:
                json.dump(epsilon_data, f)
            print("Frontal Lobe model and epsilon saved.")
        except Exception as e:
            print(f"Error saving Frontal Lobe model/epsilon: {e}")

    def load_model(self):
        if os.path.exists(self.model_path):
            print(f"Loading Frontal Lobe model weights from {self.model_path}")
            try:
                self.model.load_weights(self.model_path)
                # Important: After loading weights into main model, also update target model
                self.update_target_model()
                print("Frontal Lobe model weights loaded and target model updated.")
            except Exception as e:
                print(
                    f"Error loading Frontal Lobe model weights: {e}. Model remains initialized."
                )
        else:
            print(
                f"No pre-trained weights found for Frontal Lobe at {self.model_path}. Model is newly initialized."
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

    def consolidate(self):
        """Bedtime: Perform more replay steps and save the model."""
        print("Frontal Lobe: Starting consolidation...")
        if len(self.memory) < self.replay_batch_size:
            print(
                "Frontal Lobe: Not enough experiences in memory to consolidate with extensive replay."
            )
        else:
            # Perform more replay steps for consolidation
            consolidation_replay_count = 10  # Example: 10 replay batches
            print(
                f"Frontal Lobe: Performing {consolidation_replay_count} replay batches for consolidation."
            )
            for i in range(consolidation_replay_count):
                # print(f"Consolidation replay batch {i+1}/{consolidation_replay_count}")
                self.replay()

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
                f"Step {i+1}: Epsilon: {frontal_ai.exploration_rate_epsilon:.3f}, Memory size: {len(frontal_ai.memory)}"
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
    print("\nCleaned up test model files.")

    print("\nFrontal Lobe AI (DQN) test script finished.")
