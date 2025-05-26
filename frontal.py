# frontal.py (Decision-Making, Planning using Q-learning)
import numpy as np
import json
import os

class FrontalLobeAI:
    def __init__(self, model_path="data/frontal_model.json"):
        self.input_size = 18  # Updated: (Occipital output (5) + Parietal output (3) + Temporal output (10))
        self.output_size = 5  # Number of discrete actions

        self.weights = np.random.randn(self.input_size, self.output_size) * 0.01

        self.learning_rate_q = 0.1
        self.discount_factor_gamma = 0.0
        self.exploration_rate_epsilon = 1.0
        self.epsilon_decay_rate = 0.001
        self.min_epsilon = 0.01

        self.memory = []
        self.max_memory_size = 100

        self.model_path = model_path
        self.load_model()

    def _prepare_state_vector(self, state_data):
        try:
            state_vector_1d = np.array(state_data, dtype=float).flatten()
            if state_vector_1d.shape[0] != self.input_size:
                if state_vector_1d.shape[0] < self.input_size:
                    padding = np.zeros(self.input_size - state_vector_1d.shape[0])
                    state_vector_1d = np.concatenate((state_vector_1d, padding))
                else:
                    state_vector_1d = state_vector_1d[:self.input_size]
        except ValueError:
            state_vector_1d = np.zeros(self.input_size)
        return state_vector_1d

    def process_task(self, input_data):
        state_vector_1d = self._prepare_state_vector(input_data)
        state_vector_2d = state_vector_1d.reshape(1, -1)

        random_num = np.random.rand()
        if random_num < self.exploration_rate_epsilon:
            action = np.random.randint(0, self.output_size)
        else:
            q_values = state_vector_2d @ self.weights
            action = np.argmax(q_values.flatten())

        self.exploration_rate_epsilon = max(
            self.min_epsilon,
            self.exploration_rate_epsilon - self.epsilon_decay_rate
        )
        return int(action)

    def learn(self, state, action, reward, next_state, done):
        state_list = state.tolist() if isinstance(state, np.ndarray) else list(state)
        next_state_list = next_state.tolist() if isinstance(next_state, np.ndarray) else list(next_state)

        self.memory.append((state_list, action, reward, next_state_list, done))
        if len(self.memory) > self.max_memory_size:
            self.memory.pop(0)

        state_vector_1d = self._prepare_state_vector(state)

        current_q_values_for_state = state_vector_1d @ self.weights
        q_value_for_action_taken = current_q_values_for_state[action]

        q_target = reward

        error = q_target - q_value_for_action_taken
        self.weights[:, action] += self.learning_rate_q * error * state_vector_1d

    def consolidate(self):
        """Bedtime: Replay experiences from memory to refine Q-learning model."""
        if len(self.memory) == 0:
            self.save_model()
            return

        batch_size = 32
        # Ensure batch_size is not larger than the number of items in memory
        actual_batch_size = min(batch_size, len(self.memory))

        # Randomly sample experiences from memory
        indices = np.random.choice(len(self.memory), size=actual_batch_size, replace=False)
        batch = [self.memory[i] for i in indices]

        for state_list, action, reward, next_state_list, done in batch:
            # Convert state_list (stored as list) back to a 1D NumPy array
            state_vector_1d = self._prepare_state_vector(state_list)

            # Q-Learning Update Rule (same as in learn method)
            current_q_values_for_state = state_vector_1d @ self.weights
            q_value_for_action_taken = current_q_values_for_state[action]

            # q_target calculation (simplified due to gamma = 0.0)
            # If gamma > 0, you would need next_state_vector_1d and handle 'done'
            # next_state_vector_1d = self._prepare_state_vector(next_state_list)
            # max_q_next_state = np.max(next_state_vector_1d @ self.weights) if not done else 0.0
            # q_target = reward + self.discount_factor_gamma * max_q_next_state
            q_target = reward

            error = q_target - q_value_for_action_taken

            # Update weights for the specific action column
            # Using self.learning_rate_q for consolidation as well, as per current attributes.
            # A separate self.consolidation_learning_rate could be used if different behavior is desired.
            self.weights[:, action] += self.learning_rate_q * error * state_vector_1d

        self.save_model()

    def save_model(self):
        model_data = {
            'weights': self.weights.tolist(),
            'exploration_rate_epsilon': self.exploration_rate_epsilon
        }
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'w') as f:
            json.dump(model_data, f)

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'r') as f:
                    data = json.load(f)

                loaded_weights = np.array(data['weights'])
                if loaded_weights.shape == (self.input_size, self.output_size):
                    self.weights = loaded_weights
                else:
                    self.weights = np.random.randn(self.input_size, self.output_size) * 0.01

                self.exploration_rate_epsilon = data.get('exploration_rate_epsilon', 1.0)

            except Exception as e:
                self.weights = np.random.randn(self.input_size, self.output_size) * 0.01
                self.exploration_rate_epsilon = 1.0
        else:
            self.weights = np.random.randn(self.input_size, self.output_size) * 0.01
            self.exploration_rate_epsilon = 1.0

# Example Usage
if __name__ == "__main__":
    frontal_ai = FrontalLobeAI(model_path="data/test_frontal_q_consolidate.json")
    print(f"FrontalLobeAI initialized for Q-learning consolidate test.")

    # Populate memory with some experiences
    for i in range(50): # Add 50 experiences
        state = np.random.rand(frontal_ai.input_size)
        action = frontal_ai.process_task(state) # Epsilon will decay
        reward = np.random.choice([-1, 0, 1])
        next_state = np.random.rand(frontal_ai.input_size)
        done = (i % 10 == 0) # Some episodes terminate
        frontal_ai.learn(state, action, reward, next_state, done)

    print(f"Memory size before consolidate: {len(frontal_ai.memory)}")
    print(f"Epsilon before consolidate: {frontal_ai.exploration_rate_epsilon:.3f}")
    weights_before_consolidate_sample = frontal_ai.weights[0,0]

    frontal_ai.consolidate() # Perform experience replay

    print(f"Epsilon after consolidate (should be unchanged by consolidate): {frontal_ai.exploration_rate_epsilon:.3f}")
    weights_after_consolidate_sample = frontal_ai.weights[0,0]

    # Check if weights changed (they should if memory was not empty and learning rate > 0)
    if len(frontal_ai.memory) > 0 and frontal_ai.learning_rate_q > 0:
        assert weights_before_consolidate_sample != weights_after_consolidate_sample, "Weights did not change during consolidate."
        print("Weights changed during consolidation, as expected.")
    else:
        print("Weights did not change, possibly due to empty memory or zero learning rate.")


    if os.path.exists("data/test_frontal_q_consolidate.json"):
        os.remove("data/test_frontal_q_consolidate.json")
    print("Frontal Lobe AI Q-learning consolidate test finished.")
