import unittest
import os
import sys
import json
import shutil # For cleaning up directories

import numpy as np
from PIL import Image # For creating test images

# Add the project root directory to sys.path to allow absolute imports
# from modules in the root directory (e.g., main.py, temporal.py).
# This must be done BEFORE attempting to import any project modules.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now, import project modules
from main import (
    load_sensor_data,
    load_image_text_pairs,
    load_vision_data,
    load_text_data,
)
from temporal import TemporalLobeAI
from cerebellum import CerebellumAI
from limbic import LimbicSystemAI
from occipital import OccipitalLobeAI
from frontal import FrontalLobeAI
from parietal import ParietalLobeAI # Added ParietalLobeAI

# Define a temporary test data directory
TEST_DATA_DIR = "data_test"
TEST_IMAGES_SUBDIR = os.path.join(TEST_DATA_DIR, "images") # Subdirectory for test images

# Define paths for dummy data files within TEST_DATA_DIR
TEST_VISION_FILE = os.path.join(TEST_DATA_DIR, "test_vision.json")
TEST_SENSOR_FILE = os.path.join(TEST_DATA_DIR, "test_sensors.json")
TEST_TEXT_FILE = os.path.join(TEST_DATA_DIR, "test_text.json")
TEST_PAIRS_FILE = os.path.join(TEST_DATA_DIR, "test_pairs.json")


# Define paths for AI model files within TEST_DATA_DIR
TEST_TEMPORAL_MODEL_PATH = os.path.join(TEST_DATA_DIR, "test_temporal_model.json")
TEST_TEMPORAL_MEMORY_PATH = os.path.join(TEST_DATA_DIR, "test_temporal_memory.json")
TEST_CEREBELLUM_MODEL_PATH = os.path.join(TEST_DATA_DIR, "test_cerebellum_model.json")
TEST_LIMBIC_MODEL_PATH = os.path.join(TEST_DATA_DIR, "test_limbic_model.json")
TEST_OCCIPITAL_MODEL_PATH = os.path.join(TEST_DATA_DIR, "test_occipital_model.json")
TEST_FRONTAL_MODEL_PATH = os.path.join(TEST_DATA_DIR, "test_frontal_model.json")
TEST_PARIETAL_MODEL_PATH = os.path.join(TEST_DATA_DIR, "test_parietal_model.json") # Added for ParietalLobeAI


class TestDataLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.makedirs(TEST_DATA_DIR, exist_ok=True)
        cls.vision_content = {"image_paths": [os.path.join(TEST_IMAGES_SUBDIR, "img1.png")]}
        with open(TEST_VISION_FILE, 'w') as f: json.dump(cls.vision_content, f)
        cls.sensor_content = [[1.0, 2.0], [3.0, 4.0]]
        with open(TEST_SENSOR_FILE, 'w') as f: json.dump(cls.sensor_content, f)
        cls.text_content = ["hello", "world"]
        with open(TEST_TEXT_FILE, 'w') as f: json.dump(cls.text_content, f)
        cls.pairs_content = [{"image_path": "img1.png", "text_description": "text1", "visual_label": 0},
                                {"image_path": "img2.png", "text_description": "text2", "visual_label": 1}]
        with open(TEST_PAIRS_FILE, 'w') as f: json.dump(cls.pairs_content, f)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(TEST_DATA_DIR): shutil.rmtree(TEST_DATA_DIR)

    def test_load_vision_data_valid(self):
        data = load_vision_data(filepath=TEST_VISION_FILE)
        self.assertEqual(data, self.vision_content["image_paths"])
    def test_load_sensor_data_valid(self):
        data = load_sensor_data(filepath=TEST_SENSOR_FILE)
        self.assertEqual(data, self.sensor_content)
    def test_load_text_data_valid(self):
        data = load_text_data(filepath=TEST_TEXT_FILE)
        self.assertEqual(data, self.text_content)
    def test_load_image_text_pairs_valid(self):
        data = load_image_text_pairs(filepath=TEST_PAIRS_FILE)
        self.assertEqual(data, self.pairs_content)
    def test_load_vision_data_missing_file(self):
        self.assertEqual(load_vision_data(filepath="non_existent_vision.json"), [])
    def test_load_sensor_data_missing_file(self):
        self.assertEqual(load_sensor_data(filepath="non_existent_sensor.json"), [])
    def test_load_text_data_missing_file(self):
        self.assertEqual(load_text_data(filepath="non_existent_text.json"), [])
    def test_load_image_text_pairs_missing_file(self):
        self.assertEqual(load_image_text_pairs(filepath="non_existent_pairs.json"), [])


class TestTemporalLobeAI(unittest.TestCase):
    def setUp(self):
        os.makedirs(TEST_DATA_DIR, exist_ok=True)
        # Ensure clean state before AI initialization
        if os.path.exists(TEST_TEMPORAL_MODEL_PATH):
            os.remove(TEST_TEMPORAL_MODEL_PATH)
        if os.path.exists(TEST_TEMPORAL_MEMORY_PATH):
            os.remove(TEST_TEMPORAL_MEMORY_PATH)

        self.ai = TemporalLobeAI(model_path=TEST_TEMPORAL_MODEL_PATH, memory_path=TEST_TEMPORAL_MEMORY_PATH)
        self.ai._initialize_default_weights_biases()
        self.ai.save_model()
        self.ai.save_memory()

    def tearDown(self):
        if os.path.exists(TEST_TEMPORAL_MODEL_PATH): os.remove(TEST_TEMPORAL_MODEL_PATH)
        if os.path.exists(TEST_TEMPORAL_MEMORY_PATH): os.remove(TEST_TEMPORAL_MEMORY_PATH)

    def test_forward_prop_text_output_shapes(self):
        input_vec_1d = self.ai._to_numerical_vector("sample", self.ai.input_size)
        # Expected return: text_input_vec, text_hidden_output, text_embedding_scores
        _, text_hidden_output, text_embedding_scores = self.ai._forward_prop_text(input_vec_1d)
        self.assertEqual(text_hidden_output.shape, (self.ai.text_hidden_size,))
        self.assertEqual(text_embedding_scores.shape, (self.ai.output_size,)) # output_size is embedding_size

    def test_forward_prop_visual_assoc_output_shapes(self):
        test_embedding_vec = np.random.rand(self.ai.output_size) # output_size is embedding_size
        # Expected return: visual_input_vec (embedding), visual_hidden_output, visual_label_scores
        _, visual_hidden_output, visual_label_scores = self.ai._forward_prop_visual_assoc(test_embedding_vec)
        self.assertEqual(visual_hidden_output.shape, (self.ai.visual_assoc_hidden_size,))
        self.assertEqual(visual_label_scores.shape, (self.ai.visual_output_size,))

    def test_process_task_predict_visual_flag(self):
        output_with_visual = self.ai.process_task("text", predict_visual=True)
        self.assertIsInstance(output_with_visual, tuple); self.assertEqual(len(output_with_visual), 2)
        self.assertIsInstance(output_with_visual[0], list); self.assertIsInstance(output_with_visual[1], (int, np.integer))
        output_without_visual = self.ai.process_task("text", predict_visual=False)
        self.assertIsInstance(output_without_visual, list)
    def test_learn_updates_all_weights_and_biases(self):
        initial_w = [p.copy() for p in [self.ai.weights_text_input_hidden, self.ai.bias_text_hidden, self.ai.weights_text_hidden_embedding, self.ai.bias_text_embedding, self.ai.weights_embed_to_visual_hidden, self.ai.bias_visual_assoc_hidden, self.ai.weights_visual_hidden_to_label, self.ai.bias_visual_assoc_label]]
        self.ai.learn([("text", "target")], visual_label_as_context=1)
        final_w = [self.ai.weights_text_input_hidden, self.ai.bias_text_hidden, self.ai.weights_text_hidden_embedding, self.ai.bias_text_embedding, self.ai.weights_embed_to_visual_hidden, self.ai.bias_visual_assoc_hidden, self.ai.weights_visual_hidden_to_label, self.ai.bias_visual_assoc_label]
        for i_w, f_w in zip(initial_w, final_w): self.assertFalse(np.array_equal(i_w, f_w))
        self.assertIn([("text", "target")], self.ai.memory_db, "Sequence [('text', 'target')] not found in memory_db")
        self.assertIn(("text", 1), self.ai.cross_modal_memory)
    def test_consolidate_updates_all_weights_and_biases(self):
        self.ai.learn(sequence=[("text c1", "target c1")], visual_label_as_context=0)
        self.ai.learn(sequence=[("text c2", "target c2")], visual_label_as_context=2)
        # Capture all weights and biases
        initial_params = [p.copy() for p in [
            self.ai.weights_text_input_hidden, self.ai.bias_text_hidden,
            self.ai.weights_text_hidden_embedding, self.ai.bias_text_embedding,
            self.ai.weights_embed_to_visual_hidden, self.ai.bias_visual_assoc_hidden,
            self.ai.weights_visual_hidden_to_label, self.ai.bias_visual_assoc_label
        ]]
        self.ai.consolidate()
        final_params = [
            self.ai.weights_text_input_hidden, self.ai.bias_text_hidden,
            self.ai.weights_text_hidden_embedding, self.ai.bias_text_embedding,
            self.ai.weights_embed_to_visual_hidden, self.ai.bias_visual_assoc_hidden,
            self.ai.weights_visual_hidden_to_label, self.ai.bias_visual_assoc_label
        ]
        for i_p, f_p in zip(initial_params, final_params):
            self.assertFalse(np.array_equal(i_p, f_p), "A weight or bias attribute was not updated during consolidate.")

    def test_model_save_load_new_temporal_architecture(self):
        # Set known, distinct values for a selection of weights and biases
        self.ai.weights_text_input_hidden = np.full((self.ai.input_size, self.ai.text_hidden_size), 0.11)
        self.ai.bias_text_embedding = np.full((1, self.ai.output_size), 0.22) # output_size is embedding_size
        self.ai.weights_visual_hidden_to_label = np.full((self.ai.visual_assoc_hidden_size, self.ai.visual_output_size), 0.33)
        self.ai.bias_visual_assoc_hidden = np.full((1, self.ai.visual_assoc_hidden_size), 0.44)

        # Save and load the model
        self.ai.save_model()
        loaded_ai = TemporalLobeAI(model_path=TEST_TEMPORAL_MODEL_PATH, memory_path=TEST_TEMPORAL_MEMORY_PATH)

        # Assert that the loaded AI instance has the correct values for the set weights/biases
        np.testing.assert_array_almost_equal(loaded_ai.weights_text_input_hidden, self.ai.weights_text_input_hidden)
        np.testing.assert_array_almost_equal(loaded_ai.bias_text_embedding, self.ai.bias_text_embedding)
        np.testing.assert_array_almost_equal(loaded_ai.weights_visual_hidden_to_label, self.ai.weights_visual_hidden_to_label)
        np.testing.assert_array_almost_equal(loaded_ai.bias_visual_assoc_hidden, self.ai.bias_visual_assoc_hidden)

        # Verify architectural parameters are loaded correctly
        self.assertEqual(loaded_ai.input_size, self.ai.input_size)
        self.assertEqual(loaded_ai.text_hidden_size, self.ai.text_hidden_size)
        self.assertEqual(loaded_ai.output_size, self.ai.output_size) # embedding_size
        self.assertEqual(loaded_ai.visual_assoc_hidden_size, self.ai.visual_assoc_hidden_size)
        self.assertEqual(loaded_ai.visual_output_size, self.ai.visual_output_size)

    def test_memory_persistence_dict_format(self):
        self.ai.memory_db.append([("m_item","t_item")]); self.ai.cross_modal_memory.append(("cm_text",1))
        self.ai.save_memory()
        new_ai = TemporalLobeAI(TEST_TEMPORAL_MODEL_PATH, TEST_TEMPORAL_MEMORY_PATH)
        self.assertIn([("m_item","t_item")], new_ai.memory_db)
        self.assertIn(("cm_text",1), new_ai.cross_modal_memory)


class TestLimbicSystemAI(unittest.TestCase):
    def setUp(self):
        os.makedirs(TEST_DATA_DIR, exist_ok=True)
        # Ensure clean state before AI initialization
        if os.path.exists(TEST_LIMBIC_MODEL_PATH): os.remove(TEST_LIMBIC_MODEL_PATH)

        self.ai = LimbicSystemAI(model_path=TEST_LIMBIC_MODEL_PATH)
        self.ai._initialize_default_weights_biases()
        self.ai.save_model()

        self.sample_temporal_output = np.random.rand(self.ai.input_size).tolist()
        self.sample_true_emotion = 1; self.sample_reward = 1.0

    def tearDown(self):
        if os.path.exists(TEST_LIMBIC_MODEL_PATH): os.remove(TEST_LIMBIC_MODEL_PATH)
    def test_ensure_input_vector_shape(self):
        vec=self.ai._ensure_input_vector_shape(self.sample_temporal_output); self.assertEqual(vec.shape,(self.ai.input_size,))
    def test_forward_propagate_output_shapes_limbic(self):
        # Unpack with clearer variable names
        input_vec_1d, hidden_layer_output_1d, output_scores_1d = self.ai._forward_propagate(self.sample_temporal_output)
        self.assertEqual(input_vec_1d.shape,(self.ai.input_size,))
        self.assertEqual(hidden_layer_output_1d.shape,(self.ai.hidden_size,))
        self.assertEqual(output_scores_1d.shape,(self.ai.output_size,))

    def test_process_task_output_label_limbic(self):
        label=self.ai.process_task(self.sample_temporal_output); self.assertTrue(0<=label<self.ai.output_size)

    def test_learn_updates_and_memory_limbic(self):
        initial_params = {
            "w_ih": self.ai.weights_input_hidden.copy(),
            "b_h": self.ai.bias_hidden.copy(),
            "w_ho": self.ai.weights_hidden_output.copy(),
            "b_o": self.ai.bias_output.copy()
        }
        self.ai.learn(self.sample_temporal_output,self.sample_true_emotion,self.sample_reward)

        self.assertFalse(np.array_equal(initial_params["w_ih"], self.ai.weights_input_hidden), "weights_input_hidden did not change after learn")
        self.assertFalse(np.array_equal(initial_params["b_h"], self.ai.bias_hidden), "bias_hidden did not change after learn")
        self.assertFalse(np.array_equal(initial_params["w_ho"], self.ai.weights_hidden_output), "weights_hidden_output did not change after learn")
        self.assertFalse(np.array_equal(initial_params["b_o"], self.ai.bias_output), "bias_output did not change after learn")

        self.assertIn((self.sample_temporal_output,self.sample_true_emotion,self.sample_reward),self.ai.memory)

    def test_consolidate_updates_weights_limbic(self):
        self.ai.learn(self.sample_temporal_output,self.sample_true_emotion,self.sample_reward) # Ensure memory has data
        initial_params = {
            "w_ih": self.ai.weights_input_hidden.copy(),
            "b_h": self.ai.bias_hidden.copy(),
            "w_ho": self.ai.weights_hidden_output.copy(),
            "b_o": self.ai.bias_output.copy()
        }
        self.ai.consolidate()

        self.assertFalse(np.array_equal(initial_params["w_ih"], self.ai.weights_input_hidden), "weights_input_hidden did not change after consolidate")
        self.assertFalse(np.array_equal(initial_params["b_h"], self.ai.bias_hidden), "bias_hidden did not change after consolidate")
        self.assertFalse(np.array_equal(initial_params["w_ho"], self.ai.weights_hidden_output), "weights_hidden_output did not change after consolidate")
        self.assertFalse(np.array_equal(initial_params["b_o"], self.ai.bias_output), "bias_output did not change after consolidate")

    def test_model_save_load_new_limbic_architecture(self):
        # Set known values for specified attributes
        self.ai.weights_input_hidden = np.full((self.ai.input_size, self.ai.hidden_size), 0.77)
        self.ai.bias_output = np.full((1, self.ai.output_size), 0.88)
        # Capture other params to ensure they are saved and loaded too
        expected_bias_hidden = self.ai.bias_hidden.copy()
        expected_weights_hidden_output = self.ai.weights_hidden_output.copy()

        self.ai.save_model()
        loaded_ai = LimbicSystemAI(model_path=TEST_LIMBIC_MODEL_PATH)

        # Assert that the loaded AI instance has the correct values for the set weights/biases
        np.testing.assert_array_almost_equal(loaded_ai.weights_input_hidden, self.ai.weights_input_hidden)
        np.testing.assert_array_almost_equal(loaded_ai.bias_output, self.ai.bias_output)

        # Assert that other parameters were also saved and loaded correctly
        np.testing.assert_array_almost_equal(loaded_ai.bias_hidden, expected_bias_hidden)
        np.testing.assert_array_almost_equal(loaded_ai.weights_hidden_output, expected_weights_hidden_output)

        # Verify architectural parameters are loaded correctly
        self.assertEqual(loaded_ai.input_size, self.ai.input_size)
        self.assertEqual(loaded_ai.hidden_size, self.ai.hidden_size)
        self.assertEqual(loaded_ai.output_size, self.ai.output_size)


class TestOccipitalLobeAI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.makedirs(TEST_IMAGES_SUBDIR, exist_ok=True)
        cls.img_size=(32,32)
        # Create a white test image and save its path
        cls.test_img_path = os.path.join(TEST_IMAGES_SUBDIR, "test_image.png")
        Image.new('L', cls.img_size, 255).save(cls.test_img_path) # 255 for white
    @classmethod
    def tearDownClass(cls):
        if os.path.exists(TEST_IMAGES_SUBDIR): shutil.rmtree(TEST_IMAGES_SUBDIR)
        # Model file is now handled by individual tearDown methods or should be if specific instances are problematic
        # However, if setUpClass created a model (it doesn't here), this would be the place to clean it.
        # For safety, we can leave it, or rely on the per-test tearDown.
        # Given the task, individual tearDown is preferred.
        if os.path.exists(TEST_OCCIPITAL_MODEL_PATH): os.remove(TEST_OCCIPITAL_MODEL_PATH) # Can be removed if tearDown is robust

    def setUp(self):
        os.makedirs(TEST_DATA_DIR, exist_ok=True) # Ensure data directory exists
        # Ensure clean state before AI initialization
        if os.path.exists(TEST_OCCIPITAL_MODEL_PATH):
            os.remove(TEST_OCCIPITAL_MODEL_PATH)

        self.ai = OccipitalLobeAI(model_path=TEST_OCCIPITAL_MODEL_PATH)
        self.ai._initialize_default_weights_biases() # Added missing call
        self.ai.save_model()

    def tearDown(self):
        # Ensure model file created in setUp is removed
        if os.path.exists(TEST_OCCIPITAL_MODEL_PATH):
            os.remove(TEST_OCCIPITAL_MODEL_PATH)

    def test_forward_propagate_output_shapes(self):
        # Unpack with clearer variable names
        input_vec_1d, hidden_layer_output_1d, output_scores_1d = self.ai._forward_propagate(self.test_img_path)
        self.assertEqual(input_vec_1d.shape, (self.ai.input_size,), "Input vector shape mismatch")
        self.assertEqual(hidden_layer_output_1d.shape, (self.ai.hidden_size,), "Hidden layer output shape mismatch")
        self.assertEqual(output_scores_1d.shape, (self.ai.output_size,), "Output scores shape mismatch")

    def test_learn_updates_and_memory(self):
        initial_params = {
            "w_ih": self.ai.weights_input_hidden.copy(),
            "b_h": self.ai.bias_hidden.copy(),
            "w_ho": self.ai.weights_hidden_output.copy(),
            "b_o": self.ai.bias_output.copy()
        }
        self.ai.learn(self.test_img_path, 0) # Use the white test image

        self.assertFalse(np.array_equal(initial_params["w_ih"], self.ai.weights_input_hidden), "weights_input_hidden did not change after learn")
        self.assertFalse(np.array_equal(initial_params["b_h"], self.ai.bias_hidden), "bias_hidden did not change after learn")
        self.assertFalse(np.array_equal(initial_params["w_ho"], self.ai.weights_hidden_output), "weights_hidden_output did not change after learn")
        self.assertFalse(np.array_equal(initial_params["b_o"], self.ai.bias_output), "bias_output did not change after learn")

        self.assertIn((self.test_img_path,0),self.ai.memory) # Use the white test image path

    def test_consolidate_updates_weights(self):
        self.ai.learn(self.test_img_path, 1) # Ensure memory has data, using white test image
        initial_params = {
            "w_ih": self.ai.weights_input_hidden.copy(),
            "b_h": self.ai.bias_hidden.copy(),
            "w_ho": self.ai.weights_hidden_output.copy(),
            "b_o": self.ai.bias_output.copy()
        }
        self.ai.consolidate()

        self.assertFalse(np.array_equal(initial_params["w_ih"], self.ai.weights_input_hidden), "weights_input_hidden did not change after consolidate")
        self.assertFalse(np.array_equal(initial_params["b_h"], self.ai.bias_hidden), "bias_hidden did not change after consolidate")
        self.assertFalse(np.array_equal(initial_params["w_ho"], self.ai.weights_hidden_output), "weights_hidden_output did not change after consolidate")
        self.assertFalse(np.array_equal(initial_params["b_o"], self.ai.bias_output), "bias_output did not change after consolidate")

    def test_model_save_load_new_architecture(self):
        # Set known values for specified attributes
        self.ai.weights_input_hidden = np.full((self.ai.input_size, self.ai.hidden_size), 0.55)
        self.ai.bias_output = np.full((1, self.ai.output_size), 0.66)
        # Capture other params to ensure they are saved and loaded too
        expected_bias_hidden = self.ai.bias_hidden.copy()
        expected_weights_hidden_output = self.ai.weights_hidden_output.copy()

        self.ai.save_model()
        loaded_ai = OccipitalLobeAI(model_path=TEST_OCCIPITAL_MODEL_PATH)

        # Assert that the loaded AI instance has the correct values for the set weights/biases
        np.testing.assert_array_almost_equal(loaded_ai.weights_input_hidden, self.ai.weights_input_hidden)
        np.testing.assert_array_almost_equal(loaded_ai.bias_output, self.ai.bias_output)

        # Assert that other parameters were also saved and loaded correctly
        np.testing.assert_array_almost_equal(loaded_ai.bias_hidden, expected_bias_hidden)
        np.testing.assert_array_almost_equal(loaded_ai.weights_hidden_output, expected_weights_hidden_output)

        # Verify architectural parameters are loaded correctly
        self.assertEqual(loaded_ai.input_size, self.ai.input_size)
        self.assertEqual(loaded_ai.hidden_size, self.ai.hidden_size)
        self.assertEqual(loaded_ai.output_size, self.ai.output_size)


class TestFrontalLobeAI(unittest.TestCase):
    def setUp(self):
        os.makedirs(TEST_DATA_DIR, exist_ok=True)
        if os.path.exists(TEST_FRONTAL_MODEL_PATH): # Clean up before test
            os.remove(TEST_FRONTAL_MODEL_PATH)
        self.ai = FrontalLobeAI(model_path=TEST_FRONTAL_MODEL_PATH)
        # self.ai.weights is initialized in __init__ or load_model.
        # Default exploration_rate_epsilon is set in __init__.
        self.ai.save_model() # Save the initial model (fresh or loaded)
        self.sample_state = np.random.rand(self.ai.input_size).tolist()

    def tearDown(self):
        if os.path.exists(TEST_FRONTAL_MODEL_PATH): os.remove(TEST_FRONTAL_MODEL_PATH)

    def test_process_task_epsilon_greedy_selection(self):
        # Test with high exploration: actions should be somewhat random
        self.ai.exploration_rate_epsilon = 1.0
        action_counts_explore = {i:0 for i in range(self.ai.output_size)}
        for _ in range(1000): action_counts_explore[self.ai.process_task(self.sample_state)] += 1
        self.assertTrue(len(set(action_counts_explore.values())) > 1 or self.ai.output_size == 1)
        self.ai.exploration_rate_epsilon = 0.0
        # Ensure weights for action 1 are highest for a deterministic choice
        self.ai.weights = np.zeros((self.ai.input_size, self.ai.output_size))
        self.ai.weights[:, 1] = 1.0 # Action 1 has highest Q-value
        self.assertEqual(self.ai.process_task(self.sample_state), 1, "Should pick action with highest Q-value when epsilon is 0")

        # Test epsilon decay
        initial_epsilon = 0.5
        self.ai.exploration_rate_epsilon = initial_epsilon
        self.ai.process_task(self.sample_state) # This call will decay epsilon
        self.assertAlmostEqual(self.ai.exploration_rate_epsilon,
                               max(self.ai.min_epsilon, initial_epsilon - self.ai.epsilon_decay_rate),
                               msg="Epsilon decay is not working as expected")

    def test_learn_q_update_and_memory(self):
        # Use a fresh AI instance for this specific test if needed, or ensure state is well-defined
        # self.setUp() # This would re-initialize self.ai, which might be desired for some tests

        state = self.sample_state # Uses current self.ai.input_size
        action_taken = 1
        reward = 1.0
        next_state = np.random.rand(self.ai.input_size).tolist() # Uses current self.ai.input_size
        done = False

        initial_weights_for_action = self.ai.weights[:, action_taken].copy()
        initial_weights_all_actions = self.ai.weights.copy()

        state_vector_1d = self.ai._prepare_state_vector(state)
        next_state_vector_1d = self.ai._prepare_state_vector(next_state)

        # Q-value before learning
        q_value_before = (state_vector_1d @ initial_weights_all_actions)[action_taken]

        # Expected Q-value for next state (max Q for non-terminal, 0 for terminal)
        next_max_q = np.max(next_state_vector_1d @ self.ai.weights) if not done else 0.0

        # TD target
        target_q_value = reward + self.ai.discount_factor_gamma * next_max_q

        # TD error
        error = target_q_value - q_value_before

        # Expected weights for the action taken
        # Note: Q-learning updates the Q-value directly, which is self.weights[state_index, action_taken]
        # For a linear function approximator Q(s,a) = w_a . phi(s), the update is w_a = w_a + lr * error * phi(s)
        # So, self.weights[:, action_taken] is the weight vector w_a for action 'a'. phi(s) is state_vector_1d.
        expected_weights_for_action_after_learn = initial_weights_for_action + self.ai.learning_rate_q * error * state_vector_1d

        self.ai.learn(state, action_taken, reward, next_state, done)

        np.testing.assert_array_almost_equal(self.ai.weights[:, action_taken], expected_weights_for_action_after_learn, decimal=5,
                                            err_msg="Q-value update for the taken action is incorrect.")

        # Check that weights for other actions are not changed
        for i in range(self.ai.output_size):
            if i != action_taken:
                np.testing.assert_array_equal(self.ai.weights[:, i], initial_weights_all_actions[:, i],
                                             err_msg=f"Weights for action {i} (not taken) should not change.")

        self.assertIn((list(state), action_taken, reward, list(next_state), done), self.ai.memory)

        # Test memory limit
        self.ai.memory = []
        for _ in range(self.ai.max_memory_size + 5):
            self.ai.learn(state, action_taken, reward, next_state, done)
        self.assertEqual(len(self.ai.memory), self.ai.max_memory_size, "Memory size limit not enforced.")

    def test_consolidate_experience_replay(self):
        # Populate memory with diverse experiences reflecting new input_size
        for i in range(10):
            state = np.random.rand(self.ai.input_size).tolist()
            action = np.random.randint(0, self.ai.output_size)
            reward_val = float(np.random.choice([-1,0,1]))
            next_s = np.random.rand(self.ai.input_size).tolist()
            done_val = bool(np.random.choice([True,False]))
            self.ai.learn(state, action, reward_val, next_s, done_val)

        self.assertTrue(len(self.ai.memory) > 0, "Memory should be populated before consolidation.")
        initial_weights = self.ai.weights.copy()
        self.ai.consolidate() # This should perform experience replay and update weights
        self.assertFalse(np.array_equal(initial_weights, self.ai.weights),
                         "Weights should change after consolidation if learning rate is > 0 and memory is not empty.")

    def test_model_save_load_q_learning_params(self):
        # Set specific values to test save/load
        self.ai.exploration_rate_epsilon = 0.678
        # Ensure weights have the correct current dimensions
        self.ai.weights = np.random.rand(self.ai.input_size, self.ai.output_size) * 0.5
        current_weights_shape = self.ai.weights.shape

        known_weights_current_format = self.ai.weights.copy()
        self.ai.save_model()

        loaded_ai = FrontalLobeAI(model_path=TEST_FRONTAL_MODEL_PATH)
        self.assertAlmostEqual(loaded_ai.exploration_rate_epsilon, 0.678, msg="Exploration rate not loaded correctly.")
        np.testing.assert_array_almost_equal(loaded_ai.weights, known_weights_current_format,
                                             err_msg="Weights not loaded correctly from new format.")
        self.assertEqual(loaded_ai.weights.shape, current_weights_shape, "Loaded weights shape mismatch (new format).")

        # Test backward compatibility: loading old model format (only 'weights' key)
        # Create a dummy old-format model file. Use different dimensions to test shape handling.
        old_input_size = 10 # Assume old input size was 10 for this test
        old_weights_data = (np.random.rand(old_input_size, self.ai.output_size) * 0.3).tolist()
        with open(TEST_FRONTAL_MODEL_PATH, 'w') as f:
            json.dump({'weights': old_weights_data}, f)

        loaded_ai_old_format = FrontalLobeAI(model_path=TEST_FRONTAL_MODEL_PATH)
        self.assertEqual(loaded_ai_old_format.exploration_rate_epsilon, 1.0,
                         "Epsilon should default to 1.0 when loading old format without it.")

        # Check if weights were re-initialized due to shape mismatch
        # This assumes FrontalLobeAI's load_model re-initializes if shapes don't match.
        self.assertEqual(loaded_ai_old_format.weights.shape, (self.ai.input_size, self.ai.output_size),
                         "Weights shape should match current AI's input_size after loading old format with shape mismatch.")
        # Further check: ensure weights are not empty or zero if re-initialized (depends on __init__ behavior)
        self.assertTrue(np.any(loaded_ai_old_format.weights), "Re-initialized weights should not be all zero (assuming default init).")


class TestCerebellumAI(unittest.TestCase):
    def setUp(self):
        os.makedirs(TEST_DATA_DIR, exist_ok=True)
        # Ensure clean state before AI initialization
        if os.path.exists(TEST_CEREBELLUM_MODEL_PATH):
            os.remove(TEST_CEREBELLUM_MODEL_PATH)

        self.ai = CerebellumAI(model_path=TEST_CEREBELLUM_MODEL_PATH)
        self.ai._initialize_default_weights_biases()
        self.ai.save_model()

        self.sample_sensor_data = np.random.rand(self.ai.input_size).tolist()
        self.sample_true_command = (np.random.rand(self.ai.output_size) * 2 - 1).tolist()

    def tearDown(self):
        if os.path.exists(TEST_CEREBELLUM_MODEL_PATH):
            os.remove(TEST_CEREBELLUM_MODEL_PATH)

    def test_prepare_input_vector(self):
        vec = self.ai._prepare_input_vector(self.sample_sensor_data)
        self.assertEqual(vec.shape, (self.ai.input_size,))
        short_data = self.sample_sensor_data[:-2]
        vec_short = self.ai._prepare_input_vector(short_data)
        self.assertEqual(vec_short.shape, (self.ai.input_size,))
        self.assertTrue(np.all(vec_short[len(short_data):] == 0))
        long_data = self.sample_sensor_data + [0.1, 0.2]
        vec_long = self.ai._prepare_input_vector(long_data)
        self.assertEqual(vec_long.shape, (self.ai.input_size,))
        np.testing.assert_array_almost_equal(vec_long, np.array(long_data[:self.ai.input_size]), decimal=5)

    def test_prepare_target_command_vector(self):
        vec = self.ai._prepare_target_command_vector(self.sample_true_command)
        self.assertEqual(vec.shape, (self.ai.output_size,))
        short_data = self.sample_true_command[:-1]; vec_short = self.ai._prepare_target_command_vector(short_data)
        self.assertEqual(vec_short.shape, (self.ai.output_size,))
        self.assertTrue(np.all(vec_short[len(short_data):] == 0))
        long_data = self.sample_true_command + [0.1]; vec_long = self.ai._prepare_target_command_vector(long_data)
        self.assertEqual(vec_long.shape, (self.ai.output_size,))
        np.testing.assert_array_almost_equal(vec_long, np.array(long_data[:self.ai.output_size]), decimal=5)

    def test_forward_propagate_output_shapes_and_ranges(self):
        input_vec_1d, hidden_layer_output, final_commands_output = self.ai._forward_propagate(self.sample_sensor_data)
        self.assertEqual(input_vec_1d.shape, (self.ai.input_size,), "Input vector shape mismatch.")
        self.assertEqual(hidden_layer_output.shape, (self.ai.hidden_size,), "Hidden layer output shape mismatch.")
        self.assertEqual(final_commands_output.shape, (self.ai.output_size,), "Final commands output shape mismatch.")
        self.assertTrue(np.all(hidden_layer_output >= -1) and np.all(hidden_layer_output <= 1), "Hidden output out of Tanh range.")
        self.assertTrue(np.all(final_commands_output >= -1) and np.all(final_commands_output <= 1), "Final commands out of Tanh range.")

    def test_process_task_output(self):
        commands = self.ai.process_task(self.sample_sensor_data)
        self.assertIsInstance(commands, list)
        self.assertEqual(len(commands), self.ai.output_size)
        for val in commands: self.assertTrue(-1.0 <= val <= 1.0)

    def test_learn_updates_and_memory(self):
        initial_w_ih = self.ai.weights_input_hidden.copy()
        initial_b_h = self.ai.bias_hidden.copy()
        initial_w_ih = self.ai.weights_input_hidden.copy()
        initial_b_h = self.ai.bias_hidden.copy()
        initial_w_ho = self.ai.weights_hidden_output.copy()
        initial_b_o = self.ai.bias_output.copy()

        self.ai.learn(self.sample_sensor_data, self.sample_true_command)

        self.assertFalse(np.array_equal(initial_w_ih, self.ai.weights_input_hidden), "weights_input_hidden should change after learn")
        self.assertFalse(np.array_equal(initial_b_h, self.ai.bias_hidden), "bias_hidden should change after learn")
        self.assertFalse(np.array_equal(initial_w_ho, self.ai.weights_hidden_output), "weights_hidden_output should change after learn")
        self.assertFalse(np.array_equal(initial_b_o, self.ai.bias_output), "bias_output should change after learn")

        expected_mem_sensor = self.sample_sensor_data if isinstance(self.sample_sensor_data, list) else self.sample_sensor_data.tolist()
        expected_mem_command = self.sample_true_command if isinstance(self.sample_true_command, list) else self.sample_true_command.tolist()
        self.assertIn((expected_mem_sensor, expected_mem_command), self.ai.memory)

        self.ai.memory = []
        for i in range(105): self.ai.learn(self.sample_sensor_data, (np.array(self.sample_true_command) * (i/105.0)).tolist())
        self.assertEqual(len(self.ai.memory), 100)

    def test_consolidate_updates_weights(self):
        self.ai.learn(self.sample_sensor_data, self.sample_true_command)
        w_ih_before = self.ai.weights_input_hidden.copy(); b_o_before = self.ai.bias_output.copy()
        w_ho_before = self.ai.weights_hidden_output.copy(); b_h_before = self.ai.bias_hidden.copy()


        self.ai.consolidate()

        self.assertFalse(np.array_equal(w_ih_before, self.ai.weights_input_hidden), "weights_input_hidden should change after consolidate")
        self.assertFalse(np.array_equal(b_h_before, self.ai.bias_hidden), "bias_hidden should change after consolidate")
        self.assertFalse(np.array_equal(w_ho_before, self.ai.weights_hidden_output), "weights_hidden_output should change after consolidate")
        self.assertFalse(np.array_equal(b_o_before, self.ai.bias_output), "bias_output should change after consolidate")

    def test_model_save_load_new_cerebellum_architecture(self):
        # Set known values for all attributes
        self.ai.weights_input_hidden = np.full((self.ai.input_size, self.ai.hidden_size), 0.51)
        self.ai.bias_hidden = np.full((1, self.ai.hidden_size), 0.52)
        self.ai.weights_hidden_output = np.full((self.ai.hidden_size, self.ai.output_size), 0.53)
        self.ai.bias_output = np.full((1, self.ai.output_size), 0.54)

        self.ai.save_model()
        loaded_ai = CerebellumAI(model_path=TEST_CEREBELLUM_MODEL_PATH)

        # Assert all attributes are loaded correctly
        np.testing.assert_array_almost_equal(loaded_ai.weights_input_hidden, self.ai.weights_input_hidden)
        np.testing.assert_array_almost_equal(loaded_ai.bias_hidden, self.ai.bias_hidden)
        np.testing.assert_array_almost_equal(loaded_ai.weights_hidden_output, self.ai.weights_hidden_output)
        np.testing.assert_array_almost_equal(loaded_ai.bias_output, self.ai.bias_output)

        # Verify architectural parameters
        self.assertEqual(loaded_ai.input_size, self.ai.input_size)
        self.assertEqual(loaded_ai.hidden_size, self.ai.hidden_size)
        self.assertEqual(loaded_ai.output_size, self.ai.output_size)

    def test_load_model_backward_compatibility_cerebellum(self):
        # Simulate an old model file with only "weights"
        old_model_data = {"weights": (np.random.rand(self.ai.input_size, self.ai.output_size) * 0.01).tolist()} # Old structure
        with open(TEST_CEREBELLUM_MODEL_PATH, 'w') as f: json.dump(old_model_data, f)

        loaded_ai = CerebellumAI(model_path=TEST_CEREBELLUM_MODEL_PATH)
        # Check that new attributes are initialized (not None and have correct shapes)
        self.assertIsNotNone(loaded_ai.weights_input_hidden, "weights_input_hidden should be initialized on backward compat load.")
        self.assertEqual(loaded_ai.weights_input_hidden.shape, (self.ai.input_size, self.ai.hidden_size))
        self.assertIsNotNone(loaded_ai.bias_hidden, "bias_hidden should be initialized on backward compat load.")
        self.assertEqual(loaded_ai.bias_hidden.shape, (1, self.ai.hidden_size))
        self.assertIsNotNone(loaded_ai.weights_hidden_output, "weights_hidden_output should be initialized on backward compat load.")
        self.assertEqual(loaded_ai.weights_hidden_output.shape, (self.ai.hidden_size, self.ai.output_size))
        self.assertIsNotNone(loaded_ai.bias_output, "bias_output should be initialized on backward compat load.")
        self.assertEqual(loaded_ai.bias_output.shape, (1, self.ai.output_size))

    def test_load_model_shape_mismatch_cerebellum(self):
        self.ai.save_model() # Save a valid new model first
        with open(TEST_CEREBELLUM_MODEL_PATH, 'r') as f: model_data = json.load(f)

        # Corrupt the shape of weights_input_hidden
        model_data['weights_input_hidden'] = [[0.01, 0.02], [0.03, 0.04]] # Example of a clearly wrong shape
        with open(TEST_CEREBELLUM_MODEL_PATH, 'w') as f: json.dump(model_data, f)

        # Get default weights from a fresh instance
        temp_ai_defaults = CerebellumAI(model_path="non_existent_temp_model.json") # Ensure it initializes defaults
        default_w_ih = temp_ai_defaults.weights_input_hidden.copy()

        loaded_ai = CerebellumAI(model_path=TEST_CEREBELLUM_MODEL_PATH) # Load the corrupted model

        # Assert that the loaded weights are not the corrupted ones and have the correct (default) shape.
        # Exact value comparison to another random init is fragile.
        self.assertEqual(loaded_ai.weights_input_hidden.shape, default_w_ih.shape, "Shape of weights_input_hidden is incorrect after loading corrupted model.")
        # Check that it's not the corrupted small array by checking a value known to be outside the corrupted array's possible values if it were padded.
        # A simpler check is that it's different from the corrupted data if it were somehow loaded and reshaped.
        # The most robust check is that load_model correctly re-initialized.
        # We can verify it's not the *exact* corrupted data by checking a few elements if we knew the padding strategy.
        # For now, shape check and ensuring it's not all zeros (if default init isn't all zeros) is a good start.
        self.assertTrue(np.any(loaded_ai.weights_input_hidden != 0) or np.all(default_w_ih == 0), "Loaded weights_input_hidden seems to be all zeros or not correctly re-initialized.")
        # A more direct test: ensure the loaded weights are different from the *attempted* corrupted load.
        # This is tricky because the corrupted data is small. The key is that _initialize_default_weights_biases was called.
        # So, we check if the loaded weights are indeed the *newly initialized default* ones.
        # This means comparing them to what a fresh instance would have.
        fresh_defaults = CerebellumAI(model_path="fresh_defaults.json") # This will init defaults
        np.testing.assert_array_almost_equal(loaded_ai.weights_input_hidden, fresh_defaults.weights_input_hidden, decimal=5, err_msg="Corrupted weights_input_hidden not reset to new default.")
        if os.path.exists("fresh_defaults.json"): os.remove("fresh_defaults.json")

# Updated Test Class for ParietalLobeAI
class TestParietalLobeAI(unittest.TestCase):
    def setUp(self):
        os.makedirs(TEST_DATA_DIR, exist_ok=True)
        # Ensure clean state before AI initialization
        if os.path.exists(TEST_PARIETAL_MODEL_PATH):
            os.remove(TEST_PARIETAL_MODEL_PATH)

        self.ai = ParietalLobeAI(model_path=TEST_PARIETAL_MODEL_PATH)
        self.ai._initialize_default_weights_biases()
        self.ai.save_model()

        self.sample_sensory_data = np.random.rand(self.ai.input_size).tolist() # input_size is 20
        self.sample_true_coords = np.random.rand(self.ai.output_size).tolist() # output_size is 3

    def tearDown(self):
        if os.path.exists(TEST_PARIETAL_MODEL_PATH):
            os.remove(TEST_PARIETAL_MODEL_PATH)

    def test_prepare_input_vector_parietal(self):
        vec = self.ai._prepare_input_vector(self.sample_sensory_data)
        self.assertEqual(vec.shape, (self.ai.input_size,))
        short_data = self.sample_sensory_data[:-2]
        vec_short = self.ai._prepare_input_vector(short_data)
        self.assertEqual(vec_short.shape, (self.ai.input_size,))
        self.assertTrue(np.all(vec_short[len(short_data):] == 0))
        long_data = self.sample_sensory_data + [0.1, 0.2]
        vec_long = self.ai._prepare_input_vector(long_data)
        self.assertEqual(vec_long.shape, (self.ai.input_size,))
        np.testing.assert_array_almost_equal(vec_long, np.array(long_data[:self.ai.input_size]), decimal=5)

    def test_prepare_target_coords_vector_parietal(self):
        vec = self.ai._prepare_target_coords_vector(self.sample_true_coords)
        self.assertEqual(vec.shape, (self.ai.output_size,))
        short_data = self.sample_true_coords[:-1]
        vec_short = self.ai._prepare_target_coords_vector(short_data)
        self.assertEqual(vec_short.shape, (self.ai.output_size,))
        self.assertTrue(np.all(vec_short[len(short_data):] == 0))
        long_data = self.sample_true_coords + [0.1]
        vec_long = self.ai._prepare_target_coords_vector(long_data)
        self.assertEqual(vec_long.shape, (self.ai.output_size,))
        np.testing.assert_array_almost_equal(vec_long, np.array(long_data[:self.ai.output_size]), decimal=5)

    def test_forward_propagate_output_shapes_parietal(self):
        input_vec_1d, hidden_layer_output, output_coords_final = self.ai._forward_propagate(self.sample_sensory_data)
        self.assertEqual(input_vec_1d.shape, (self.ai.input_size,), "Input vector shape mismatch.")
        self.assertEqual(hidden_layer_output.shape, (self.ai.hidden_size,), "Hidden layer output shape mismatch.")
        self.assertEqual(output_coords_final.shape, (self.ai.output_size,), "Output coordinates shape mismatch.")
        self.assertTrue(np.all(hidden_layer_output >= -1) and np.all(hidden_layer_output <= 1), "Hidden output out of Tanh range.")
        # Output_coords has no explicit activation, so no range check beyond shape.

    def test_process_task_output_parietal(self):
        coords = self.ai.process_task(self.sample_sensory_data)
        self.assertIsInstance(coords, list)
        self.assertEqual(len(coords), self.ai.output_size)

    def test_learn_updates_and_memory_parietal(self):
        initial_w_ih = self.ai.weights_input_hidden.copy()
        initial_b_h = self.ai.bias_hidden.copy()
        initial_w_ih = self.ai.weights_input_hidden.copy()
        initial_b_h = self.ai.bias_hidden.copy()
        initial_w_ho = self.ai.weights_hidden_output.copy()
        initial_b_o = self.ai.bias_output.copy()

        self.ai.learn(self.sample_sensory_data, self.sample_true_coords)

        self.assertFalse(np.array_equal(initial_w_ih, self.ai.weights_input_hidden), "weights_input_hidden should change after learn")
        self.assertFalse(np.array_equal(initial_b_h, self.ai.bias_hidden), "bias_hidden should change after learn")
        self.assertFalse(np.array_equal(initial_w_ho, self.ai.weights_hidden_output), "weights_hidden_output should change after learn")
        self.assertFalse(np.array_equal(initial_b_o, self.ai.bias_output), "bias_output should change after learn")

        expected_mem_sensor = self.sample_sensory_data if isinstance(self.sample_sensory_data, list) else self.sample_sensory_data.tolist()
        expected_mem_coords = self.sample_true_coords if isinstance(self.sample_true_coords, list) else self.sample_true_coords.tolist()
        self.assertIn((expected_mem_sensor, expected_mem_coords), self.ai.memory)

        self.ai.memory = []
        for i in range(self.ai.max_memory_size + 5): # Test memory limit
            self.ai.learn(self.sample_sensory_data, (np.array(self.sample_true_coords) * (i / (self.ai.max_memory_size + 4.0))).tolist())
        self.assertEqual(len(self.ai.memory), self.ai.max_memory_size)

    def test_consolidate_updates_weights_parietal(self):
        self.ai.learn(self.sample_sensory_data, self.sample_true_coords) # Populate memory
        initial_params = {
            "w_ih": self.ai.weights_input_hidden.copy(),
            "b_h": self.ai.bias_hidden.copy(),
            "w_ho": self.ai.weights_hidden_output.copy(),
            "b_o": self.ai.bias_output.copy()
        }
        self.ai.consolidate()

        self.assertFalse(np.array_equal(initial_params["w_ih"], self.ai.weights_input_hidden), "weights_input_hidden should change after consolidate")
        self.assertFalse(np.array_equal(initial_params["b_h"], self.ai.bias_hidden), "bias_hidden should change after consolidate")
        self.assertFalse(np.array_equal(initial_params["w_ho"], self.ai.weights_hidden_output), "weights_hidden_output should change after consolidate")
        self.assertFalse(np.array_equal(initial_params["b_o"], self.ai.bias_output), "bias_output should change after consolidate")

    def test_model_save_load_new_parietal_architecture(self):
        # Set known values for all attributes
        self.ai.weights_input_hidden = np.full((self.ai.input_size, self.ai.hidden_size), 0.81)
        self.ai.bias_hidden = np.full((1, self.ai.hidden_size), 0.82)
        self.ai.weights_hidden_output = np.full((self.ai.hidden_size, self.ai.output_size), 0.83)
        self.ai.bias_output = np.full((1, self.ai.output_size), 0.84)

        self.ai.save_model()
        loaded_ai = ParietalLobeAI(model_path=TEST_PARIETAL_MODEL_PATH)

        np.testing.assert_array_almost_equal(loaded_ai.weights_input_hidden, self.ai.weights_input_hidden)
        np.testing.assert_array_almost_equal(loaded_ai.bias_hidden, self.ai.bias_hidden)
        np.testing.assert_array_almost_equal(loaded_ai.weights_hidden_output, self.ai.weights_hidden_output)
        np.testing.assert_array_almost_equal(loaded_ai.bias_output, self.ai.bias_output)

        self.assertEqual(loaded_ai.input_size, self.ai.input_size)
        self.assertEqual(loaded_ai.hidden_size, self.ai.hidden_size)
        self.assertEqual(loaded_ai.output_size, self.ai.output_size)

    def test_load_model_backward_compatibility_parietal(self):
        old_model_data = {"weights": (np.random.rand(self.ai.input_size, self.ai.output_size) * 0.01).tolist()} # Old structure
        with open(TEST_PARIETAL_MODEL_PATH, 'w') as f: json.dump(old_model_data, f)

        loaded_ai = ParietalLobeAI(model_path=TEST_PARIETAL_MODEL_PATH)
        self.assertIsNotNone(loaded_ai.weights_input_hidden, "weights_input_hidden should be initialized.")
        self.assertEqual(loaded_ai.weights_input_hidden.shape, (self.ai.input_size, self.ai.hidden_size))
        self.assertIsNotNone(loaded_ai.bias_hidden, "bias_hidden should be initialized.")
        self.assertEqual(loaded_ai.bias_hidden.shape, (1, self.ai.hidden_size))
        self.assertIsNotNone(loaded_ai.weights_hidden_output, "weights_hidden_output should be initialized.")
        self.assertEqual(loaded_ai.weights_hidden_output.shape, (self.ai.hidden_size, self.ai.output_size))
        self.assertIsNotNone(loaded_ai.bias_output, "bias_output should be initialized.")
        self.assertEqual(loaded_ai.bias_output.shape, (1, self.ai.output_size))

    def test_load_model_shape_mismatch_parietal(self):
        self.ai.save_model() # Save a valid new model first
        with open(TEST_PARIETAL_MODEL_PATH, 'r') as f: model_data = json.load(f)

        model_data['weights_input_hidden'] = [[0.01, 0.02]] # Corrupt shape
        with open(TEST_PARIETAL_MODEL_PATH, 'w') as f: json.dump(model_data, f)

        temp_ai_defaults = ParietalLobeAI(model_path="non_existent_temp_model.json")
        # default_w_ih = temp_ai_defaults.weights_input_hidden.copy() # Keep for shape reference

        loaded_ai = ParietalLobeAI(model_path=TEST_PARIETAL_MODEL_PATH) # Load corrupted model

        # Check if weights were re-initialized to new defaults
        fresh_defaults = ParietalLobeAI(model_path="fresh_parietal_defaults.json")
        np.testing.assert_array_almost_equal(loaded_ai.weights_input_hidden, fresh_defaults.weights_input_hidden, decimal=5, err_msg="Corrupted parietal weights_input_hidden not reset to new default.")
        if os.path.exists("fresh_parietal_defaults.json"): os.remove("fresh_parietal_defaults.json")


if __name__ == '__main__':
    unittest.main()
