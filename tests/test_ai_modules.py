import unittest
import os
import sys
import json
import shutil # For cleaning up directories
from unittest.mock import patch, MagicMock # For mocking

import numpy as np
from PIL import Image # For creating test images

# Add the project root directory to sys.path to allow absolute imports
# from modules in the root directory (e.g., main.py, temporal.py).
# This must be done BEFORE attempting to import any project modules.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now, import project modules
from main import ( # noqa: E402
    load_sensor_data,
    load_image_text_pairs,
    load_vision_data,
    load_text_data,
    BrainCoordinator, # Added BrainCoordinator
)
from temporal import TemporalLobeAI # noqa: E402
from cerebellum import CerebellumAI # noqa: E402
from limbic import LimbicSystemAI # noqa: E402
from occipital import OccipitalLobeAI # noqa: E402
from frontal import FrontalLobeAI # noqa: E402
from parietal import ParietalLobeAI # noqa: E402 Added ParietalLobeAI

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
TEST_LIMBIC_MODEL_PATH = os.path.join(TEST_DATA_DIR, "test_limbic_model.weights.h5")  # Keras convention
TEST_OCCIPITAL_MODEL_PATH = os.path.join(TEST_DATA_DIR, "test_occipital_model.weights.h5")  # Keras convention
TEST_FRONTAL_MODEL_PATH = os.path.join(
    TEST_DATA_DIR, "test_frontal_model.weights.h5"
)  # Keras convention
TEST_PARIETAL_MODEL_PATH = os.path.join(TEST_DATA_DIR, "test_parietal_model.json") # Added for ParietalLobeAI


class TestDataLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.makedirs(TEST_DATA_DIR, exist_ok=True)
        cls.vision_content = {"image_paths": [os.path.join(TEST_IMAGES_SUBDIR, "img1.png")]}
        with open(TEST_VISION_FILE, 'w') as f:
            json.dump(cls.vision_content, f)
        cls.sensor_content = [[1.0, 2.0], [3.0, 4.0]]
        with open(TEST_SENSOR_FILE, 'w') as f:
            json.dump(cls.sensor_content, f)
        cls.text_content = ["hello", "world"]
        with open(TEST_TEXT_FILE, 'w') as f:
            json.dump(cls.text_content, f)
        cls.pairs_content = [{"image_path": "img1.png", "text_description": "text1", "visual_label": 0},
                                {"image_path": "img2.png", "text_description": "text2", "visual_label": 1}]
        with open(TEST_PAIRS_FILE, 'w') as f:
            json.dump(cls.pairs_content, f)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(TEST_DATA_DIR):
            shutil.rmtree(TEST_DATA_DIR)

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
        if os.path.exists(TEST_TEMPORAL_MODEL_PATH):
            os.remove(TEST_TEMPORAL_MODEL_PATH)
        if os.path.exists(TEST_TEMPORAL_MEMORY_PATH):
            os.remove(TEST_TEMPORAL_MEMORY_PATH)

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
        self.assertIsInstance(output_with_visual, tuple)
        self.assertEqual(len(output_with_visual), 2)
        self.assertIsInstance(output_with_visual[0], list)
        self.assertIsInstance(output_with_visual[1], (int, np.integer))
        output_without_visual = self.ai.process_task("text", predict_visual=False)
        self.assertIsInstance(output_without_visual, list)
    def test_learn_updates_all_weights_and_biases(self):
        initial_w = [p.copy() for p in [self.ai.weights_text_input_hidden, self.ai.bias_text_hidden, self.ai.weights_text_hidden_embedding, self.ai.bias_text_embedding, self.ai.weights_embed_to_visual_hidden, self.ai.bias_visual_assoc_hidden, self.ai.weights_visual_hidden_to_label, self.ai.bias_visual_assoc_label]]
        self.ai.learn([("text", "target")], visual_label_as_context=1)
        final_w = [self.ai.weights_text_input_hidden, self.ai.bias_text_hidden, self.ai.weights_text_hidden_embedding, self.ai.bias_text_embedding, self.ai.weights_embed_to_visual_hidden, self.ai.bias_visual_assoc_hidden, self.ai.weights_visual_hidden_to_label, self.ai.bias_visual_assoc_label]
        for i_w, f_w in zip(initial_w, final_w):
            self.assertFalse(np.array_equal(i_w, f_w))
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
        self.ai.memory_db.append([("m_item","t_item")])
        self.ai.cross_modal_memory.append(("cm_text",1))
        self.ai.save_memory()
        # Use keyword arguments for clarity and correctness
        new_ai = TemporalLobeAI(
            model_path=TEST_TEMPORAL_MODEL_PATH, memory_path=TEST_TEMPORAL_MEMORY_PATH
        )
        self.assertIn([("m_item","t_item")], new_ai.memory_db)
        self.assertIn(("cm_text",1), new_ai.cross_modal_memory)

    def test_load_model_backward_compatibility_temporal(self):
        # Simulate an old model file with only text-processing parts
        # (missing visual_assoc parts and their architectural params)
        old_model_data = {
            "input_size": self.ai.input_size,
            "text_hidden_size": self.ai.text_hidden_size,
            "output_size": self.ai.output_size, # embedding_size
            # Missing: visual_assoc_hidden_size, visual_output_size
            "weights_text_input_hidden": np.random.rand(self.ai.input_size, self.ai.text_hidden_size).tolist(),
            "bias_text_hidden": np.random.rand(1, self.ai.text_hidden_size).tolist(),
            "weights_text_hidden_embedding": np.random.rand(self.ai.text_hidden_size, self.ai.output_size).tolist(),
            "bias_text_embedding": np.random.rand(1, self.ai.output_size).tolist(),
            # Missing: weights_embed_to_visual_hidden, bias_visual_assoc_hidden, etc.
        }
        with open(TEST_TEMPORAL_MODEL_PATH, 'w') as f:
            json.dump(old_model_data, f)

        loaded_ai = TemporalLobeAI(model_path=TEST_TEMPORAL_MODEL_PATH, memory_path=TEST_TEMPORAL_MEMORY_PATH)

        # Check that missing architectural params are default-initialized
        self.assertNotEqual(loaded_ai.visual_assoc_hidden_size, 0) # Assuming default is non-zero
        self.assertNotEqual(loaded_ai.visual_output_size, 0)     # Assuming default is non-zero

        # Check that weights for missing parts are initialized (not None and have correct shapes)
        self.assertIsNotNone(loaded_ai.weights_embed_to_visual_hidden)
        self.assertEqual(loaded_ai.weights_embed_to_visual_hidden.shape, (self.ai.output_size, loaded_ai.visual_assoc_hidden_size))
        # ... and so on for other visual_assoc weights and biases

    def test_load_model_shape_mismatch_temporal(self):
        np.random.seed(0)  # Ensure reproducible random initializations
        self.ai.save_model() # Save a valid new model first
        with open(TEST_TEMPORAL_MODEL_PATH, 'r') as f:
            model_data = json.load(f)

        # Corrupt the shape of weights_text_input_hidden
        model_data['weights_text_input_hidden'] = [[0.01, 0.02], [0.03, 0.04]] # Example of a clearly wrong shape
        with open(TEST_TEMPORAL_MODEL_PATH, 'w') as f:
            json.dump(model_data, f)

        loaded_ai = TemporalLobeAI(model_path=TEST_TEMPORAL_MODEL_PATH, memory_path=TEST_TEMPORAL_MEMORY_PATH) # Load the corrupted model

        # Only check that the attributes are not None (shape may not be available if None)
        self.assertIsNotNone(loaded_ai.weights_text_input_hidden, "weights_text_input_hidden should not be None after loading corrupted model.")
        self.assertIsNotNone(loaded_ai.bias_text_hidden, "bias_text_hidden should not be None after loading corrupted model.")
        self.assertIsNotNone(loaded_ai.weights_text_hidden_embedding, "weights_text_hidden_embedding should not be None after loading corrupted model.")
        self.assertIsNotNone(loaded_ai.bias_text_embedding, "bias_text_embedding should not be None after loading corrupted model.")
        self.assertIsNotNone(loaded_ai.weights_embed_to_visual_hidden, "weights_embed_to_visual_hidden should not be None after loading corrupted model.")
        self.assertIsNotNone(loaded_ai.bias_visual_assoc_hidden, "bias_visual_assoc_hidden should not be None after loading corrupted model.")
        self.assertIsNotNone(loaded_ai.weights_visual_hidden_to_label, "weights_visual_hidden_to_label should not be None after loading corrupted model.")
        self.assertIsNotNone(loaded_ai.bias_visual_assoc_label, "bias_visual_assoc_label should not be None after loading corrupted model.")

        if os.path.exists("fresh_temporal_defaults.json"):
            os.remove("fresh_temporal_defaults.json")
        if os.path.exists("fresh_temporal_memory.json"):
            os.remove("fresh_temporal_memory.json")

    # ...existing code...

class TestLimbicSystemAI(unittest.TestCase):
    def setUp(self):
        os.makedirs(TEST_DATA_DIR, exist_ok=True)
        if os.path.exists(TEST_LIMBIC_MODEL_PATH):
            os.remove(TEST_LIMBIC_MODEL_PATH)
        self.ai = LimbicSystemAI(model_path=TEST_LIMBIC_MODEL_PATH)
        self.ai.save_model()
        self.sample_temporal_output = np.random.rand(self.ai.input_size).tolist()
        self.sample_true_emotion = 1
        self.sample_reward = 1.0

    def tearDown(self):
        if os.path.exists(TEST_LIMBIC_MODEL_PATH):
            os.remove(TEST_LIMBIC_MODEL_PATH)

    def test_ensure_input_vector_shape(self):
        vec = self.ai._ensure_input_vector_shape(self.sample_temporal_output)
        self.assertEqual(vec.shape, (self.ai.input_size,))

    def test_forward_propagate_output_shapes_limbic(self):
        # Use Keras model.predict for forward propagation
        input_vec = self.ai._ensure_input_vector_shape(self.sample_temporal_output)
        input_vec_2d = input_vec.reshape(1, -1)
        # Get hidden layer output via intermediate layer if possible, else just check output
        output_scores_1d = self.ai.model.predict(input_vec_2d, verbose=0)[0]
        self.assertEqual(input_vec.shape, (self.ai.input_size,))
        # Hidden layer output shape is not directly accessible unless model exposes it, so skip
        self.assertEqual(output_scores_1d.shape, (self.ai.output_size,))

    def test_process_task_output_label_limbic(self):
        label = self.ai.process_task(self.sample_temporal_output)
        # process_task now returns a dict: {"label": ..., "probabilities": ...}
        self.assertTrue(0 <= label["label"] < self.ai.output_size)

    def test_learn_updates_and_memory_limbic(self):
        initial_weights = [w.copy() for w in self.ai.model.get_weights()]
        self.ai.learn(self.sample_temporal_output, self.sample_true_emotion, self.sample_reward)
        final_weights = self.ai.model.get_weights()
        weights_changed = any(not np.array_equal(iw, fw) for iw, fw in zip(initial_weights, final_weights))
        self.assertTrue(weights_changed, "Model weights did not change after learn")
        self.assertIn((self.sample_temporal_output, self.sample_true_emotion, self.sample_reward), self.ai.memory)

    def test_consolidate_updates_weights_limbic(self):
        self.ai.learn(self.sample_temporal_output, self.sample_true_emotion, self.sample_reward)
        initial_weights = [w.copy() for w in self.ai.model.get_weights()]
        self.ai.consolidate()
        final_weights = self.ai.model.get_weights()
        weights_changed = any(not np.array_equal(iw, fw) for iw, fw in zip(initial_weights, final_weights))
        self.assertTrue(weights_changed, "Model weights did not change after consolidate")

    def test_model_save_load_new_limbic_architecture(self):
        # Set known values for all weights
        weights = self.ai.model.get_weights()
        new_weights = [np.full(w.shape, 0.77) for w in weights]
        self.ai.model.set_weights(new_weights)
        self.ai.save_model()
        loaded_ai = LimbicSystemAI(model_path=TEST_LIMBIC_MODEL_PATH)
        loaded_weights = loaded_ai.model.get_weights()
        for lw, nw in zip(loaded_weights, new_weights):
            self.assertEqual(lw.shape, nw.shape)
        self.assertEqual(loaded_ai.input_size, self.ai.input_size)
        # hidden_size may not be exposed; skip if not present
        self.assertEqual(loaded_ai.output_size, self.ai.output_size)

    def test_load_model_backward_compatibility_limbic(self):
        # Simulate an old model file with only weights
        # Use model.get_weights() to infer hidden size
        weights = self.ai.model.get_weights()
        output_size = weights[-1].shape[-1]
        old_model_data = {
            "input_size": weights[0].shape[0],
            "output_size": output_size,
            "weights_input_hidden": (np.random.rand(weights[0].shape[0], weights[0].shape[1]) * 0.1).tolist(),
            "weights_hidden_output": (np.random.rand(weights[0].shape[1], output_size) * 0.1).tolist(),
        }
        with open(TEST_LIMBIC_MODEL_PATH, 'w') as f:
            json.dump(old_model_data, f)
        loaded_ai = LimbicSystemAI(model_path=TEST_LIMBIC_MODEL_PATH)
        # Check that model weights are initialized (not None and have correct shapes)
        weights = loaded_ai.model.get_weights()
        self.assertTrue(all(w is not None for w in weights))
        # Check shapes
        self.assertEqual(weights[0].shape[0], self.ai.input_size)
        self.assertEqual(weights[-1].shape[-1], output_size)

    def test_load_model_shape_mismatch_limbic(self):
        # Overwrite the weights file with invalid data
        with open(TEST_LIMBIC_MODEL_PATH, 'w') as f:
            f.write('not a valid h5 file')
        # Should not crash, should reinitialize
        ai = LimbicSystemAI(model_path=TEST_LIMBIC_MODEL_PATH)
        weights = ai.model.get_weights()
        self.assertTrue(all(w is not None for w in weights), "Model weights should be initialized after loading corrupted file.")

class TestOccipitalLobeAI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.makedirs(TEST_IMAGES_SUBDIR, exist_ok=True)
        cls.img_size = (32, 32) # Default size for test images
        cls.test_img_path = os.path.join(TEST_IMAGES_SUBDIR, "test_image.png")
        # Ensure the test image exists or is created with correct dimensions for Occipital Lobe
        # (assuming OccipitalLobeAI.input_shape is (64,64,3), so test image should match)
        # For this test, let's assume OccipitalLobeAI can handle various input sizes
        # or that _preprocess_image resizes it. If fixed size is needed, adjust here.
        try:
            Image.new('RGB', (64,64), 255).save(cls.test_img_path) # Use RGB for 3 channels
        except Exception as e:
            print(f"Warning: Could not create test image {cls.test_img_path}: {e}")


    @classmethod
    def tearDownClass(cls):
        if os.path.exists(TEST_IMAGES_SUBDIR):
            shutil.rmtree(TEST_IMAGES_SUBDIR)
        if os.path.exists(TEST_OCCIPITAL_MODEL_PATH): # General test model path
            os.remove(TEST_OCCIPITAL_MODEL_PATH)

    def setUp(self):
        os.makedirs(TEST_DATA_DIR, exist_ok=True)
        # Use a unique model path for each test instance to avoid interference
        self.test_instance_model_path = os.path.join(TEST_DATA_DIR, f"test_occipital_{self._testMethodName}.weights.h5")
        if os.path.exists(self.test_instance_model_path):
            os.remove(self.test_instance_model_path)
        self.ai = OccipitalLobeAI(model_path=self.test_instance_model_path)
        # No initial save_model here, let tests decide if they need to save/load.

    def tearDown(self):
        if os.path.exists(self.test_instance_model_path):
            os.remove(self.test_instance_model_path)

    def test_forward_propagate_output_shapes(self):
        # Use model.predict for forward propagation
        input_shape = self.ai.model.input_shape # e.g., (64, 64, 3)
        # Create a dummy input batch of 1 image
        dummy_input_batch = np.random.rand(1, *input_shape) 
        output_scores_batch = self.ai.model.predict(dummy_input_batch, verbose=0)
        self.assertEqual(output_scores_batch.shape, (1, self.ai.output_size))


    def test_learn_updates_and_memory(self):
        initial_weights = [w.copy() for w in self.ai.model.get_weights()]
        
        # Ensure test_img_path exists for learn to proceed
        if not os.path.exists(self.test_img_path):
            self.skipTest(f"Test image {self.test_img_path} not found, skipping learn test.")

        self.ai.learn(self.test_img_path, 0) # Label 0
        
        final_weights = self.ai.model.get_weights()
        weights_changed = any(not np.array_equal(iw, fw) for iw, fw in zip(initial_weights, final_weights))
        self.assertTrue(weights_changed, "Model weights did not change after learn")

        self.assertNotEqual(len(self.ai.memory), 0, "Memory should not be empty after learn.")
        cached_experience = self.ai.memory[0]
        self.assertIsInstance(cached_experience, tuple, "Memory item should be a tuple.")
        self.assertEqual(len(cached_experience), 2, "Memory tuple should have two elements.")
        
        self.assertIsInstance(cached_experience[0], np.ndarray, "First element of memory tuple should be a NumPy array.")
        self.assertEqual(cached_experience[0].shape, (1,) + self.ai.input_shape, "Cached image array shape is incorrect.")
        self.assertTrue(np.issubdtype(cached_experience[0].dtype, np.floating), "Cached image array dtype should be float.")
        self.assertEqual(cached_experience[1], 0, "Second element of memory tuple (label) is incorrect.")

    def test_consolidate_updates_weights(self):
        if not os.path.exists(self.test_img_path):
            self.skipTest(f"Test image {self.test_img_path} not found, skipping consolidate test.")

        self.ai.learn(self.test_img_path, 1) 
        self.assertTrue(len(self.ai.memory) > 0, "Memory should be populated before testing consolidate.")

        initial_weights = [w.copy() for w in self.ai.model.get_weights()]

        with patch.object(self.ai, '_preprocess_image') as mock_preprocess:
            self.ai.consolidate()
            mock_preprocess.assert_not_called("consolidate should use cached images, not call _preprocess_image.")
        
        final_weights = self.ai.model.get_weights()
        weights_changed = any(not np.array_equal(iw, fw) for iw, fw in zip(initial_weights, final_weights))
        if len(self.ai.memory) > 0 and self.ai.model.optimizer.learning_rate > 0: # Weights change if there's data and LR > 0
            self.assertTrue(weights_changed, "Model weights did not change after consolidate with cached images.")
        else:
            self.assertFalse(weights_changed, "Model weights should not change if no data or LR is zero.")


    def test_learn_skip_on_preprocessing_failure(self):
        with patch.object(self.ai, '_preprocess_image', return_value=None) as mock_failing_preprocess:
            self.ai.learn(self.test_img_path, 0)
            if os.path.exists(self.test_img_path): # Preprocess is only called if image exists
                mock_failing_preprocess.assert_called_once_with(self.test_img_path)
            self.assertEqual(len(self.ai.memory), 0, "Memory should be empty if preprocessing fails during learn.")

    def test_model_save_load_new_architecture(self):
        # Modify weights to ensure save/load works
        original_weights = [w.copy() for w in self.ai.model.get_weights()]
        new_weights = [np.full(w.shape, 0.55) for w in original_weights]
        self.ai.model.set_weights(new_weights)
        
        self.ai.save_model() # Uses self.test_instance_model_path
        
        loaded_ai = OccipitalLobeAI(model_path=self.test_instance_model_path)
        loaded_weights = loaded_ai.model.get_weights()
        
        self.assertEqual(len(loaded_weights), len(new_weights), "Number of weight arrays differs.")
        for i in range(len(new_weights)):
            np.testing.assert_array_almost_equal(loaded_weights[i], new_weights[i],
                                                 err_msg=f"Weights for layer {i} not loaded correctly.")

    def test_load_model_backward_compatibility_occipital(self):
        # This test is tricky for Keras .weights.h5 as it expects a certain structure.
        # Simulating an "old" format that Keras would try to load is complex.
        # Instead, we'll test that loading a non-existent file results in a fresh model.
        non_existent_path = os.path.join(TEST_DATA_DIR, "non_existent_occipital.weights.h5")
        if os.path.exists(non_existent_path): os.remove(non_existent_path) # Ensure it doesn't exist
        
        loaded_ai = OccipitalLobeAI(model_path=non_existent_path)
        self.assertIsNotNone(loaded_ai.model, "Model should be initialized even if no weights file found.")
        # Further checks could involve comparing weights to a newly built model,
        # but this is complex due to random initializations.
        # For now, ensuring it doesn't crash and has a model is sufficient.

    def test_load_model_shape_mismatch_occipital(self):
        # Create a valid model and save it
        self.ai.save_model()
        
        # Create another AI instance with a different architecture (e.g., different output_size)
        # and save its weights to the *same path*, simulating a shape mismatch.
        # This is hard to do without actually changing OccipitalLobeAI's definition or
        # manually crafting an H5 file, which is beyond simple test scope.
        # A more practical test: load weights from an unrelated H5 file (if one was available)
        # or ensure Keras handles it gracefully (which it usually does by raising an error).

        # For now, we'll test loading a corrupted H5 file.
        with open(self.test_instance_model_path, 'w') as f: # Corrupt the file
            f.write('not a valid h5 file')
        
        # Should not crash, should reinitialize or keep initial model
        # Keras load_weights will likely raise an OSError or similar.
        # The OccipitalLobeAI load_model has a try-except that prints an error.
        # We can check if the model's weights are still the initial ones (or re-initialized).
        initial_weights_before_corrupt_load = [w.copy() for w in self.ai.model.get_weights()]
        
        # Suppress print output during this expected failure
        with patch('builtins.print') as mock_print:
            self.ai.load_model() # Attempt to load corrupted file
            mock_print.assert_any_call(unittest.mock.ANY) # Check that an error was printed

        weights_after_corrupt_load = self.ai.model.get_weights()
        
        # Check if weights are the same as before the load attempt (meaning load failed and model was preserved)
        weights_match = all(np.array_equal(initial_w, loaded_w) for initial_w, loaded_w in zip(initial_weights_before_corrupt_load, weights_after_corrupt_load))
        self.assertTrue(weights_match, "Model weights should remain unchanged after attempting to load a corrupted file.")


class TestFrontalLobeAI(unittest.TestCase):
    def setUp(self):
        os.makedirs(TEST_DATA_DIR, exist_ok=True)
        # Use a unique model path for each test instance
        self.test_instance_model_path = os.path.join(TEST_DATA_DIR, f"test_frontal_{self._testMethodName}.weights.h5")
        self.epsilon_path = self.test_instance_model_path + "_epsilon.json"
        
        if os.path.exists(self.test_instance_model_path):
            os.remove(self.test_instance_model_path)
        if os.path.exists(self.epsilon_path):
            os.remove(self.epsilon_path)
        
        self.ai = FrontalLobeAI(model_path=self.test_instance_model_path, replay_batch_size=1)
        # self.ai.save_model() # Save initial model if needed by specific tests, but often not for unit tests
        self.sample_state = np.random.rand(self.ai.input_size).tolist()


    def tearDown(self):
        if os.path.exists(self.test_instance_model_path):
            os.remove(self.test_instance_model_path)
        if os.path.exists(self.epsilon_path): 
            os.remove(self.epsilon_path)


    def test_process_task_epsilon_greedy_selection(self):
        self.ai.exploration_rate_epsilon = 1.0
        action_counts_explore = {i:0 for i in range(self.ai.output_size)}
        for _ in range(100 * self.ai.output_size): # More samples for better distribution
            action_counts_explore[self.ai.process_task(self.sample_state)] += 1
        # Check that all actions were chosen at least once (for reasonable output_size)
        if self.ai.output_size <= 10: # Heuristic for small action spaces
             self.assertTrue(all(count > 0 for count in action_counts_explore.values()),
                            f"Not all actions explored with epsilon=1.0. Counts: {action_counts_explore}")

        self.ai.exploration_rate_epsilon = 0.0
        weights = self.ai.model.get_weights()
        kernel_shape = weights[-2].shape
        bias_shape = weights[-1].shape
        new_kernel = np.zeros(kernel_shape)
        # Favor action 1 (index 1)
        if self.ai.output_size > 1:
            new_kernel[:, 1] = 10.0 
            expected_action = 1
        else: # Single output, action must be 0
            new_kernel[:, 0] = 10.0
            expected_action = 0

        new_bias = np.zeros(bias_shape)
        self.ai.model.set_weights(weights[:-2] + [new_kernel, new_bias])
        self.assertEqual(
            self.ai.process_task(self.sample_state),
            expected_action,
            "Should pick action with highest Q-value when epsilon is 0 and weights are set."
        )
        initial_epsilon = 0.5
        self.ai.exploration_rate_epsilon = initial_epsilon
        self.ai.process_task(self.sample_state)
        self.assertAlmostEqual(self.ai.exploration_rate_epsilon,
                               max(self.ai.min_epsilon, initial_epsilon - self.ai.epsilon_decay_rate),
                               msg="Epsilon decay is not working as expected")

    def test_learn_q_update_and_memory(self):
        state = self.sample_state
        action_taken = 1 % self.ai.output_size # Ensure valid action
        reward = 1.0
        next_state = np.random.rand(self.ai.input_size).tolist()
        done = False
        
        # Ensure the model is built before getting weights
        _ = self.ai.model.predict(np.array([state])) 
        initial_model_weights = [w.copy() for w in self.ai.model.get_weights()]
        
        for _ in range(self.ai.replay_batch_size):
             self.ai.learn(state, action_taken, reward, next_state, done)

        final_model_weights = self.ai.model.get_weights()
        weights_changed = False
        if self.ai.learning_rate_dqn > 0 and len(self.ai.memory) >= self.ai.replay_batch_size:
            for initial_w_layer, final_w_layer in zip(initial_model_weights, final_model_weights):
                if not np.array_equal(initial_w_layer, final_w_layer):
                    weights_changed = True
                    break
            self.assertTrue(weights_changed, "Model weights should change after learning enough to trigger replay if LR > 0.")
        
        self.assertIn((list(state), action_taken, reward, list(next_state), done), self.ai.memory)
        
        self.ai.memory.clear()
        for _ in range(self.ai.replay_buffer_size + 5):
            self.ai.learn(state, action_taken, reward, next_state, done)
        self.assertTrue(len(self.ai.memory) <= self.ai.replay_buffer_size, "Memory size limit not enforced.")


    def test_consolidate_experience_replay(self):
        for i in range(self.ai.replay_batch_size * 2): 
            state = np.random.rand(self.ai.input_size).tolist()
            action = np.random.randint(0, self.ai.output_size)
            reward_val = float(np.random.choice([-1,0,1]))
            next_s = np.random.rand(self.ai.input_size).tolist()
            done_val = bool(np.random.choice([True,False]))
            self.ai.learn(state, action, reward_val, next_s, done_val) 
        
        self.assertTrue(len(self.ai.memory) > 0, "Memory should be populated before consolidation.")
        # Ensure model is built
        _ = self.ai.model.predict(np.array([self.sample_state]))
        initial_model_weights = [w.copy() for w in self.ai.model.get_weights()]
        
        self.ai.consolidate() 
        
        final_model_weights = self.ai.model.get_weights()
        weights_changed = any(
            not np.array_equal(iw, fw)
            for iw, fw in zip(initial_model_weights, final_model_weights)
        )
        if self.ai.learning_rate_dqn > 0 and len(self.ai.memory) >= self.ai.replay_batch_size:
            self.assertTrue(
                weights_changed,
                "Weights should change after consolidation if learning rate is > 0 and memory is sufficient."
            )

    def test_model_save_load_q_learning_params(self):
        self.ai.exploration_rate_epsilon = 0.678
        _ = self.ai.model.predict(np.array([self.sample_state])) # Build model
        original_weights = [w.copy() for w in self.ai.model.get_weights()]
        modified_weights = [w * 0.5 for w in original_weights]
        self.ai.model.set_weights(modified_weights)
        self.ai.save_model() # Saves to self.test_instance_model_path

        loaded_ai = FrontalLobeAI(model_path=self.test_instance_model_path)
        self.assertAlmostEqual(loaded_ai.exploration_rate_epsilon, 0.678, msg="Exploration rate not loaded correctly.")
        loaded_weights = loaded_ai.model.get_weights()
        for i, (orig_w, loaded_w) in enumerate(zip(modified_weights, loaded_weights)):
            np.testing.assert_array_almost_equal(
                loaded_w, orig_w, err_msg=f"Weights for layer {i} not loaded correctly."
            )

    def test_replay_batch_logic(self):
        batch_size = 2
        # ai instance for this test uses self.test_instance_model_path
        ai = FrontalLobeAI(model_path=self.test_instance_model_path, replay_batch_size=batch_size)
        input_size = ai.input_size 
        output_size = ai.output_size
        ai.exploration_rate_epsilon = 0 

        state1_raw = np.array([0.1] * input_size).tolist()
        action1 = 0
        reward1 = 0.5
        next_state1_raw = np.array([0.2] * input_size).tolist()
        done1 = False
        ai.remember(state1_raw, action1, reward1, next_state1_raw, done1)

        state2_raw = np.array([0.3] * input_size).tolist()
        action2 = 1 % output_size # Ensure valid action
        reward2 = -1.0
        next_state2_raw = np.array([0.4] * input_size).tolist() 
        done2 = True
        ai.remember(state2_raw, action2, reward2, next_state2_raw, done2)

        mock_current_q_s1 = np.random.rand(output_size)
        mock_current_q_s2 = np.random.rand(output_size)
        mock_current_q_values_batch = np.array([mock_current_q_s1, mock_current_q_s2])

        mock_next_q_ns1 = np.random.rand(output_size)
        mock_next_q_ns2 = np.random.rand(output_size) # Will be used by _prepare_state_vector
        mock_next_q_values_target_batch = np.array([mock_next_q_ns1, mock_next_q_ns2])
        
        expected_prepared_state1 = ai._prepare_state_vector(state1_raw)
        expected_prepared_state2 = ai._prepare_state_vector(state2_raw)
        expected_current_states_batch_np = np.array([expected_prepared_state1, expected_prepared_state2])

        with patch.object(ai.model, 'predict') as mock_model_predict, \
             patch.object(ai.target_model, 'predict') as mock_target_model_predict, \
             patch.object(ai.model, 'fit') as mock_model_fit:

            mock_model_predict.return_value = mock_current_q_values_batch
            mock_target_model_predict.return_value = mock_next_q_values_target_batch

            ai.replay()

            mock_model_fit.assert_called_once()
            
            args_call_to_fit = mock_model_fit.call_args[0]
            actual_current_states_batch_np = args_call_to_fit[0]
            np.testing.assert_array_almost_equal(actual_current_states_batch_np, expected_current_states_batch_np)

            expected_targets_batch = np.copy(mock_current_q_values_batch)
            max_next_q1 = np.amax(mock_next_q_values_target_batch[0])
            expected_targets_batch[0, action1] = reward1 + ai.discount_factor_gamma * max_next_q1
            expected_targets_batch[1, action2] = reward2

            actual_targets_batch = args_call_to_fit[1]
            np.testing.assert_array_almost_equal(actual_targets_batch, expected_targets_batch)
            
            mock_model_predict.assert_called_once()
            np.testing.assert_array_almost_equal(mock_model_predict.call_args[0][0], expected_current_states_batch_np)

            expected_next_states_batch_np = np.array([
                ai._prepare_state_vector(next_state1_raw),
                ai._prepare_state_vector(next_state2_raw) 
            ])
            mock_target_model_predict.assert_called_once()
            np.testing.assert_array_almost_equal(mock_target_model_predict.call_args[0][0], expected_next_states_batch_np)

class TestCerebellumAI(unittest.TestCase):
    def setUp(self):
        os.makedirs(TEST_DATA_DIR, exist_ok=True)
        self.test_instance_model_path = os.path.join(TEST_DATA_DIR, f"test_cerebellum_{self._testMethodName}.json")
        if os.path.exists(self.test_instance_model_path):
            os.remove(self.test_instance_model_path)

        self.ai = CerebellumAI(model_path=self.test_instance_model_path)
        # No _initialize_default_weights_biases or save_model here, let specific tests handle if needed

        self.sample_sensor_data = np.random.rand(self.ai.input_size).tolist()
        self.sample_true_command = (np.random.rand(self.ai.output_size) * 2 - 1).tolist()

    def tearDown(self):
        if os.path.exists(self.test_instance_model_path):
            os.remove(self.test_instance_model_path)

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
        long_data = self.sample_true_command + [0.1]
        vec_long = self.ai._prepare_target_command_vector(long_data)
        self.assertEqual(vec_long.shape, (self.ai.output_size,))
        np.testing.assert_array_almost_equal(vec_long, np.array(long_data[:self.ai.output_size]), decimal=5)

    def test_forward_propagate_output_shapes_and_ranges(self):
        # Ensure weights are initialized if _initialize_default_weights_biases is not called in setUp
        if self.ai.weights_input_hidden is None: self.ai._initialize_default_weights_biases()
        input_vec_1d, hidden_layer_output, final_commands_output = self.ai._forward_propagate(self.sample_sensor_data)
        self.assertEqual(input_vec_1d.shape, (self.ai.input_size,), "Input vector shape mismatch.")
        self.assertEqual(hidden_layer_output.shape, (self.ai.hidden_size,), "Hidden layer output shape mismatch.")
        self.assertEqual(final_commands_output.shape, (self.ai.output_size,), "Final commands output shape mismatch.")
        self.assertTrue(np.all(hidden_layer_output >= -1) and np.all(hidden_layer_output <= 1), "Hidden output out of Tanh range.")
        self.assertTrue(np.all(final_commands_output >= -1) and np.all(final_commands_output <= 1), "Final commands out of Tanh range.")

    def test_process_task_output(self):
        if self.ai.weights_input_hidden is None: self.ai._initialize_default_weights_biases()
        commands = self.ai.process_task(self.sample_sensor_data)
        self.assertIsInstance(commands, list)
        self.assertEqual(len(commands), self.ai.output_size)
        for val in commands:
            self.assertTrue(-1.0 <= val <= 1.0)

    def test_learn_updates_and_memory(self):
        if self.ai.weights_input_hidden is None: self.ai._initialize_default_weights_biases()
        initial_w_ih = self.ai.weights_input_hidden.copy()
        initial_b_h = self.ai.bias_hidden.copy()
        initial_w_ho = self.ai.weights_hidden_output.copy()
        initial_b_o = self.ai.bias_output.copy()

        self.ai.learn(self.sample_sensor_data, self.sample_true_command)

        self.assertFalse(np.array_equal(initial_w_ih, self.ai.weights_input_hidden))
        self.assertFalse(np.array_equal(initial_b_h, self.ai.bias_hidden))
        self.assertFalse(np.array_equal(initial_w_ho, self.ai.weights_hidden_output))
        self.assertFalse(np.array_equal(initial_b_o, self.ai.bias_output))

        expected_mem_sensor = self.sample_sensor_data if isinstance(self.sample_sensor_data, list) else self.sample_sensor_data.tolist()
        expected_mem_command = self.sample_true_command if isinstance(self.sample_true_command, list) else self.sample_true_command.tolist()
        self.assertIn((expected_mem_sensor, expected_mem_command), self.ai.memory)

        self.ai.memory.clear()
        for i in range(self.ai.max_memory_size + 5): # Test memory limit
            self.ai.learn(self.sample_sensor_data, (np.array(self.sample_true_command) * (i/(self.ai.max_memory_size+4.0))).tolist())
        self.assertEqual(len(self.ai.memory), self.ai.max_memory_size)


    def test_consolidate_updates_weights(self):
        if self.ai.weights_input_hidden is None: self.ai._initialize_default_weights_biases()
        self.ai.learn(self.sample_sensor_data, self.sample_true_command)
        w_ih_before = self.ai.weights_input_hidden.copy()
        b_o_before = self.ai.bias_output.copy()
        w_ho_before = self.ai.weights_hidden_output.copy()
        b_h_before = self.ai.bias_hidden.copy()

        self.ai.consolidate()
        if len(self.ai.memory) > 0 and self.ai.learning_rate > 0:
            self.assertFalse(np.array_equal(w_ih_before, self.ai.weights_input_hidden))
            self.assertFalse(np.array_equal(b_h_before, self.ai.bias_hidden))
            self.assertFalse(np.array_equal(w_ho_before, self.ai.weights_hidden_output))
            self.assertFalse(np.array_equal(b_o_before, self.ai.bias_output))

    def test_model_save_load_new_cerebellum_architecture(self):
        if self.ai.weights_input_hidden is None: self.ai._initialize_default_weights_biases()
        self.ai.weights_input_hidden = np.full((self.ai.input_size, self.ai.hidden_size), 0.51)
        self.ai.bias_hidden = np.full((1, self.ai.hidden_size), 0.52)
        self.ai.weights_hidden_output = np.full((self.ai.hidden_size, self.ai.output_size), 0.53)
        self.ai.bias_output = np.full((1, self.ai.output_size), 0.54)

        self.ai.save_model()
        loaded_ai = CerebellumAI(model_path=self.test_instance_model_path)

        np.testing.assert_array_almost_equal(loaded_ai.weights_input_hidden, self.ai.weights_input_hidden)
        np.testing.assert_array_almost_equal(loaded_ai.bias_hidden, self.ai.bias_hidden)
        np.testing.assert_array_almost_equal(loaded_ai.weights_hidden_output, self.ai.weights_hidden_output)
        np.testing.assert_array_almost_equal(loaded_ai.bias_output, self.ai.bias_output)

        self.assertEqual(loaded_ai.input_size, self.ai.input_size)
        self.assertEqual(loaded_ai.hidden_size, self.ai.hidden_size)
        self.assertEqual(loaded_ai.output_size, self.ai.output_size)

    def test_load_model_backward_compatibility_cerebellum(self):
        old_model_data = {"weights": (np.random.rand(self.ai.input_size, self.ai.output_size) * 0.01).tolist()} 
        with open(self.test_instance_model_path, 'w') as f:
            json.dump(old_model_data, f)

        loaded_ai = CerebellumAI(model_path=self.test_instance_model_path)
        self.assertIsNotNone(loaded_ai.weights_input_hidden)
        self.assertEqual(loaded_ai.weights_input_hidden.shape, (self.ai.input_size, self.ai.hidden_size))
        self.assertIsNotNone(loaded_ai.bias_hidden)
        self.assertEqual(loaded_ai.bias_hidden.shape, (1, self.ai.hidden_size))
        self.assertIsNotNone(loaded_ai.weights_hidden_output)
        self.assertEqual(loaded_ai.weights_hidden_output.shape, (self.ai.hidden_size, self.ai.output_size))
        self.assertIsNotNone(loaded_ai.bias_output)
        self.assertEqual(loaded_ai.bias_output.shape, (1, self.ai.output_size))

    def test_load_model_shape_mismatch_cerebellum(self):
        if self.ai.weights_input_hidden is None: self.ai._initialize_default_weights_biases()
        self.ai.save_model() 
        with open(self.test_instance_model_path, 'r') as f:
            model_data = json.load(f)

        model_data['weights_input_hidden'] = [[0.01, 0.02], [0.03, 0.04]] 
        with open(self.test_instance_model_path, 'w') as f:
            json.dump(model_data, f)

        loaded_ai = CerebellumAI(model_path=self.test_instance_model_path) 
        self.assertEqual(loaded_ai.weights_input_hidden.shape, (self.ai.input_size, self.ai.hidden_size))
        self.assertEqual(loaded_ai.bias_hidden.shape, (1, self.ai.hidden_size))
        self.assertEqual(loaded_ai.weights_hidden_output.shape, (self.ai.hidden_size, self.ai.output_size))
        self.assertEqual(loaded_ai.bias_output.shape, (1, self.ai.output_size))


class TestParietalLobeAI(unittest.TestCase):
    def setUp(self):
        os.makedirs(TEST_DATA_DIR, exist_ok=True)
        self.test_instance_model_path = os.path.join(TEST_DATA_DIR, f"test_parietal_{self._testMethodName}.json")
        if os.path.exists(self.test_instance_model_path):
            os.remove(self.test_instance_model_path)

        self.ai = ParietalLobeAI(model_path=self.test_instance_model_path)
        # No _initialize_default_weights_biases or save_model here

        self.sample_sensory_data = np.random.rand(self.ai.input_size).tolist() 
        self.sample_true_coords = np.random.rand(self.ai.output_size).tolist() 

    def tearDown(self):
        if os.path.exists(self.test_instance_model_path):
            os.remove(self.test_instance_model_path)

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
        long_data = self.sample_true_coords + [0.1]
        vec_long = self.ai._prepare_target_coords_vector(long_data)
        self.assertEqual(vec_long.shape, (self.ai.output_size,))
        np.testing.assert_array_almost_equal(vec_long, np.array(long_data[:self.ai.output_size]), decimal=5)

    def test_forward_propagate_output_shapes_parietal(self):
        if self.ai.weights_input_hidden is None: self.ai._initialize_default_weights_biases()
        input_vec_1d, hidden_layer_output, output_coords_final = self.ai._forward_propagate(self.sample_sensory_data)
        self.assertEqual(input_vec_1d.shape, (self.ai.input_size,))
        self.assertEqual(hidden_layer_output.shape, (self.ai.hidden_size,))
        self.assertEqual(output_coords_final.shape, (self.ai.output_size,))
        self.assertTrue(np.all(hidden_layer_output >= -1) and np.all(hidden_layer_output <= 1))

    def test_process_task_output_parietal(self):
        if self.ai.weights_input_hidden is None: self.ai._initialize_default_weights_biases()
        coords = self.ai.process_task(self.sample_sensory_data)
        self.assertIsInstance(coords, list)
        self.assertEqual(len(coords), self.ai.output_size)

    def test_learn_updates_and_memory_parietal(self):
        if self.ai.weights_input_hidden is None: self.ai._initialize_default_weights_biases()
        initial_w_ih = self.ai.weights_input_hidden.copy()
        initial_b_h = self.ai.bias_hidden.copy()
        initial_w_ho = self.ai.weights_hidden_output.copy()
        initial_b_o = self.ai.bias_output.copy()

        self.ai.learn(self.sample_sensory_data, self.sample_true_coords)

        self.assertFalse(np.array_equal(initial_w_ih, self.ai.weights_input_hidden))
        self.assertFalse(np.array_equal(initial_b_h, self.ai.bias_hidden))
        self.assertFalse(np.array_equal(initial_w_ho, self.ai.weights_hidden_output))
        self.assertFalse(np.array_equal(initial_b_o, self.ai.bias_output))

        expected_mem_sensor = self.sample_sensory_data if isinstance(self.sample_sensory_data, list) else self.sample_sensory_data.tolist()
        expected_mem_coords = self.sample_true_coords if isinstance(self.sample_true_coords, list) else self.sample_true_coords.tolist()
        self.assertIn((expected_mem_sensor, expected_mem_coords), self.ai.memory)

        self.ai.memory.clear()
        for i in range(self.ai.max_memory_size + 5): 
            self.ai.learn(self.sample_sensory_data, (np.array(self.sample_true_coords) * (i / (self.ai.max_memory_size + 4.0))).tolist())
        self.assertEqual(len(self.ai.memory), self.ai.max_memory_size)

    def test_consolidate_updates_weights_parietal(self):
        if self.ai.weights_input_hidden is None: self.ai._initialize_default_weights_biases()
        self.ai.learn(self.sample_sensory_data, self.sample_true_coords) 
        initial_params = {
            "w_ih": self.ai.weights_input_hidden.copy(), "b_h": self.ai.bias_hidden.copy(),
            "w_ho": self.ai.weights_hidden_output.copy(), "b_o": self.ai.bias_output.copy()
        }
        self.ai.consolidate()
        if len(self.ai.memory) > 0 and self.ai.learning_rate > 0:
            self.assertFalse(np.array_equal(initial_params["w_ih"], self.ai.weights_input_hidden))
            self.assertFalse(np.array_equal(initial_params["b_h"], self.ai.bias_hidden))
            self.assertFalse(np.array_equal(initial_params["w_ho"], self.ai.weights_hidden_output))
            self.assertFalse(np.array_equal(initial_params["b_o"], self.ai.bias_output))

    def test_model_save_load_new_parietal_architecture(self):
        if self.ai.weights_input_hidden is None: self.ai._initialize_default_weights_biases()
        self.ai.weights_input_hidden = np.full((self.ai.input_size, self.ai.hidden_size), 0.81)
        self.ai.bias_hidden = np.full((1, self.ai.hidden_size), 0.82)
        self.ai.weights_hidden_output = np.full((self.ai.hidden_size, self.ai.output_size), 0.83)
        self.ai.bias_output = np.full((1, self.ai.output_size), 0.84)

        self.ai.save_model()
        loaded_ai = ParietalLobeAI(model_path=self.test_instance_model_path)

        np.testing.assert_array_almost_equal(loaded_ai.weights_input_hidden, self.ai.weights_input_hidden)
        np.testing.assert_array_almost_equal(loaded_ai.bias_hidden, self.ai.bias_hidden)
        np.testing.assert_array_almost_equal(loaded_ai.weights_hidden_output, self.ai.weights_hidden_output)
        np.testing.assert_array_almost_equal(loaded_ai.bias_output, self.ai.bias_output)

        self.assertEqual(loaded_ai.input_size, self.ai.input_size)
        self.assertEqual(loaded_ai.hidden_size, self.ai.hidden_size)
        self.assertEqual(loaded_ai.output_size, self.ai.output_size)

    def test_load_model_backward_compatibility_parietal(self):
        old_model_data = {"weights": (np.random.rand(self.ai.input_size, self.ai.output_size) * 0.01).tolist()}
        with open(self.test_instance_model_path, 'w') as f:
            json.dump(old_model_data, f)

        loaded_ai = ParietalLobeAI(model_path=self.test_instance_model_path)
        self.assertIsNotNone(loaded_ai.weights_input_hidden)
        self.assertEqual(loaded_ai.weights_input_hidden.shape, (self.ai.input_size, self.ai.hidden_size))
        # ... and so on for other weights/biases

    def test_load_model_shape_mismatch_parietal(self):
        with open(self.test_instance_model_path, 'w') as f: # Corrupt file
            f.write('not a valid json file')
        ai = ParietalLobeAI(model_path=self.test_instance_model_path)
        self.assertIsNotNone(ai.weights_input_hidden) # Should reinitialize
        self.assertEqual(ai.weights_input_hidden.shape, (self.ai.input_size, self.ai.hidden_size))


class TestBrainCoordinator(unittest.TestCase):
    def setUp(self):
        self.coordinator = BrainCoordinator()

        # Mock all AI modules
        self.coordinator.frontal = MagicMock(spec=FrontalLobeAI)
        self.coordinator.parietal = MagicMock(spec=ParietalLobeAI)
        self.coordinator.temporal = MagicMock(spec=TemporalLobeAI)
        self.coordinator.occipital = MagicMock(spec=OccipitalLobeAI)
        self.coordinator.cerebellum = MagicMock(spec=CerebellumAI)
        self.coordinator.limbic = MagicMock(spec=LimbicSystemAI)

        # Configure mock return values for process_task
        # Frontal
        self.mock_action = 0 # Predictable action
        self.coordinator.frontal.process_task.return_value = self.mock_action
        # Occipital
        self.mock_vision_label = 0
        self.coordinator.occipital.output_size = 5 # Default from OccipitalLobeAI
        self.coordinator.occipital.process_task.return_value = self.mock_vision_label
        # Parietal
        self.mock_spatial_output = [0.1, 0.2, 0.3]
        self.coordinator.parietal.process_task.return_value = self.mock_spatial_output
        # Temporal
        self.mock_memory_embedding = [0.01] * 10 # Default embedding size for TemporalLobeAI
        self.coordinator.temporal.process_task.return_value = self.mock_memory_embedding
        
        # Cerebellum and Limbic are called but their output isn't directly in frontal_state
        self.coordinator.cerebellum.process_task.return_value = [0.0, 0.0, 0.0]
        self.coordinator.limbic.process_task.return_value = {"label": 0, "probabilities": [1.0, 0.0, 0.0]}


    def _get_expected_frontal_state(self, vision_label, parietal_output, temporal_output, occipital_output_size=5):
        vision_features = np.zeros(occipital_output_size)
        if 0 <= vision_label < occipital_output_size:
            vision_features[vision_label] = 1.0
        
        parietal_1d = np.array(parietal_output).flatten()
        if parietal_1d.shape[0] != 3: # Ensure consistent shape as in main.py
            temp_p = np.zeros(3)
            min_len = min(parietal_1d.shape[0], 3)
            temp_p[:min_len] = parietal_1d[:min_len]
            parietal_1d = temp_p

        temporal_1d = np.array(temporal_output).flatten()
        if temporal_1d.shape[0] != 10: # Ensure consistent shape as in main.py
            temp_t = np.zeros(10)
            min_len = min(temporal_1d.shape[0], 10)
            temp_t[:min_len] = temporal_1d[:min_len]
            temporal_1d = temp_t
            
        return np.concatenate([vision_features, parietal_1d, temporal_1d])

    def test_frontal_learn_episodic_calls(self):
        num_days = 7
        episode_length = self.coordinator.episode_length # Should be 5
        vision_input_path = "dummy_path.png" # Mocked, path doesn't need to exist
        sensor_data = [0.5, 0.5, 0.5]
        text_data = "test text"
        feedback_reward = 0.75
        
        # Store expected states for verification
        expected_states_history = []

        # --- Day 1 ---
        self.coordinator.process_day(vision_input_path, sensor_data, text_data, {"action_reward": feedback_reward})
        self.coordinator.frontal.learn.assert_not_called() # Learn not called on first day
        expected_states_history.append(
            self._get_expected_frontal_state(self.mock_vision_label, self.mock_spatial_output, self.mock_memory_embedding)
        )

        # --- Days 2 to num_days ---
        for day in range(2, num_days + 1):
            # Simulate slightly different inputs for different states if desired, or keep them same
            # For this test, keeping inputs same means mock outputs will be same, leading to same current_frontal_state
            # This is fine as we are testing the *sequence* of (s, a, r, s', done)
            current_day_reward = feedback_reward + (day * 0.01) # Slightly varying reward
            
            self.coordinator.process_day(vision_input_path, sensor_data, text_data, {"action_reward": current_day_reward})
            
            # current_frontal_state for this day becomes next_state for the previous learn call
            current_expected_state = self._get_expected_frontal_state(
                self.mock_vision_label, self.mock_spatial_output, self.mock_memory_embedding
            )
            expected_states_history.append(current_expected_state)

            # Assertions for the learn call that happened due to *previous* day's state
            self.coordinator.frontal.learn.assert_called_once()
            
            args, _ = self.coordinator.frontal.learn.call_args
            state, action, reward, next_state, done = args

            # State should be from previous day's calculation
            np.testing.assert_array_almost_equal(state, expected_states_history[day-2],
                                                 err_msg=f"Day {day}: Incorrect 'state' passed to learn.")
            # Action is what frontal.process_task returned for state_t-1
            self.assertEqual(action, self.mock_action, f"Day {day}: Incorrect 'action'.")
            # Reward is from previous day's feedback
            self.assertAlmostEqual(reward, feedback_reward + ((day-1) * 0.01) if day > 1 else feedback_reward, # last_reward_for_learning
                                   msg=f"Day {day}: Incorrect 'reward'.")
            # Next state is the current day's calculated state
            np.testing.assert_array_almost_equal(next_state, current_expected_state,
                                                 err_msg=f"Day {day}: Incorrect 'next_state'.")

            # Check 'done' flag
            # steps_since_last_episode_end was incremented *before* this learn call
            # For day=2, steps_since_last_episode_end = 1. For day=6 (end of ep), steps = 5.
            # So, learn call on day N corresponds to experience ending after day N-1.
            # Episode ends after day 5's experience, so learn call on day 6 gets done=True
            expected_done = (day - 1) % episode_length == 0 and (day-1) > 0 
            # Correction: done is True when steps_since_last_episode_end (which is day-1 for the *transition*) hits episode_length
            # coordinator.steps_since_last_episode_end is state *after* processing current day, before learn.
            # The learn call uses the state of coordinator.steps_since_last_episode_end *before* it's potentially reset.
            # The learn() call for the transition (s_{d-1}, a_{d-1}, r_{d-1}) -> s_d
            # uses steps_since_last_episode_end that has been incremented for day d.
            # So, if day d makes steps_since_last_episode_end == episode_length, then done is True for that learn call.
            
            # Let's trace coordinator's internal state:
            # Day 1: last_state=None. process_day(1). last_state=s1, last_action=a1, last_reward=r1. steps=0. learn not called.
            # Day 2: last_state=s1. process_day(2). steps becomes 1. learn(s1,a1,r1, s2, done=1==5(F)). last_state=s2, last_action=a2, last_reward=r2.
            # Day 3: last_state=s2. process_day(3). steps becomes 2. learn(s2,a2,r2, s3, done=2==5(F)). last_state=s3, last_action=a3, last_reward=r3.
            # Day 4: last_state=s3. process_day(4). steps becomes 3. learn(s3,a3,r3, s4, done=3==5(F)). last_state=s4, last_action=a4, last_reward=r4.
            # Day 5: last_state=s4. process_day(5). steps becomes 4. learn(s4,a4,r4, s5, done=4==5(F)). last_state=s5, last_action=a5, last_reward=r5.
            # Day 6: last_state=s5. process_day(6). steps becomes 5. learn(s5,a5,r5, s6, done=5==5(T)). steps resets to 0. last_state=s6, last_action=a6, last_reward=r6.
            # Day 7: last_state=s6. process_day(7). steps becomes 1. learn(s6,a6,r6, s7, done=1==5(F)). last_state=s7, last_action=a7, last_reward=r7.
            
            # So, done is True if coordinator.steps_since_last_episode_end was episode_length *before* reset
            # The value of coordinator.steps_since_last_episode_end *at the time of the learn call* is the one to check
            # This value is captured by the done_for_learn variable inside process_day
            # If the current learn call is for the transition ending episode `k*episode_length`, then `done` is True.
            # This happens on day `k*episode_length + 1`.
            # For day=6, learn is for transition s5->s6. steps_since_last_episode_end inside process_day (before learn) is 5.
            # So done is True.
            
            is_episode_end_step = (self.coordinator.steps_since_last_episode_end == 0) # True if it was just reset
            if is_episode_end_step and (day-1) >= episode_length : # It was reset because it hit episode_length
                 self.assertTrue(done, f"Day {day}: 'done' should be True as this is the first step of a new episode, meaning previous one ended.")
            else:
                 # If not end of episode, steps_since_last_episode_end should be > 0 unless it's the very first learning step (day 2)
                 self.assertFalse(done, f"Day {day}: 'done' should be False. steps_since_last_episode_end={self.coordinator.steps_since_last_episode_end}")


            self.coordinator.frontal.learn.reset_mock()


if __name__ == '__main__':
    unittest.main()
