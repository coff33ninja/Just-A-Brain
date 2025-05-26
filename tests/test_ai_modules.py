import unittest
import numpy as np
import os
import json
import shutil # For cleaning up directories

# Adjust import paths if necessary, assuming 'main.py' and AI modules are in the parent directory
# or accessible via PYTHONPATH. For this environment, direct imports should work if files are at root.
# If 'main.py' is in root and AI modules are in a subdirectory (e.g., 'ai_modules'), adjust accordingly.
# Assuming all .py files (main, temporal, cerebellum, limbic, occipital) are in the root for this example.

# From main.py
from main import load_vision_data, load_sensor_data, load_text_data, load_image_text_pairs

# AI Classes
from temporal import TemporalLobeAI, softmax as temporal_softmax
from cerebellum import CerebellumAI
from limbic import LimbicSystemAI, softmax as limbic_softmax
from occipital import OccipitalLobeAI, softmax as occipital_softmax
from frontal import FrontalLobeAI
from parietal import ParietalLobeAI # Added ParietalLobeAI
from PIL import Image # For creating test images

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
        self.ai = TemporalLobeAI(model_path=TEST_TEMPORAL_MODEL_PATH, memory_path=TEST_TEMPORAL_MEMORY_PATH)
        if os.path.exists(TEST_TEMPORAL_MODEL_PATH): os.remove(TEST_TEMPORAL_MODEL_PATH)
        if os.path.exists(TEST_TEMPORAL_MEMORY_PATH): os.remove(TEST_TEMPORAL_MEMORY_PATH)
        self.ai._initialize_default_weights_biases(); self.ai.save_model(); self.ai.save_memory()
    def tearDown(self):
        if os.path.exists(TEST_TEMPORAL_MODEL_PATH): os.remove(TEST_TEMPORAL_MODEL_PATH)
        if os.path.exists(TEST_TEMPORAL_MEMORY_PATH): os.remove(TEST_TEMPORAL_MEMORY_PATH)
    def test_forward_prop_text_output_shapes(self):
        input_vec_1d = self.ai._to_numerical_vector("sample", self.ai.input_size)
        _, hidden_out, embedding_scores = self.ai._forward_prop_text(input_vec_1d)
        self.assertEqual(hidden_out.shape, (self.ai.text_hidden_size,))
        self.assertEqual(embedding_scores.shape, (self.ai.output_size,))
    def test_forward_prop_visual_assoc_output_shapes(self):
        test_embedding_vec = np.random.rand(self.ai.output_size)
        _, hidden_out, visual_scores = self.ai._forward_prop_visual_assoc(test_embedding_vec)
        self.assertEqual(hidden_out.shape, (self.ai.visual_assoc_hidden_size,))
        self.assertEqual(visual_scores.shape, (self.ai.visual_output_size,))
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
        self.assertIn((("text", "target")), self.ai.memory_db)
        self.assertIn(("text", 1), self.ai.cross_modal_memory)
    def test_consolidate_updates_all_weights_and_biases(self):
        self.ai.learn(sequence=[("text c1", "target c1")], visual_label_as_context=0)
        self.ai.learn(sequence=[("text c2", "target c2")], visual_label_as_context=2)
        initial_w = [p.copy() for p in [self.ai.weights_text_input_hidden, self.ai.weights_text_hidden_embedding, self.ai.weights_embed_to_visual_hidden, self.ai.weights_visual_hidden_to_label]]
        self.ai.consolidate()
        final_w = [self.ai.weights_text_input_hidden, self.ai.weights_text_hidden_embedding, self.ai.weights_embed_to_visual_hidden, self.ai.weights_visual_hidden_to_label]
        for i_w, f_w in zip(initial_w, final_w): self.assertFalse(np.array_equal(i_w, f_w))
    def test_model_save_load_new_temporal_architecture(self):
        self.ai.weights_text_input_hidden = np.full((self.ai.input_size, self.ai.text_hidden_size), 0.1)
        self.ai.save_model()
        loaded_ai = TemporalLobeAI(model_path=TEST_TEMPORAL_MODEL_PATH, memory_path=TEST_TEMPORAL_MEMORY_PATH)
        np.testing.assert_array_almost_equal(loaded_ai.weights_text_input_hidden, self.ai.weights_text_input_hidden)
    def test_memory_persistence_dict_format(self):
        self.ai.memory_db.append([("m_item","t_item")]); self.ai.cross_modal_memory.append(("cm_text",1))
        self.ai.save_memory()
        new_ai = TemporalLobeAI(TEST_TEMPORAL_MODEL_PATH, TEST_TEMPORAL_MEMORY_PATH)
        self.assertIn([("m_item","t_item")], new_ai.memory_db)
        self.assertIn(("cm_text",1), new_ai.cross_modal_memory)


class TestLimbicSystemAI(unittest.TestCase):
    def setUp(self):
        os.makedirs(TEST_DATA_DIR, exist_ok=True)
        self.ai = LimbicSystemAI(model_path=TEST_LIMBIC_MODEL_PATH)
        if os.path.exists(TEST_LIMBIC_MODEL_PATH): os.remove(TEST_LIMBIC_MODEL_PATH)
        self.ai._initialize_default_weights_biases(); self.ai.save_model()
        self.sample_temporal_output = np.random.rand(self.ai.input_size).tolist()
        self.sample_true_emotion = 1; self.sample_reward = 1.0
    def tearDown(self):
        if os.path.exists(TEST_LIMBIC_MODEL_PATH): os.remove(TEST_LIMBIC_MODEL_PATH)
    def test_ensure_input_vector_shape(self):
        vec=self.ai._ensure_input_vector_shape(self.sample_temporal_output); self.assertEqual(vec.shape,(self.ai.input_size,))
    def test_forward_propagate_output_shapes_limbic(self):
        i,h,o = self.ai._forward_propagate(self.sample_temporal_output)
        self.assertEqual(i.shape,(self.ai.input_size,)); self.assertEqual(h.shape,(self.ai.hidden_size,)); self.assertEqual(o.shape,(self.ai.output_size,))
    def test_process_task_output_label_limbic(self):
        label=self.ai.process_task(self.sample_temporal_output); self.assertTrue(0<=label<self.ai.output_size)
    def test_learn_updates_and_memory_limbic(self):
        w_ih,b_h,w_ho,b_o = [p.copy() for p in [self.ai.weights_input_hidden,self.ai.bias_hidden,self.ai.weights_hidden_output,self.ai.bias_output]]
        self.ai.learn(self.sample_temporal_output,self.sample_true_emotion,self.sample_reward)
        self.assertFalse(np.array_equal(w_ih,self.ai.weights_input_hidden))
        self.assertIn((self.sample_temporal_output,self.sample_true_emotion,self.sample_reward),self.ai.memory)
    def test_consolidate_updates_weights_limbic(self):
        self.ai.learn(self.sample_temporal_output,self.sample_true_emotion,self.sample_reward)
        w_ih=self.ai.weights_input_hidden.copy(); self.ai.consolidate(); self.assertFalse(np.array_equal(w_ih,self.ai.weights_input_hidden))
    def test_model_save_load_new_limbic_architecture(self):
        self.ai.weights_input_hidden=np.full((self.ai.input_size,self.ai.hidden_size),0.7); self.ai.save_model()
        loaded_ai=LimbicSystemAI(model_path=TEST_LIMBIC_MODEL_PATH)
        np.testing.assert_array_almost_equal(loaded_ai.weights_input_hidden,self.ai.weights_input_hidden)


class TestOccipitalLobeAI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.makedirs(TEST_IMAGES_SUBDIR, exist_ok=True)
        cls.img_size=(32,32); cls.black_img_path=os.path.join(TEST_IMAGES_SUBDIR,"b.png"); Image.new('L',cls.img_size,0).save(cls.black_img_path)
    @classmethod
    def tearDownClass(cls):
        if os.path.exists(TEST_IMAGES_SUBDIR): shutil.rmtree(TEST_IMAGES_SUBDIR)
        if os.path.exists(TEST_OCCIPITAL_MODEL_PATH): os.remove(TEST_OCCIPITAL_MODEL_PATH)
    def setUp(self):
        self.ai = OccipitalLobeAI(model_path=TEST_OCCIPITAL_MODEL_PATH)
        if os.path.exists(TEST_OCCIPITAL_MODEL_PATH): os.remove(TEST_OCCIPITAL_MODEL_PATH)
        self.ai.save_model()
    def test_forward_propagate_output_shapes(self):
        i,h,o = self.ai._forward_propagate(self.black_img_path)
        self.assertEqual(i.shape,(self.ai.input_size,)); self.assertEqual(h.shape,(self.ai.hidden_size,)); self.assertEqual(o.shape,(self.ai.output_size,))
    def test_learn_updates_and_memory(self):
        w_ih,b_h,w_ho,b_o = [p.copy() for p in [self.ai.weights_input_hidden,self.ai.bias_hidden,self.ai.weights_hidden_output,self.ai.bias_output]]
        self.ai.learn(self.black_img_path,0)
        self.assertFalse(np.array_equal(w_ih,self.ai.weights_input_hidden)); self.assertIn((self.black_img_path,0),self.ai.memory)
    def test_consolidate_updates_weights(self):
        self.ai.learn(self.black_img_path,1)
        w_ih = self.ai.weights_input_hidden.copy()
        self.ai.consolidate()
        self.assertFalse(np.array_equal(w_ih, self.ai.weights_input_hidden))
    def test_model_save_load_new_architecture(self):
        self.ai.weights_input_hidden = np.full((self.ai.input_size, self.ai.hidden_size), 0.1)
        self.ai.save_model()
        loaded_ai = OccipitalLobeAI(model_path=TEST_OCCIPITAL_MODEL_PATH)
        np.testing.assert_array_almost_equal(loaded_ai.weights_input_hidden, self.ai.weights_input_hidden)


class TestFrontalLobeAI(unittest.TestCase):
    def setUp(self):
        os.makedirs(TEST_DATA_DIR, exist_ok=True)
        self.ai = FrontalLobeAI(model_path=TEST_FRONTAL_MODEL_PATH)
        if os.path.exists(TEST_FRONTAL_MODEL_PATH): os.remove(TEST_FRONTAL_MODEL_PATH)
        self.ai.weights = np.random.rand(self.ai.input_size, self.ai.output_size) * 0.01
        self.ai.exploration_rate_epsilon = 1.0
        self.ai.save_model()
        self.sample_state = np.random.rand(self.ai.input_size).tolist()
    def tearDown(self):
        if os.path.exists(TEST_FRONTAL_MODEL_PATH): os.remove(TEST_FRONTAL_MODEL_PATH)
    def test_process_task_epsilon_greedy_selection(self):
        self.ai.exploration_rate_epsilon = 1.0
        action_counts_explore = {i:0 for i in range(self.ai.output_size)}
        for _ in range(1000): action_counts_explore[self.ai.process_task(self.sample_state)] += 1
        self.assertTrue(len(set(action_counts_explore.values())) > 1 or self.ai.output_size == 1)
        self.ai.exploration_rate_epsilon = 0.0
        self.ai.weights = np.zeros((self.ai.input_size, self.ai.output_size)); self.ai.weights[:, 1] = 1.0
        self.assertEqual(self.ai.process_task(self.sample_state), 1)
        initial_epsilon = 0.5; self.ai.exploration_rate_epsilon = initial_epsilon
        self.ai.process_task(self.sample_state)
        self.assertAlmostEqual(self.ai.exploration_rate_epsilon, max(self.ai.min_epsilon, initial_epsilon - self.ai.epsilon_decay_rate))
    def test_learn_q_update_and_memory(self):
        state=self.sample_state; action_taken=1; reward=1.0; next_state=np.random.rand(self.ai.input_size).tolist(); done=False
        self.setUp()
        initial_weights_all_actions = self.ai.weights.copy()
        state_vector_1d = self.ai._prepare_state_vector(state)
        q_value_before = (state_vector_1d @ initial_weights_all_actions)[action_taken]
        self.ai.learn(state, action_taken, reward, next_state, done)
        expected_weights_for_action = initial_weights_all_actions[:,action_taken] + self.ai.learning_rate_q * (reward - q_value_before) * state_vector_1d
        np.testing.assert_array_almost_equal(self.ai.weights[:,action_taken], expected_weights_for_action, decimal=5)
        for i in range(self.ai.output_size):
            if i != action_taken: np.testing.assert_array_equal(self.ai.weights[:,i], initial_weights_all_actions[:,i])
        self.assertIn((list(state),action_taken,reward,list(next_state),done), self.ai.memory)
        self.ai.memory=[]; [self.ai.learn(state,action_taken,reward,next_state,done) for _ in range(self.ai.max_memory_size + 5)]
        self.assertEqual(len(self.ai.memory), self.ai.max_memory_size)
    def test_consolidate_experience_replay(self):
        for i in range(10): self.ai.learn(np.random.rand(self.ai.input_size).tolist(), np.random.randint(0,self.ai.output_size), float(np.random.choice([-1,0,1])), np.random.rand(self.ai.input_size).tolist(), bool(np.random.choice([True,False])))
        self.assertTrue(len(self.ai.memory)>0); initial_weights = self.ai.weights.copy()
        self.ai.consolidate()
        self.assertFalse(np.array_equal(initial_weights, self.ai.weights))
    def test_model_save_load_q_learning_params(self):
        self.ai.exploration_rate_epsilon = 0.678; self.ai.weights = np.random.rand(self.ai.input_size, self.ai.output_size)*0.5
        known_weights = self.ai.weights.copy(); self.ai.save_model()
        loaded_ai = FrontalLobeAI(model_path=TEST_FRONTAL_MODEL_PATH)
        self.assertAlmostEqual(loaded_ai.exploration_rate_epsilon, 0.678)
        np.testing.assert_array_almost_equal(loaded_ai.weights, known_weights)
        with open(TEST_FRONTAL_MODEL_PATH, 'w') as f: json.dump({'weights': known_weights.tolist()}, f)
        loaded_ai_old = FrontalLobeAI(model_path=TEST_FRONTAL_MODEL_PATH)
        self.assertEqual(loaded_ai_old.exploration_rate_epsilon, 1.0)


class TestCerebellumAI(unittest.TestCase):
    def setUp(self):
        os.makedirs(TEST_DATA_DIR, exist_ok=True)
        self.ai = CerebellumAI(model_path=TEST_CEREBELLUM_MODEL_PATH)
        if os.path.exists(TEST_CEREBELLUM_MODEL_PATH):
            os.remove(TEST_CEREBELLUM_MODEL_PATH)
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
        short_data = self.sample_sensor_data[:-2]; vec_short = self.ai._prepare_input_vector(short_data)
        self.assertEqual(vec_short.shape, (self.ai.input_size,))
        self.assertTrue(np.all(vec_short[len(short_data):] == 0))
        long_data = self.sample_sensor_data + [0.1, 0.2]; vec_long = self.ai._prepare_input_vector(long_data)
        self.assertEqual(vec_long.shape, (self.ai.input_size,))
        np.testing.assert_array_equal(vec_long, np.array(long_data[:self.ai.input_size]))

    def test_prepare_target_command_vector(self):
        vec = self.ai._prepare_target_command_vector(self.sample_true_command)
        self.assertEqual(vec.shape, (self.ai.output_size,))
        short_data = self.sample_true_command[:-1]; vec_short = self.ai._prepare_target_command_vector(short_data)
        self.assertEqual(vec_short.shape, (self.ai.output_size,))
        self.assertTrue(np.all(vec_short[len(short_data):] == 0))
        long_data = self.sample_true_command + [0.1]; vec_long = self.ai._prepare_target_command_vector(long_data)
        self.assertEqual(vec_long.shape, (self.ai.output_size,))
        np.testing.assert_array_equal(vec_long, np.array(long_data[:self.ai.output_size]))

    def test_forward_propagate_output_shapes_and_ranges(self):
        input_vec, hidden_out, final_commands = self.ai._forward_propagate(self.sample_sensor_data)
        self.assertEqual(input_vec.shape, (self.ai.input_size,))
        self.assertEqual(hidden_out.shape, (self.ai.hidden_size,))
        self.assertEqual(final_commands.shape, (self.ai.output_size,))
        self.assertTrue(np.all(hidden_out >= -1) and np.all(hidden_out <= 1))
        self.assertTrue(np.all(final_commands >= -1) and np.all(final_commands <= 1))

    def test_process_task_output(self):
        commands = self.ai.process_task(self.sample_sensor_data)
        self.assertIsInstance(commands, list)
        self.assertEqual(len(commands), self.ai.output_size)
        for val in commands: self.assertTrue(-1.0 <= val <= 1.0)

    def test_learn_updates_and_memory(self):
        initial_w_ih = self.ai.weights_input_hidden.copy(); initial_b_h = self.ai.bias_hidden.copy()
        initial_w_ho = self.ai.weights_hidden_output.copy(); initial_b_o = self.ai.bias_output.copy()

        self.ai.learn(self.sample_sensor_data, self.sample_true_command)

        self.assertFalse(np.array_equal(initial_w_ih, self.ai.weights_input_hidden))
        self.assertFalse(np.array_equal(initial_b_h, self.ai.bias_hidden))
        self.assertFalse(np.array_equal(initial_w_ho, self.ai.weights_hidden_output))
        self.assertFalse(np.array_equal(initial_b_o, self.ai.bias_output))

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

        self.assertFalse(np.array_equal(w_ih_before, self.ai.weights_input_hidden))
        self.assertFalse(np.array_equal(b_h_before, self.ai.bias_hidden))
        self.assertFalse(np.array_equal(w_ho_before, self.ai.weights_hidden_output))
        self.assertFalse(np.array_equal(b_o_before, self.ai.bias_output))

    def test_model_save_load_new_cerebellum_architecture(self):
        self.ai.weights_input_hidden = np.full((self.ai.input_size, self.ai.hidden_size), 0.5)
        self.ai.bias_output = np.full((1, self.ai.output_size), 0.6)
        self.ai.bias_hidden = np.full((1, self.ai.hidden_size), 0.55)
        self.ai.weights_hidden_output = np.full((self.ai.hidden_size, self.ai.output_size), 0.65)

        self.ai.save_model()

        loaded_ai = CerebellumAI(model_path=TEST_CEREBELLUM_MODEL_PATH)
        np.testing.assert_array_almost_equal(loaded_ai.weights_input_hidden, self.ai.weights_input_hidden)
        np.testing.assert_array_almost_equal(loaded_ai.bias_hidden, self.ai.bias_hidden)
        np.testing.assert_array_almost_equal(loaded_ai.weights_hidden_output, self.ai.weights_hidden_output)
        np.testing.assert_array_almost_equal(loaded_ai.bias_output, self.ai.bias_output)
        self.assertEqual(loaded_ai.hidden_size, self.ai.hidden_size)

    def test_load_model_backward_compatibility_cerebellum(self):
        old_model_data = {"weights": (np.random.rand(self.ai.input_size, self.ai.output_size) * 0.01).tolist()}
        with open(TEST_CEREBELLUM_MODEL_PATH, 'w') as f: json.dump(old_model_data, f)

        loaded_ai = CerebellumAI(model_path=TEST_CEREBELLUM_MODEL_PATH)
        self.assertIsNotNone(loaded_ai.weights_input_hidden)
        self.assertEqual(loaded_ai.weights_input_hidden.shape, (self.ai.input_size, self.ai.hidden_size))
        self.assertEqual(loaded_ai.weights_hidden_output.shape, (self.ai.hidden_size, self.ai.output_size))

    def test_load_model_shape_mismatch_cerebellum(self):
        self.ai.save_model()
        with open(TEST_CEREBELLUM_MODEL_PATH, 'r') as f: model_data = json.load(f)
        model_data['weights_input_hidden'] = [[0.01]]
        with open(TEST_CEREBELLUM_MODEL_PATH, 'w') as f: json.dump(model_data, f)

        temp_ai_defaults = CerebellumAI(model_path="non_existent_cerebellum.json")
        default_w_ih = temp_ai_defaults.weights_input_hidden.copy()

        loaded_ai = CerebellumAI(model_path=TEST_CEREBELLUM_MODEL_PATH)
        np.testing.assert_array_equal(loaded_ai.weights_input_hidden, default_w_ih)


# Updated Test Class for ParietalLobeAI
class TestParietalLobeAI(unittest.TestCase):
    def setUp(self):
        os.makedirs(TEST_DATA_DIR, exist_ok=True)
        self.ai = ParietalLobeAI(model_path=TEST_PARIETAL_MODEL_PATH)
        if os.path.exists(TEST_PARIETAL_MODEL_PATH):
            os.remove(TEST_PARIETAL_MODEL_PATH)
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
        np.testing.assert_array_equal(vec_long, np.array(long_data[:self.ai.input_size]))

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
        np.testing.assert_array_equal(vec_long, np.array(long_data[:self.ai.output_size]))

    def test_forward_propagate_output_shapes_parietal(self):
        input_vec, hidden_out, output_coords = self.ai._forward_propagate(self.sample_sensory_data)
        self.assertEqual(input_vec.shape, (self.ai.input_size,))
        self.assertEqual(hidden_out.shape, (self.ai.hidden_size,))
        self.assertEqual(output_coords.shape, (self.ai.output_size,))
        self.assertTrue(np.all(hidden_out >= -1) and np.all(hidden_out <= 1)) # Tanh for hidden

    def test_process_task_output_parietal(self):
        coords = self.ai.process_task(self.sample_sensory_data)
        self.assertIsInstance(coords, list)
        self.assertEqual(len(coords), self.ai.output_size)

    def test_learn_updates_and_memory_parietal(self):
        initial_w_ih = self.ai.weights_input_hidden.copy(); initial_b_h = self.ai.bias_hidden.copy()
        initial_w_ho = self.ai.weights_hidden_output.copy(); initial_b_o = self.ai.bias_output.copy()

        self.ai.learn(self.sample_sensory_data, self.sample_true_coords)

        self.assertFalse(np.array_equal(initial_w_ih, self.ai.weights_input_hidden))
        self.assertFalse(np.array_equal(initial_b_h, self.ai.bias_hidden))
        self.assertFalse(np.array_equal(initial_w_ho, self.ai.weights_hidden_output))
        self.assertFalse(np.array_equal(initial_b_o, self.ai.bias_output))

        expected_mem_sensor = self.sample_sensory_data if isinstance(self.sample_sensory_data, list) else self.sample_sensory_data.tolist()
        expected_mem_coords = self.sample_true_coords if isinstance(self.sample_true_coords, list) else self.sample_true_coords.tolist()
        self.assertIn((expected_mem_sensor, expected_mem_coords), self.ai.memory)

        self.ai.memory = []
        for i in range(self.ai.max_memory_size + 5): # Test memory limit
            self.ai.learn(self.sample_sensory_data, (np.array(self.sample_true_coords) * (i / (self.ai.max_memory_size + 4.0))).tolist())
        self.assertEqual(len(self.ai.memory), self.ai.max_memory_size)

    def test_consolidate_updates_weights_parietal(self):
        self.ai.learn(self.sample_sensory_data, self.sample_true_coords) # Populate memory

        w_ih_before = self.ai.weights_input_hidden.copy(); b_o_before = self.ai.bias_output.copy()

        self.ai.consolidate()

        self.assertFalse(np.array_equal(w_ih_before, self.ai.weights_input_hidden))
        self.assertFalse(np.array_equal(b_o_before, self.ai.bias_output))

    def test_model_save_load_new_parietal_architecture(self):
        self.ai.weights_input_hidden = np.full((self.ai.input_size, self.ai.hidden_size), 0.8)
        self.ai.bias_output = np.full((1, self.ai.output_size), 0.9)
        self.ai.save_model()

        loaded_ai = ParietalLobeAI(model_path=TEST_PARIETAL_MODEL_PATH)
        np.testing.assert_array_almost_equal(loaded_ai.weights_input_hidden, self.ai.weights_input_hidden)
        np.testing.assert_array_almost_equal(loaded_ai.bias_output, self.ai.bias_output)
        self.assertEqual(loaded_ai.hidden_size, self.ai.hidden_size)

    def test_load_model_backward_compatibility_parietal(self):
        old_model_data = {"weights": (np.random.rand(self.ai.input_size, self.ai.output_size) * 0.01).tolist()}
        with open(TEST_PARIETAL_MODEL_PATH, 'w') as f: json.dump(old_model_data, f)

        loaded_ai = ParietalLobeAI(model_path=TEST_PARIETAL_MODEL_PATH)
        self.assertIsNotNone(loaded_ai.weights_input_hidden)
        self.assertEqual(loaded_ai.weights_input_hidden.shape, (self.ai.input_size, self.ai.hidden_size))

    def test_load_model_shape_mismatch_parietal(self):
        self.ai.save_model()
        with open(TEST_PARIETAL_MODEL_PATH, 'r') as f: model_data = json.load(f)
        model_data['weights_input_hidden'] = [[0.01]] # Corrupt shape
        with open(TEST_PARIETAL_MODEL_PATH, 'w') as f: json.dump(model_data, f)

        temp_ai_defaults = ParietalLobeAI(model_path="non_existent_parietal.json")
        default_w_ih = temp_ai_defaults.weights_input_hidden.copy()

        loaded_ai = ParietalLobeAI(model_path=TEST_PARIETAL_MODEL_PATH)
        np.testing.assert_array_equal(loaded_ai.weights_input_hidden, default_w_ih)


if __name__ == '__main__':
    unittest.main()
