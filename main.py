# main.py (Central Coordinator)
import numpy as np
from frontal import FrontalLobeAI
from parietal import ParietalLobeAI
from temporal import TemporalLobeAI
from occipital import OccipitalLobeAI
from cerebellum import CerebellumAI
from limbic import LimbicSystemAI
import os
import json


DATA_DIR = "data"
SENSOR_FILE = os.path.join(DATA_DIR, "sensors.json")
IMAGE_TEXT_PAIRS_FILE = os.path.join(DATA_DIR, "image_text_pairs.json")
DEFAULT_IMAGE_PATH = "data/images/default_image.png"


def load_sensor_data(filepath=SENSOR_FILE):
    if not os.path.exists(filepath):
        print(
            f"Warning: Sensor data file not found at {filepath}. Returning empty list."
        )
        return []
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading sensor data from {filepath}: {e}. Returning empty list.")
        return []


def load_image_text_pairs(filepath=IMAGE_TEXT_PAIRS_FILE):
    if not os.path.exists(filepath):
        print(
            f"Error: Image-text pairs file not found at {filepath}. Returning empty list."
        )
        return []
    try:
        with open(filepath, "r") as f:
            pairs = json.load(f)
        if not isinstance(pairs, list):
            print(
                f"Error: Image-text pairs file {filepath} is not a list. Returning empty list."
            )
            return []
        return pairs
    except Exception as e:
        print(
            f"Error loading image-text pairs from {filepath}: {e}. Returning empty list."
        )
        return []


class BrainCoordinator:
    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR, "images"), exist_ok=True)
        self.frontal = FrontalLobeAI()
        self.parietal = ParietalLobeAI()
        self.temporal = TemporalLobeAI()
        self.occipital = OccipitalLobeAI()
        self.cerebellum = CerebellumAI()
        self.limbic = LimbicSystemAI()

    def process_day(
        self, vision_input_path, sensor_data, text_data, feedback_data=None
    ):
        current_vision_path = (
            vision_input_path
            if vision_input_path and os.path.exists(vision_input_path)
            else DEFAULT_IMAGE_PATH
        )
        if not os.path.exists(current_vision_path) and vision_input_path:
            print(f"Warning: Vision image path {current_vision_path} not found.")

        current_sensor_data = (
            np.array(sensor_data)
            if sensor_data is not None
            else np.random.randn(getattr(self.parietal, "input_size", 3))
        )  # Parietal output is 3
        current_text_data = (
            text_data if isinstance(text_data, str) else "Default sample text"
        )

        current_feedback = feedback_data if feedback_data is not None else {}

        # Construct full feedback dictionary with defaults
        feedback = {
            "action_reward": current_feedback.get(
                "action_reward", np.random.choice([-1, 0, 1])
            ),
            "spatial_error": current_feedback.get(
                "spatial_error", np.random.rand(3).tolist()
            ),
            "memory_target": current_feedback.get("memory_target", current_text_data),
            "vision_label": current_feedback.get("vision_label", 0),
            "motor_command": current_feedback.get(
                "motor_command", np.random.rand(3).tolist()
            ),
            "emotion_label": current_feedback.get(
                "emotion_label",
                np.random.randint(0, getattr(self.limbic, "output_size", 3)),
            ),
        }

        # 1. Process tasks from other lobes
        vision_result_label = self.occipital.process_task(current_vision_path)

        vision_features_for_frontal = np.zeros(
            self.occipital.output_size
        )  # Expected size 5
        if 0 <= vision_result_label < self.occipital.output_size:
            vision_features_for_frontal[vision_result_label] = 1.0

        spatial_result_raw = self.parietal.process_task(
            current_sensor_data
        )  # Expected output size 3
        spatial_result_1d = np.array(spatial_result_raw).flatten()
        if spatial_result_1d.shape[0] != 3:  # Ensure parietal output is size 3
            # print(f"Warning: Parietal output size mismatch. Expected 3, got {spatial_result_1d.shape[0]}. Adjusting.")
            temp_spatial = np.zeros(3)
            min_len_spatial = min(spatial_result_1d.shape[0], 3)
            temp_spatial[:min_len_spatial] = spatial_result_1d[:min_len_spatial]
            spatial_result_1d = temp_spatial

        # Temporal lobe process_task (text embedding pathway)
        # predict_visual=False ensures it returns only the text embedding
        memory_result_embedding_raw = self.temporal.process_task(
            current_text_data, predict_visual=False
        )  # Expected output size 10
        memory_result_embedding_1d = np.array(memory_result_embedding_raw).flatten()
        if (
            memory_result_embedding_1d.shape[0] != 10
        ):  # Ensure temporal output is size 10
            # print(f"Warning: Temporal embedding size mismatch. Expected 10, got {memory_result_embedding_1d.shape[0]}. Adjusting.")
            temp_memory = np.zeros(10)
            min_len_memory = min(memory_result_embedding_1d.shape[0], 10)
            temp_memory[:min_len_memory] = memory_result_embedding_1d[:min_len_memory]
            memory_result_embedding_1d = temp_memory

        # 2. Assemble input for Frontal Lobe (State for Q-learning)
        # Expected total size: 5 (vision) + 3 (parietal) + 10 (temporal) = 18
        concatenated_input_for_frontal = np.concatenate(
            [vision_features_for_frontal, spatial_result_1d, memory_result_embedding_1d]
        )

        frontal_state = concatenated_input_for_frontal  # This is a NumPy array, FrontalLobeAI._prepare_state_vector can handle it

        # 3. Frontal Lobe processes the state to choose an action
        action = self.frontal.process_task(
            frontal_state
        )  # process_task expects a state vector

        # Other lobes process their respective inputs (no change here)
        motor_command = self.cerebellum.process_task(current_sensor_data)
        emotion = self.limbic.process_task(
            memory_result_embedding_1d
        )  # Limbic uses temporal embedding

        # 4. Prepare parameters for Frontal Lobe learning
        state_for_learn = frontal_state
        action_for_learn = action  # Action taken by frontal lobe
        reward_for_learn = feedback.get("action_reward", 0)  # Reward for the action

        # For simplicity, next_state is same as current_state and episode always ends (done=True)
        # This aligns with discount_factor_gamma = 0.0 where only immediate reward matters.
        next_state_for_learn = frontal_state
        done_for_learn = True

        # 5. Learn from feedback - Update Frontal Lobe with Q-learning
        self.frontal.learn(
            state_for_learn,
            action_for_learn,
            reward_for_learn,
            next_state_for_learn,
            done_for_learn,
        )

        # Other lobes learn (no change to their learn calls here)
        self.occipital.learn(current_vision_path, feedback["vision_label"])
        self.parietal.learn(current_sensor_data, feedback["spatial_error"])
        self.temporal.learn(
            [
                (current_text_data, feedback["memory_target"])
            ],  # Text processing pathway target
            visual_label_as_context=feedback.get(
                "vision_label"
            ),  # Visual context for association
        )
        self.cerebellum.learn(current_sensor_data, feedback["motor_command"])
        self.limbic.learn(
            memory_result_embedding_1d,
            feedback["emotion_label"],
            feedback["action_reward"],
        )

        return {
            "action": action,
            "motor": motor_command,
            "emotion": emotion,
            "vision_label": vision_result_label,
        }

    def bedtime(self):
        print("Starting bedtime consolidation...")
        self.frontal.consolidate()
        self.parietal.consolidate()
        self.temporal.consolidate()
        self.occipital.consolidate()
        self.cerebellum.consolidate()
        self.limbic.consolidate()
        print("Consolidation complete.")


def main():
    coordinator = BrainCoordinator()
    image_text_pairs_list = load_image_text_pairs()
    sensor_data_list = load_sensor_data()

    if not image_text_pairs_list:
        print("No image-text pairs loaded. Exiting simulation.")
        return

    num_simulation_days = len(image_text_pairs_list)
    print(
        f"Starting simulation for {num_simulation_days} days based on image-text pairs..."
    )

    for day_index, pair in enumerate(image_text_pairs_list):
        print(f"Day {day_index + 1}")
        current_image_path = pair.get("image_path")
        current_text_description = pair.get("text_description")
        current_visual_label = pair.get("visual_label")

        if (
            current_image_path is None
            or current_text_description is None
            or current_visual_label is None
        ):
            print(f"Skipping pair at index {day_index} due to missing data: {pair}")
            continue

        current_sensor_data = (
            sensor_data_list[day_index % len(sensor_data_list)]
            if sensor_data_list
            else None
        )

        current_feedback = {
            "vision_label": current_visual_label,
            "memory_target": current_text_description,
            "action_reward": np.random.choice([-1, 0, 1]),  # Frontal lobe reward
            # Other feedback components can remain random/default for now
            "spatial_error": np.random.rand(3).tolist(),
            "motor_command": np.random.rand(3).tolist(),
            "emotion_label": np.random.randint(
                0, getattr(coordinator.limbic, "output_size", 3)
            ),
        }

        result = coordinator.process_day(
            vision_input_path=current_image_path,
            sensor_data=current_sensor_data,
            text_data=current_text_description,
            feedback_data=current_feedback,
        )
        # Ensure result['action'] is subscriptable if it's a list/array, or handle if it's scalar
        action_display = result["action"]
        if isinstance(action_display, (list, np.ndarray)) and len(action_display) > 1:
            action_display = action_display[
                :2
            ]  # Show first 2 elements if it's an array/list

        print(
            f"Processed Pair: Image='{os.path.basename(current_image_path)}', Text='{current_text_description}'. "
            f"VisionPredLabel: {result['vision_label']}, Action: {action_display}, Emotion: {result['emotion']}"
        )
        coordinator.bedtime()


if __name__ == "__main__":
    main()
