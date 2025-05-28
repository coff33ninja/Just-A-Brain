# main.py (Central Coordinator)
"""
Baby AI System - Command-Line Interface

Usage:
  python main.py                # Start interactive simulation (default)
  python main.py --book path.txt  # Train on a book (text file, Q&A log, or story)
  python main.py --help         # Show help and usage

During interactive mode, you can:
- Provide both a 'Text Data' and an 'Expected Response' (Q&A, next-sentence, or dialogue pair)
- Provide a correction if the AI's output is wrong (reinforces correct answer)
- Upload a book/text file for sequential training (see --book)
- (Audio input is not yet supported, but will be in the future)

See README.md for more details.
"""
import sys
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
VISION_FILE = os.path.join(DATA_DIR, "vision.json") # Added for vision data
TEXT_FILE = os.path.join(DATA_DIR, "text.json")     # Added for text data
IMAGE_TEXT_PAIRS_FILE = os.path.join(DATA_DIR, "image_text_pairs.json")
DEFAULT_IMAGE_PATH = "data/images/default_image.png"

def load_vision_data(filepath=VISION_FILE):
    if not os.path.exists(filepath):
        print(f"Warning: Vision data file not found at {filepath}. Returning empty list.")
        return []
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        # Expecting data to be a dict with "image_paths": [...]
        if isinstance(data, dict) and "image_paths" in data and isinstance(data["image_paths"], list):
            return data["image_paths"]
        else:
            print(f"Error: Vision data file {filepath} is not in the expected format (dict with 'image_paths' list). Returning empty list.")
            return []
    except Exception as e:
        print(f"Error loading vision data from {filepath}: {e}. Returning empty list.")
        return []


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

def load_text_data(filepath=TEXT_FILE):
    if not os.path.exists(filepath):
        print(f"Warning: Text data file not found at {filepath}. Returning empty list.")
        return []
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        if not isinstance(data, list): # Expecting a list of strings
            print(f"Error: Text data file {filepath} is not a list. Returning empty list.")
            return []
        return data
    except Exception as e:
        print(f"Error loading text data from {filepath}: {e}. Returning empty list.")
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

        # Episode Tracking Attributes for FrontalLobeAI
        self.episode_length = 5 
        self.steps_since_last_episode_end = 0
        self.last_frontal_state_for_learning = None
        self.last_action_for_learning = None
        self.last_reward_for_learning = None


    def process_day(
        self, vision_input_path, sensor_data, text_data, feedback_data=None,
        language_training_pair=None, correction_text=None
    ):
        current_vision_path = (
            vision_input_path
            if vision_input_path and os.path.exists(vision_input_path)
            else DEFAULT_IMAGE_PATH
        )
        if not os.path.exists(current_vision_path) and vision_input_path:
            print(
                f"Warning: Vision image path {current_vision_path} not found. Using default image."
            )

        current_sensor_data = (
            np.array(sensor_data)
            if sensor_data is not None
            else np.random.randn(getattr(self.parietal, "input_size", 3))
        )
        current_text_data = (
            text_data if isinstance(text_data, str) else "Default sample text"
        )

        # Ensure feedback_data is a dictionary
        current_feedback = feedback_data if feedback_data is not None else {}

        # --- Calculate current_frontal_state (s_t or s_{t+1}) ---
        vision_result_label = self.occipital.process_task(current_vision_path)
        vision_features_for_frontal = np.zeros(self.occipital.output_size)
        if 0 <= vision_result_label < self.occipital.output_size:
            vision_features_for_frontal[vision_result_label] = 1.0

        spatial_result_raw = self.parietal.process_task(current_sensor_data)
        spatial_result_1d = np.array(spatial_result_raw).flatten()
        if spatial_result_1d.shape[0] != 3: # Ensure correct shape for concatenation
            temp_spatial = np.zeros(3)
            min_len_spatial = min(spatial_result_1d.shape[0], 3)
            temp_spatial[:min_len_spatial] = spatial_result_1d[:min_len_spatial]
            spatial_result_1d = temp_spatial

        memory_result_embedding_raw = self.temporal.process_task(
            current_text_data, predict_visual=False
        )
        memory_result_embedding_1d = np.array(memory_result_embedding_raw).flatten()
        if memory_result_embedding_1d.shape[0] != 10: # Ensure correct shape
            temp_memory = np.zeros(10)
            min_len_memory = min(memory_result_embedding_1d.shape[0], 10)
            temp_memory[:min_len_memory] = memory_result_embedding_1d[:min_len_memory]
            memory_result_embedding_1d = temp_memory
        
        current_frontal_state = np.concatenate(
            [vision_features_for_frontal, spatial_result_1d, memory_result_embedding_1d]
        )

        # --- Action Selection (a_t) ---
        action_from_current_state = self.frontal.process_task(current_frontal_state)
        
        # --- Reward Determination (r_{t+1}) ---
        # Default reward if not specified in feedback_data
        current_reward = current_feedback.get("action_reward", 0) 

        # --- Frontal Lobe Learning Step ---
        if self.last_frontal_state_for_learning is not None:
            self.steps_since_last_episode_end += 1
            done_for_learn = self.steps_since_last_episode_end >= self.episode_length

            self.frontal.learn(
                state=self.last_frontal_state_for_learning, # s_t
                action=self.last_action_for_learning,       # a_t
                reward=self.last_reward_for_learning,       # r_t (reward after a_t in s_t)
                next_state=current_frontal_state,           # s_{t+1}
                done=done_for_learn
            )

            if done_for_learn:
                self.steps_since_last_episode_end = 0
        
        # --- Update Stored Values for Next Step's Learning ---
        self.last_frontal_state_for_learning = current_frontal_state
        self.last_action_for_learning = action_from_current_state
        self.last_reward_for_learning = current_reward # This is r_{t+1} which becomes r_t for next cycle

        # --- Other Module Learning ---
        # These use the current inputs and specific feedback for their tasks
        self.occipital.learn(current_vision_path, current_feedback.get("vision_label", 0))
        self.parietal.learn(current_sensor_data, current_feedback.get("spatial_error", np.random.rand(3).tolist()))
        
        temporal_target = current_feedback.get("memory_target", current_text_data)
        self.temporal.learn(
            [(current_text_data, temporal_target)],
            visual_label_as_context=current_feedback.get("vision_label")
        )
        if language_training_pair: # From main loop
            self.temporal.learn(language_training_pair)
        if correction_text and text_data: # From main loop
            self.temporal.learn([(text_data, correction_text)])

        self.cerebellum.learn(current_sensor_data, current_feedback.get("motor_command", np.random.rand(3).tolist()))
        
        # Limbic system processing (after other sensory inputs are processed for context)
        emotion_processing_result = self.limbic.process_task(memory_result_embedding_1d)
        emotion_label = emotion_processing_result["label"]
        emotion_probabilities = emotion_processing_result["probabilities"]
        
        self.limbic.learn(
            memory_result_embedding_1d,
            current_feedback.get("emotion_label", emotion_label), # Use predicted if no specific feedback
            current_reward # Use the general action_reward for limbic learning too
        )

        # --- AI's Textual Response (based on current memory state) ---
        ai_textual_response = None
        current_text_data_str = str(text_data).strip().lower()
        for seq in self.temporal.memory_db:
            for q, a in seq:
                if str(q).strip().lower() == current_text_data_str:
                    ai_textual_response = a
                    break
            if ai_textual_response is not None:
                break
        
        # --- Motor Command (based on current sensor data) ---
        motor_command = self.cerebellum.process_task(current_sensor_data)

        return {
            "action": action_from_current_state, # This is the action chosen for the current state
            "motor": motor_command,
            "emotion_label": emotion_label,
            "emotion_probabilities": emotion_probabilities,
            "vision_label": vision_result_label,
            "ai_textual_response": ai_textual_response
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


def list_learned_qa(coordinator):
    print("\n--- Learned Q&A Pairs (Temporal Lobe) ---")
    if not coordinator.temporal.memory_db:
        print("No Q&A pairs learned yet.")
        return
    for idx, seq in enumerate(coordinator.temporal.memory_db):
        for q, a in seq:
            print(f"{idx+1}. Q: {q}\n   A: {a}")
    print("--- End of Q&A List ---\n")


def main():
    # Introductory Message
    print("Welcome to the Interactive Baby AI Simulation!\n")
    print("This simulation allows you to observe and interact with an AI as it learns.")
    print("You will have options to:")
    print("- Run the simulation for a specific number of 'days'.")
    print("- Run it indefinitely, providing input for each 'day'.")
    print("- For each 'day', you can let the AI process pre-scheduled data, or you can")
    print("  provide new data (images, text, feedback) to guide its learning.\n")
    print(
        "This interactive process allows the AI to remain active and train on data you provide."
    )
    print("Let's get started!")
    print("-------------------------------------------------------------------\n")

    coordinator = BrainCoordinator()
    image_text_pairs_list = load_image_text_pairs()
    sensor_data_list = load_sensor_data()

    # Initialize variables to store the last used data
    last_image_path = DEFAULT_IMAGE_PATH
    last_text_description = "Default sample text"
    last_sensor_data = None # Will be initialized with random if not provided
    last_visual_label = 0
    last_action_reward = 0 # For feedback

    image_path_for_processing = last_image_path
    text_description_for_processing = last_text_description
    sensor_data_for_processing = last_sensor_data
    visual_label_for_processing = last_visual_label
    action_reward_for_processing = last_action_reward


    # Prompt for simulation mode
    num_simulation_days = float("inf")  # Default to indefinite
    run_mode_input = (
        input(
            "Run for a specific number of days (d) or run indefinitely in interactive mode (i)? (Default: i): "
        )
        .lower()
        .strip()
    )

    if run_mode_input == "d":
        try:
            days_input_str = input("Enter the number of days to simulate: ").strip()
            n_days = int(days_input_str)
            if n_days > 0:
                num_simulation_days = n_days
            else:
                default_days = (
                    len(image_text_pairs_list) if image_text_pairs_list else 1
                )
                print(
                    f"Number of days must be positive. Defaulting to {default_days} day(s)."
                )
                num_simulation_days = default_days
        except ValueError:
            default_days = len(image_text_pairs_list) if image_text_pairs_list else 1
            print(f"Invalid input. Defaulting to {default_days} day(s).")
            num_simulation_days = default_days
    elif run_mode_input == "i" or not run_mode_input:  # Default to indefinite
        print("Running indefinitely.")
    else:  # Invalid mode selection
        print("Invalid mode selected. Defaulting to run indefinitely.")

    day_index = 0
    user_choice_for_current_day_data_source = "n"

    while day_index < num_simulation_days:
        day_display_str = f"--- Day {day_index + 1}"
        if num_simulation_days != float("inf"):
            day_display_str += (
                f"/{int(num_simulation_days)}" 
            )
        day_display_str += " ---"
        print(f"\n{day_display_str}")

        expected_response = None
        # audio_input_path = None # Not used yet

        if user_choice_for_current_day_data_source == "i":
            print(
                "Provide new input for the current day (press Enter to use defaults):"
            )

            new_image_path_input = input(
                f"Enter new image path (default: '{last_image_path}'): "
            ).strip()
            if new_image_path_input:
                if os.path.exists(new_image_path_input):
                    image_path_for_processing = new_image_path_input
                else:
                    print(
                        f"Warning: Path '{new_image_path_input}' not found. Using last: '{last_image_path}'."
                    )
                    image_path_for_processing = last_image_path
            else:
                image_path_for_processing = last_image_path

            new_text_description_input = input(
                f"Enter new text (default: '{last_text_description}'): "
            ).strip()
            text_description_for_processing = (
                new_text_description_input
                if new_text_description_input
                else last_text_description
            )

            expected_response_input = input(
                "Enter expected response (answer, next sentence, or target text) [optional]: "
            ).strip()
            if expected_response_input:
                expected_response = expected_response_input

            new_visual_label_input = input(
                f"Enter new visual label (int, default: {last_visual_label}): "
            ).strip()
            if new_visual_label_input:
                try:
                    visual_label_for_processing = int(new_visual_label_input)
                except ValueError:
                    print(f"Warning: Invalid label. Using last: {last_visual_label}.")
                    visual_label_for_processing = last_visual_label
            else:
                visual_label_for_processing = last_visual_label
            
            new_action_reward_input = input(
                f"Enter action reward for Frontal Lobe (e.g., -1, 0, 1, default: {last_action_reward}): "
            ).strip()
            if new_action_reward_input:
                try:
                    action_reward_for_processing = int(new_action_reward_input)
                except ValueError:
                    print(f"Warning: Invalid reward. Using last: {last_action_reward}.")
                    action_reward_for_processing = last_action_reward
            else:
                action_reward_for_processing = last_action_reward


            generate_new_sensor_input = (
                input(
                    "Generate new random sensor data? (y/n, default n - uses last): "
                )
                .strip()
                .lower()
            )
            if generate_new_sensor_input == "y":
                sensor_data_for_processing = None # Will generate random in process_day
            else:
                sensor_data_for_processing = last_sensor_data
            
            # audio_input_path = input("Enter path to audio file for input (optional, not yet used): ").strip()
            # if audio_input_path:
            #     print(f"Audio input received: {audio_input_path} (audio training not yet implemented)")

        else:  # 'n' - Use scheduled data
            if not image_text_pairs_list:
                if day_index == 0:
                    print("No scheduled image-text pairs available for Day 1.")
                    print(
                        "Simulation cannot proceed without initial data for Day 1 if not providing input."
                    )
                    first_day_input_choice = (
                        input("Provide input for Day 1 (i) or quit (q)? ")
                        .lower()
                        .strip()
                    )
                    if first_day_input_choice == "i":
                        user_choice_for_current_day_data_source = "i"
                        continue
                    else:
                        print("Quitting simulation.")
                        break
                else: 
                    print(
                        "No more scheduled image-text pairs available to process with 'n'."
                    )
                    pass # Will go to next action prompt

            if image_text_pairs_list: 
                current_pair_index = day_index % len(image_text_pairs_list)
                pair = image_text_pairs_list[current_pair_index]

                scheduled_image_path = pair.get("image_path")
                scheduled_text_description = pair.get("text_description")
                scheduled_visual_label = pair.get("visual_label")
                # For scheduled data, let's use a default reward or make it part of the schedule if needed
                action_reward_for_processing = pair.get("action_reward", 0) 


                if (
                    scheduled_image_path is None
                    or scheduled_text_description is None
                    or scheduled_visual_label is None
                ):
                    print(
                        f"Data missing in scheduled pair at data index {current_pair_index}: {pair}"
                    )
                    missing_data_action = (
                        input(
                            "Next day (n), provide input for this day (i), or quit (q)? "
                        )
                        .lower()
                        .strip()
                    )
                    if missing_data_action == "q":
                        break
                    elif missing_data_action == "i":
                        user_choice_for_current_day_data_source = "i"
                        continue
                    else: 
                        if missing_data_action != "n":
                            print("Invalid input. Proceeding to next day.")
                        day_index += 1 
                        user_choice_for_current_day_data_source = "n"
                        if (
                            day_index >= num_simulation_days
                        ): 
                            print(
                                f"Specified number of simulation days ({int(num_simulation_days)}) reached by skipping."
                            )
                            break
                        continue 

                image_path_for_processing = scheduled_image_path
                text_description_for_processing = scheduled_text_description
                visual_label_for_processing = scheduled_visual_label

                if sensor_data_list:
                    sensor_data_for_processing = sensor_data_list[
                        day_index % len(sensor_data_list)
                    ]
                else:
                    print(
                        "Warning: Sensor data list is empty. Using random sensor data for this day."
                    )
                    sensor_data_for_processing = None # Will generate random
            elif (
                not image_text_pairs_list
                and user_choice_for_current_day_data_source == "n"
            ):
                print(
                    "Internal state: 'n' chosen but no scheduled data. Re-evaluating action."
                )
                continue

        # --- Common Processing Logic ---
        # current_feedback is now built inside process_day based on its needs.
        # We pass the specific feedback items needed.
        current_feedback_for_day = {
            "vision_label": visual_label_for_processing,
            "memory_target": text_description_for_processing, # Default target if no explicit expected_response
            "action_reward": action_reward_for_processing, # This is the crucial reward for frontal.learn
            # Other feedback items can be defaulted or randomized inside process_day if not provided here
            "spatial_error": np.random.rand(3).tolist(), 
            "motor_command": np.random.rand(3).tolist(),
            "emotion_label": np.random.randint(0, getattr(coordinator.limbic, "output_size", 3)),
        }
        if expected_response: # If user provided an explicit expected response, use it as memory_target
            current_feedback_for_day["memory_target"] = expected_response


        img_display_name = (
            os.path.basename(image_path_for_processing)
            if image_path_for_processing
            else "N/A"
        )
        print(
            f"Processing: Image='{img_display_name}', Text='{text_description_for_processing}', Action Reward='{action_reward_for_processing}'"
        )

        language_training_pair = None
        if text_description_for_processing and expected_response:
            language_training_pair = [(text_description_for_processing, expected_response)]
        # Removed the else case that defaulted to (text, text) as it's less useful for Q&A style

        list_qa_input = input("Type 'list' to see all learned Q&A pairs, or press Enter to continue: ").strip().lower()
        if list_qa_input == "list":
            list_learned_qa(coordinator)

        result = coordinator.process_day(
            vision_input_path=image_path_for_processing,
            sensor_data=sensor_data_for_processing,
            text_data=text_description_for_processing,
            feedback_data=current_feedback_for_day, # Pass the constructed feedback
            language_training_pair=language_training_pair,
            correction_text=None 
        )
        
        ai_textual_response_from_day = result.get("ai_textual_response")
        if ai_textual_response_from_day is not None:
            print(f"[Memory] The AI's current answer for '{text_description_for_processing}': {ai_textual_response_from_day}")
        else:
            print(f"[Memory] The AI has no specific answer for '{text_description_for_processing}' after this cycle.")

        last_image_path = image_path_for_processing
        last_text_description = text_description_for_processing
        last_sensor_data = sensor_data_for_processing # Save the actual sensor data used
        last_visual_label = visual_label_for_processing
        last_action_reward = action_reward_for_processing


        action_res = result["action"]
        # Simplified action display for clarity
        print(
            f"Result: VisionPredLabel: {result['vision_label']}, Action: {action_res}, Emotion: {result['emotion_label']}"
        )

        if expected_response:
            if ai_textual_response_from_day and ai_textual_response_from_day.strip().lower() == expected_response.strip().lower():
                print("[Check] The AI's answer matches the expected response. No correction needed.")
                reinforce = input("Do you still want to reinforce this answer? (y/n, default n): ").strip().lower()
                if reinforce == "y":
                    # Reinforce with the original question and the (correct) expected response
                    coordinator.temporal.learn([(text_description_for_processing, expected_response)])
            else:
                print("[Check] The AI's answer does NOT match the expected response.")
                print(f"AI's answer: {ai_textual_response_from_day if ai_textual_response_from_day is not None else '[No answer]'}")
                print(f"Expected: {expected_response}")
                print("Reinforcing correct answer...")
                coordinator.temporal.learn([(text_description_for_processing, expected_response)])
        
        correction_text_input = input("If the AI output was wrong, enter the correct response here (or press Enter to skip): ").strip()
        if correction_text_input and text_description_for_processing:
            print("Reinforcing correct answer with correction text...")
            # Use the original input text as the "question" and the correction as the "answer"
            coordinator.process_day(
                vision_input_path=last_image_path, # Use last context
                sensor_data=last_sensor_data,
                text_data=text_description_for_processing, # Original question
                feedback_data={"memory_target": correction_text_input, "action_reward": 1}, # Positive reward for correction
                correction_text=correction_text_input # Explicitly pass for temporal.learn
            )


        coordinator.bedtime()

        day_index += 1 

        if day_index >= num_simulation_days:
            if num_simulation_days != float(
                "inf"
            ): 
                print(
                    f"Specified number of simulation days ({int(num_simulation_days)}) reached."
                )
            else: 
                print(
                    "Simulation cycle complete (indefinite run ended unexpectedly here)."
                )
            break

        while True:
            can_use_scheduled_data = bool(
                image_text_pairs_list and (day_index < len(image_text_pairs_list) if num_simulation_days == float('inf') else True)
            ) 

            if can_use_scheduled_data:
                next_action_prompt = f"Next day ({day_index + 1}) using scheduled data (n), provide input (i), or quit (q)? "
            else: 
                next_action_prompt = f"No more scheduled data. Provide input for next day ({day_index + 1}) (i) or quit (q)? "

            user_input_next_action = input(next_action_prompt).lower().strip()

            if user_input_next_action == "n":
                if can_use_scheduled_data:
                    user_choice_for_current_day_data_source = "n"
                    break
                else:
                    print(
                        "Invalid choice: No scheduled data available for 'n'. Please choose 'i' or 'q'."
                    )
            elif user_input_next_action == "i":
                user_choice_for_current_day_data_source = "i"
                break
            elif user_input_next_action == "q":
                user_choice_for_current_day_data_source = "q"
                break
            else:
                print("Invalid input. Please enter 'n', 'i', or 'q' (if available).")

        if user_choice_for_current_day_data_source == "q":
            break

    print("Simulation ended.")

def train_on_book_cli(book_file_path, coordinator):
    if not os.path.exists(book_file_path):
        print(f"Book file '{book_file_path}' not found.")
        return
    with open(book_file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if len(paragraphs) < 2:
        import re
        paragraphs = re.split(r'(?<=[.!?]) +', text)
    pairs = []
    for i in range(len(paragraphs) - 1):
        pairs.append((paragraphs[i], paragraphs[i+1]))
    if not pairs:
        print("Book is too short for training.")
        return
    print(f"Training on {len(pairs)} text pairs from book...")
    for day_idx, pair in enumerate(pairs):
        print(f"Book training day {day_idx + 1}/{len(pairs)}")
        # Simulate a "day" for each pair to allow FrontalLobe to also learn from sequences
        coordinator.process_day(
            vision_input_path=DEFAULT_IMAGE_PATH, # Default visual
            sensor_data=None, # Random sensor
            text_data=pair[0], # Current paragraph as "input"
            feedback_data={
                "memory_target": pair[1], # Next paragraph as "target"
                "action_reward": 1, # Assume positive reward for sequential text
                "vision_label": 0, # Default
                "spatial_error": np.random.rand(3).tolist(),
                "motor_command": np.random.rand(3).tolist(),
                "emotion_label": np.random.randint(0, getattr(coordinator.limbic, "output_size", 3))
            },
            language_training_pair=[pair] # Explicitly pass for temporal.learn
        )
        if (day_idx + 1) % 10 == 0: # Consolidate every 10 pairs
            coordinator.bedtime()
    coordinator.bedtime() # Final consolidation
    print("Book training complete.")

if __name__ == "__main__":
    if '--help' in sys.argv or '-h' in sys.argv:
        print(__doc__)
        sys.exit(0)
    
    coordinator = BrainCoordinator() # Create coordinator once

    if '--book' in sys.argv:
        idx = sys.argv.index('--book')
        if idx + 1 < len(sys.argv):
            book_path = sys.argv[idx + 1]
            train_on_book_cli(book_path, coordinator) # Pass coordinator
            sys.exit(0)
        else:
            print("Usage: python main.py --book path/to/book.txt")
            sys.exit(1)
    
    main()
