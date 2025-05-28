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
    """
    Loads a list of image paths from a vision data JSON file.
    
    If the file does not exist, is not a dictionary with an "image_paths" list, or cannot be read, returns an empty list and prints a warning or error message.
    """
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
    """
    Loads sensor data from a JSON file.
    
    If the file does not exist or cannot be loaded, returns an empty list and prints a warning or error message.
    """
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
    """
    Loads text data from a JSON file, expecting a list of strings.
    
    If the file does not exist, is not a list, or cannot be read, returns an empty list and prints a warning or error message.
    """
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
    """
    Loads image-text pair data from a JSON file.
    
    If the file does not exist, is not a list, or cannot be loaded, returns an empty list.
    """
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
        """
        Initializes the BrainCoordinator by creating necessary data directories and instantiating all AI brain region modules.
        """
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR, "images"), exist_ok=True)
        self.frontal = FrontalLobeAI()
        self.parietal = ParietalLobeAI()
        self.temporal = TemporalLobeAI()
        self.occipital = OccipitalLobeAI()
        self.cerebellum = CerebellumAI()
        self.limbic = LimbicSystemAI()

    def process_day(
        self, vision_input_path, sensor_data, text_data, feedback_data=None,
        language_training_pair=None, correction_text=None
    ):
        """
        Processes a single simulation day by coordinating all AI brain region modules.
        
        Args:
            vision_input_path: Path to the vision input image file.
            sensor_data: Sensor data array or list for the current day.
            text_data: Textual input for the current day.
            feedback_data: Optional dictionary with feedback signals (e.g., rewards, labels).
            language_training_pair: Optional list of (question, answer) pairs for language training.
            correction_text: Optional correction text to reinforce learning if the AI's response was incorrect.
        
        Returns:
            A dictionary containing the action, motor command, emotion label, emotion probabilities,
            predicted vision label, and the AI's textual response from memory if available.
        """
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

        current_feedback = feedback_data if feedback_data is not None else {}

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

        vision_result_label = self.occipital.process_task(current_vision_path)
        vision_features_for_frontal = np.zeros(self.occipital.output_size)
        if 0 <= vision_result_label < self.occipital.output_size:
            vision_features_for_frontal[vision_result_label] = 1.0

        spatial_result_raw = self.parietal.process_task(current_sensor_data)
        spatial_result_1d = np.array(spatial_result_raw).flatten()
        if spatial_result_1d.shape[0] != 3:
            temp_spatial = np.zeros(3)
            min_len_spatial = min(spatial_result_1d.shape[0], 3)
            temp_spatial[:min_len_spatial] = spatial_result_1d[:min_len_spatial]
            spatial_result_1d = temp_spatial

        memory_result_embedding_raw = self.temporal.process_task(
            current_text_data, predict_visual=False
        )
        memory_result_embedding_1d = np.array(memory_result_embedding_raw).flatten()
        if memory_result_embedding_1d.shape[0] != 10:
            temp_memory = np.zeros(10)
            min_len_memory = min(memory_result_embedding_1d.shape[0], 10)
            temp_memory[:min_len_memory] = memory_result_embedding_1d[:min_len_memory]
            memory_result_embedding_1d = temp_memory

        concatenated_input_for_frontal = np.concatenate(
            [vision_features_for_frontal, spatial_result_1d, memory_result_embedding_1d]
        )
        frontal_state = concatenated_input_for_frontal
        action = self.frontal.process_task(frontal_state)
        motor_command = self.cerebellum.process_task(current_sensor_data)
        # Limbic system processing
        emotion_processing_result = self.limbic.process_task(memory_result_embedding_1d)
        emotion_label = emotion_processing_result["label"]
        emotion_probabilities = emotion_processing_result["probabilities"]

        state_for_learn = frontal_state
        action_for_learn = action
        reward_for_learn = feedback.get("action_reward", 0)
        next_state_for_learn = frontal_state
        done_for_learn = True

        self.frontal.learn(
            state_for_learn,
            action_for_learn,
            reward_for_learn,
            next_state_for_learn,
            done_for_learn,
        )
        self.occipital.learn(current_vision_path, feedback["vision_label"])
        self.parietal.learn(current_sensor_data, feedback["spatial_error"])
        self.temporal.learn(
            [(current_text_data, feedback["memory_target"])],
            visual_label_as_context=feedback.get("vision_label"),
        )
        # New: richer language/Q&A training
        if language_training_pair:
            self.temporal.learn(language_training_pair)
        if correction_text and text_data:
            self.temporal.learn([(text_data, correction_text)])
        self.cerebellum.learn(current_sensor_data, feedback["motor_command"])
        self.limbic.learn(
            memory_result_embedding_1d,
            feedback.get("emotion_label", emotion_label), # Use predicted label if feedback not provided
            feedback["action_reward"],
        )

        # Determine AI's textual response based on current memory (after learning for this cycle)
        ai_textual_response = None
        # Ensure current_text_data is a string for comparison
        current_text_data_str = str(text_data).strip().lower() # Use the input text_data for lookup
        for seq in self.temporal.memory_db:
            for q, a in seq:
                if str(q).strip().lower() == current_text_data_str:
                    ai_textual_response = a
                    break
            if ai_textual_response is not None: # Check for not None, as empty string is a valid answer
                break
        # If no match, ai_textual_response remains None

        return {
            "action": action,
            "motor": motor_command,
            "emotion_label": emotion_label, # Keep the label
            "emotion_probabilities": emotion_probabilities, # Add probabilities
            "vision_label": vision_result_label,
            "ai_textual_response": ai_textual_response # Added here
        }


    def bedtime(self):
        """
        Performs end-of-day consolidation for all AI brain region modules.
        
        Triggers the consolidate method on each brain region AI module to update and reinforce learned information at the end of a simulation day.
        """
        print("Starting bedtime consolidation...")
        self.frontal.consolidate()
        self.parietal.consolidate()
        self.temporal.consolidate()
        self.occipital.consolidate()
        self.cerebellum.consolidate()
        self.limbic.consolidate()
        print("Consolidation complete.")


def list_learned_qa(coordinator):
    """
    Displays all learned question-and-answer pairs stored in the temporal AI's memory database.
    
    Args:
        coordinator: The BrainCoordinator instance containing the temporal AI module.
    """
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
    """
    Runs the interactive Baby AI simulation, allowing users to guide and observe AI learning.
    
    This function launches a command-line interface where users can simulate the Baby AI system over multiple days. Users may choose to run the simulation for a fixed number of days or indefinitely, and for each day, can either provide new input data (images, text, feedback) or use pre-scheduled data if available. The simulation supports reinforcement and correction of AI responses, displays learned Q&A pairs, and provides feedback on the AI's actions, vision predictions, and emotions. The simulation loop continues until the specified number of days is reached or the user chooses to quit.
    """
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
    last_sensor_data = None
    last_visual_label = 0

    image_path_for_processing = last_image_path
    text_description_for_processing = last_text_description
    sensor_data_for_processing = last_sensor_data
    visual_label_for_processing = last_visual_label

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
    # user_choice_for_current_day_data_source is what the user selected at the end of the *previous*
    # iteration to determine data source for the *current* day_index.
    # It's 'n' initially to use scheduled data for the first day, unless overridden by specific conditions.
    user_choice_for_current_day_data_source = "n"

    while day_index < num_simulation_days:
        day_display_str = f"--- Day {day_index + 1}"
        if num_simulation_days != float("inf"):
            day_display_str += (
                f"/{int(num_simulation_days)}"  # Ensure Y is int for display
            )
        day_display_str += " ---"
        print(f"\n{day_display_str}")

        expected_response = None
        audio_input_path = None

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

            # Q&A/Dialogue/Next-sentence training
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

            generate_new_sensor_input = (
                input(
                    "Generate new random sensor data? (y/n, default n - uses last): "
                )
                .strip()
                .lower()
            )
            if generate_new_sensor_input == "y":
                sensor_data_for_processing = None
            else:
                sensor_data_for_processing = last_sensor_data

            # Audio input placeholder
            audio_input_path = input("Enter path to audio file for input (optional, not yet used): ").strip()
            if audio_input_path:
                print(f"Audio input received: {audio_input_path} (audio training not yet implemented)")

        else:  # 'n' - Use scheduled data
            if not image_text_pairs_list:
                # This condition means there's no scheduled data AT ALL, or we've run out.
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
                        # day_index remains 0, next iteration will hit the 'i' block above
                        continue
                    else:
                        print("Quitting simulation.")
                        break
                else:  # Not the first day, but no scheduled data (e.g. list exhausted)
                    print(
                        "No more scheduled image-text pairs available to process with 'n'."
                    )
                    # The loop will go to the next action prompt below.
                    # If user then selects 'n', it will be caught by the more specific prompt there.
                    pass

            if image_text_pairs_list:  # Proceed to try and use scheduled data
                current_pair_index = day_index % len(image_text_pairs_list)
                pair = image_text_pairs_list[current_pair_index]

                scheduled_image_path = pair.get("image_path")
                scheduled_text_description = pair.get("text_description")
                scheduled_visual_label = pair.get("visual_label")

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
                        # day_index is not incremented, next iteration for same day_index hits 'i' block
                        continue
                    else:  # 'n' or invalid
                        if missing_data_action != "n":
                            print("Invalid input. Proceeding to next day.")
                        day_index += 1  # Advance day_index to skip this problematic scheduled day
                        user_choice_for_current_day_data_source = (
                            "n"  # Ensure next is 'n'
                        )
                        if (
                            day_index >= num_simulation_days
                        ):  # Check if skipping made us reach the end
                            print(
                                f"Specified number of simulation days ({int(num_simulation_days)}) reached by skipping."
                            )
                            break
                        continue  # To the top of the while loop for the new day_index

                image_path_for_processing = scheduled_image_path
                text_description_for_processing = scheduled_text_description
                visual_label_for_processing = scheduled_visual_label

                if sensor_data_list:
                    sensor_data_for_processing = sensor_data_list[
                        day_index % len(sensor_data_list)
                    ]
                else:
                    # This warning applies if scheduled image/text is used but sensor_data_list is empty
                    print(
                        "Warning: Sensor data list is empty. Using random sensor data for this day."
                    )
                    sensor_data_for_processing = None
            elif (
                not image_text_pairs_list
                and user_choice_for_current_day_data_source == "n"
            ):
                # This means image_text_pairs_list is empty (and was from start of this iteration or before)
                # AND user wanted 'n'. Since there's no scheduled data, this 'n' cannot be fulfilled.
                # This case is now primarily handled by the prompt at the end of the loop.
                # If we reach here, it implies a state where 'n' was chosen but no data exists.
                # The prompt at the end of the loop will force 'i' or 'q'.
                print(
                    "Internal state: 'n' chosen but no scheduled data. Re-evaluating action."
                )
                continue

        # --- Common Processing Logic ---
        current_feedback = {
            "vision_label": visual_label_for_processing,
            "memory_target": text_description_for_processing,
            "action_reward": np.random.choice([-1, 0, 1]),
            "spatial_error": np.random.rand(3).tolist(),
            "motor_command": np.random.rand(3).tolist(),
            "emotion_label": np.random.randint(
                0, getattr(coordinator.limbic, "output_size", 3)
            ),
        }

        img_display_name = (
            os.path.basename(image_path_for_processing)
            if image_path_for_processing
            else "N/A"
        )
        print(
            f"Processing: Image='{img_display_name}', Text='{text_description_for_processing}'"
        )

        # Prepare language training pair
        language_training_pair = None
        if text_description_for_processing and expected_response:
            language_training_pair = [(text_description_for_processing, expected_response)]
        elif text_description_for_processing:
            language_training_pair = [(text_description_for_processing, text_description_for_processing)]

        # New: Option to list learned Q&A pairs
        list_qa_input = input("Type 'list' to see all learned Q&A pairs, or press Enter to continue: ").strip().lower()
        if list_qa_input == "list":
            list_learned_qa(coordinator)

        result = coordinator.process_day(
            vision_input_path=image_path_for_processing,
            sensor_data=sensor_data_for_processing,
            text_data=text_description_for_processing,
            feedback_data=current_feedback,
            language_training_pair=language_training_pair,
            correction_text=None  # Will prompt for correction after output
        )
        
        # Get the AI's textual response from the result
        ai_textual_response_from_day = result.get("ai_textual_response")
        if ai_textual_response_from_day is not None:
            print(f"[Memory] The AI's current answer for '{text_description_for_processing}': {ai_textual_response_from_day}")
        else:
            print(f"[Memory] The AI has no specific answer for '{text_description_for_processing}' after this cycle.")

        last_image_path = image_path_for_processing
        last_text_description = text_description_for_processing
        last_sensor_data = sensor_data_for_processing
        last_visual_label = visual_label_for_processing

        action_res = result["action"]
        action_display_str = ""
        if isinstance(action_res, (list, np.ndarray)):
            action_display_str = (
                str(action_res[:2])
                if len(action_res) > 1
                else str(action_res[0] if len(action_res) == 1 else "")
            )
        else:
            action_display_str = str(action_res)

        print(
            f"Result: VisionPredLabel: {result['vision_label']}, Action: {action_display_str}, Emotion: {result['emotion']}"
        )

        # Compare AI output to expected/correct answer
        # Use ai_textual_response_from_day for comparison
        if expected_response:
            if ai_textual_response_from_day and ai_textual_response_from_day.strip().lower() == expected_response.strip().lower():
                print("[Check] The AI's answer matches the expected response. No correction needed.")
                reinforce = input("Do you still want to reinforce this answer? (y/n, default n): ").strip().lower()
                if reinforce == "y":
                    coordinator.temporal.learn([(text_description_for_processing, expected_response)])
            else:
                print("[Check] The AI's answer does NOT match the expected response.")
                print(f"AI's answer: {ai_textual_response_from_day if ai_textual_response_from_day is not None else '[No answer]'}")
                print(f"Expected: {expected_response}")
                print("Reinforcing correct answer...")
                coordinator.temporal.learn([(text_description_for_processing, expected_response)])

        # Correction/feedback mechanism
        correction_text_input = input("If the AI output was wrong, enter the correct response here (or press Enter to skip): ").strip()
        if correction_text_input and text_description_for_processing:
            print("Reinforcing correct answer with correction text...")
            coordinator.temporal.learn([(text_description_for_processing, correction_text_input)])

        coordinator.bedtime()

        day_index += 1  # CRITICAL: Increment day_index after all processing for the current day number is complete

        if day_index >= num_simulation_days:
            if num_simulation_days != float(
                "inf"
            ):  # Only print if it's a finite number of days
                print(
                    f"Specified number of simulation days ({int(num_simulation_days)}) reached."
                )
            else:  # Should not happen if inf, but as a safeguard
                print(
                    "Simulation cycle complete (indefinite run ended unexpectedly here)."
                )
            break

        # User prompt for next action (for the *next* day_index)
        while True:
            can_use_scheduled_data = bool(
                image_text_pairs_list
            )  # True if there's any scheduled data at all

            if can_use_scheduled_data:
                next_action_prompt = f"Next day ({day_index + 1}) using scheduled data (n), provide input (i), or quit (q)? "
            else:  # No scheduled data left or never existed
                next_action_prompt = f"No scheduled data. Provide input for next day ({day_index + 1}) (i) or quit (q)? "

            user_input_next_action = input(next_action_prompt).lower().strip()

            if user_input_next_action == "n":
                if can_use_scheduled_data:
                    user_choice_for_current_day_data_source = "n"
                    break
                else:
                    print(
                        "Invalid choice: No scheduled data available for 'n'. Please choose 'i' or 'q'."
                    )
                    # Loop continues for this prompt
            elif user_input_next_action == "i":
                user_choice_for_current_day_data_source = "i"
                break
            elif user_input_next_action == "q":
                user_choice_for_current_day_data_source = (
                    "q"  # Will cause outer loop to break
                )
                break
            else:
                print("Invalid input. Please enter 'n', 'i', or 'q' (if available).")

        if user_choice_for_current_day_data_source == "q":
            break
        # If 'n' or 'i', the loop continues. user_choice_for_current_day_data_source will determine
        # data source at the start of the next iteration for the new day_index.

    print("Simulation ended.")

def train_on_book_cli(book_file_path, coordinator):
    """
    Trains the temporal AI module on sequential text pairs extracted from a book file.
    
    Reads the specified book file, splits its content into paragraphs or sentences, forms consecutive text pairs, and uses them to train the temporal AI's language memory. Prints progress and completion messages.
    """
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
    for pair in pairs:
        coordinator.temporal.learn([pair])
    print("Book training complete.")

if __name__ == "__main__":
    if '--help' in sys.argv or '-h' in sys.argv:
        print(__doc__)
        sys.exit(0)
    if '--book' in sys.argv:
        idx = sys.argv.index('--book')
        if idx + 1 < len(sys.argv):
            book_path = sys.argv[idx + 1]
            coordinator = BrainCoordinator()
            train_on_book_cli(book_path, coordinator)
            sys.exit(0)
        else:
            print("Usage: python main.py --book path/to/book.txt")
            sys.exit(1)
    # ...existing code for main()...
    main()
