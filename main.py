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
        emotion = self.limbic.process_task(memory_result_embedding_1d)

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
                    f"Generate new random sensor data? (y/n, default n - uses last): "
                )
                .strip()
                .lower()
            )
            if generate_new_sensor_input == "y":
                sensor_data_for_processing = None
            else:
                sensor_data_for_processing = last_sensor_data

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
                # We add a continue to re-evaluate at the top, which will then lead to that prompt.
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

        result = coordinator.process_day(
            vision_input_path=image_path_for_processing,
            sensor_data=sensor_data_for_processing,
            text_data=text_description_for_processing,
            feedback_data=current_feedback,
        )

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


if __name__ == "__main__":
    main()
