import gradio as gr
import os
import sys
import json
import numpy as np
from io import StringIO
import traceback

# Add project root to sys.path to allow imports from other project files
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from main import BrainCoordinator, DEFAULT_IMAGE_PATH

# Global instance of BrainCoordinator
# This ensures the AI models retain their state across Gradio interactions
print("Initializing BrainCoordinator for Gradio GUI...")
coordinator = BrainCoordinator()
print("BrainCoordinator initialized.")

# Helper to parse JSON string input for lists
def parse_json_list(json_string, default_value, expected_type=list):
    if not json_string: # If string is empty or None
        return default_value
    try:
        parsed = json.loads(json_string)
        if isinstance(parsed, expected_type):
            return parsed
        else:
            print(f"Warning: Parsed JSON '{json_string}' is not of type {expected_type}. Got {type(parsed)}. Using default.")
            return default_value
    except json.JSONDecodeError:
        print(f"Warning: Invalid JSON string '{json_string}'. Using default.")
        return default_value

# Wrapper function for Gradio interface
def run_ai_day_interface(
    vision_input_path,
    text_data,
    sensor_data_str,
    action_reward,
    spatial_error_str,
    memory_target,
    vision_label,
    motor_command_str,
    emotion_label,
):
    # Capture stdout for logging
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    try:
        # Prepare inputs
        current_vision_path = vision_input_path if vision_input_path and os.path.exists(vision_input_path) else DEFAULT_IMAGE_PATH

        if sensor_data_str:
            sensor_data = parse_json_list(sensor_data_str, None)
            if sensor_data is None:
                 print(f"Using random sensor data as input string '{sensor_data_str}' was empty or invalid.")
                 sensor_data = np.random.randn(coordinator.parietal.input_size).tolist()
        else:
            print("No sensor data provided by user, generating random sensor data.")
            sensor_data = np.random.randn(coordinator.parietal.input_size).tolist()

        # Feedback data construction
        feedback_data = {
            "action_reward": int(action_reward),
            "spatial_error": parse_json_list(spatial_error_str, np.random.rand(coordinator.parietal.output_size).tolist()),
            "memory_target": memory_target if memory_target else text_data,
            "vision_label": int(vision_label),
            "motor_command": parse_json_list(motor_command_str, np.random.rand(coordinator.cerebellum.output_size).tolist()),
            "emotion_label": int(emotion_label),
        }

        print(f"--- GUI: Processing Day ---")
        print(f"Vision Path: {os.path.basename(current_vision_path) if current_vision_path else 'N/A'}")
        print(f"Text Data: '{text_data}'")
        print(f"Sensor Data (first 5): {sensor_data[:5] if sensor_data else 'N/A'}...")
        print(f"Feedback Data: {feedback_data}")

        results = coordinator.process_day(
            vision_input_path=current_vision_path,
            sensor_data=sensor_data,
            text_data=text_data,
            feedback_data=feedback_data,
        )

        print(f"--- GUI: Day Processed. Results: {results} ---")
        print(f"--- GUI: Starting Bedtime Consolidation ---")
        coordinator.bedtime()
        print(f"--- GUI: Bedtime Consolidation Complete ---")

        log_output = captured_output.getvalue()
        sys.stdout = old_stdout
        return results, log_output

    except Exception as e:
        sys.stdout = old_stdout
        error_log = captured_output.getvalue()
        error_message = f"An error occurred: {str(e)}\nTraceback:\n{traceback.format_exc()}\nLog:\n{error_log}"
        print(error_message)
        return {"error": str(e), "details": traceback.format_exc()}, error_message

# Define Gradio Inputs
vision_input_path_component = gr.Image(type="filepath", label="Vision Input Image", value=DEFAULT_IMAGE_PATH)
text_data_component = gr.Textbox(label="Text Data", value="A small red block is on the table.", lines=2)
sensor_data_component = gr.Textbox(
    label=f"Sensor Data (JSON list)",
    placeholder=f"e.g., a list of {coordinator.parietal.input_size} floats like [0.1, 0.2, ...]. If blank, random data is used.",
    lines=2
)

feedback_action_reward_component = gr.Radio([-1, 0, 1], label="Feedback: Action Reward", value=0, type="value")
feedback_spatial_error_component = gr.Textbox(
    label=f"Feedback: Spatial Error (JSON list)",
    placeholder=f"e.g., list of {coordinator.parietal.output_size} floats like [0.0, 0.1, -0.05]. If blank, random.",
    lines=1
)
feedback_memory_target_component = gr.Textbox(label="Feedback: Memory Target Text", lines=2, placeholder="Text AI should learn for input. If blank, uses input Text Data.")
feedback_vision_label_component = gr.Number(label="Feedback: True Vision Label (int)", value=0, precision=0)
feedback_motor_command_component = gr.Textbox(
    label=f"Feedback: True Motor Command (JSON list)",
    placeholder=f"e.g., list of {coordinator.cerebellum.output_size} floats like [0.5, -0.2, 0.0]. If blank, random.",
    lines=1
)
feedback_emotion_label_component = gr.Number(label="Feedback: True Emotion Label (int)", value=0, precision=0)

# Define Gradio Outputs
results_component = gr.JSON(label="Processing Results")
log_component = gr.Textbox(label="Log Output", lines=20, interactive=False, autoscroll=True)

inputs_list = [
    vision_input_path_component,
    text_data_component,
    sensor_data_component,
    feedback_action_reward_component,
    feedback_spatial_error_component,
    feedback_memory_target_component,
    feedback_vision_label_component,
    feedback_motor_command_component,
    feedback_emotion_label_component,
]
outputs_list = [results_component, log_component]

with gr.Blocks(title="Baby AI Interactive Simulation", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Baby AI Interactive Simulation")
    gr.Markdown("Interact with the AI by providing inputs for a 'day' of experience and observe its learning. All AI model weights are saved after each day's consolidation.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Sensory Inputs")
            vision_input_path_component.render()
            text_data_component.render()
            sensor_data_component.render()

        with gr.Column(scale=1):
            gr.Markdown("### Feedback Signals (Guidance for Learning)")
            feedback_action_reward_component.render()
            feedback_spatial_error_component.render()
            feedback_memory_target_component.render()
            feedback_vision_label_component.render()
            feedback_motor_command_component.render()
            feedback_emotion_label_component.render()

    process_button = gr.Button("Process One Day & Consolidate Brain State", variant="primary", scale=2)

    gr.Markdown("---")
    gr.Markdown("### AI Outputs & Logs")
    with gr.Row():
        results_component.render()
    with gr.Row():
        log_component.render()

    process_button.click(
        fn=run_ai_day_interface,
        inputs=inputs_list,
        outputs=outputs_list,
        api_name="process_day"
    )

    gr.Markdown(
        """
        **How to Use:**
        1.  **Sensory Inputs:** Provide an image, text, and optionally sensor data (as a JSON formatted list). If sensor data is blank, random values will be used.
        2.  **Feedback Signals:** Adjust these values to guide the AI's learning for the current set of inputs. If text/JSON list fields are blank, random appropriate values will be used.
        3.  Click "Process One Day & Consolidate Brain State".
        4.  Observe the "Processing Results" (JSON output from the AI) and "Log Output" (detailed print statements from the AI's operations).
        5.  The AI's internal models are updated and saved after each 'day'. You can run multiple 'days' to see how it learns and adapts.
        """
    )

if __name__ == "__main__":
    demo.launch()
