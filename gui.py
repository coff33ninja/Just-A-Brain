import gradio as gr
import os
import sys
import json
import numpy as np
from io import StringIO
import traceback
from main import BrainCoordinator, DEFAULT_IMAGE_PATH

# Add project root to sys.path to allow imports from other project files
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
    expected_response,
    sensor_data_str,
    action_reward,
    spatial_error_str,
    memory_target,
    vision_label,
    motor_command_str,
    emotion_label,
    correction_text,
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

        # Prepare new language training data
        language_training_pair = None
        if text_data and expected_response:
            language_training_pair = [(text_data, expected_response)]
        elif text_data:
            language_training_pair = [(text_data, text_data)]  # fallback: self-association

        print("--- GUI: Processing Day ---")
        print(f"Vision Path: {os.path.basename(current_vision_path) if current_vision_path else 'N/A'}")
        print(f"Text Data: '{text_data}'")
        print(f"Sensor Data (first 5): {sensor_data[:5] if sensor_data else 'N/A'}...")
        print(f"Feedback Data: {feedback_data}")

        results = coordinator.process_day(
            vision_input_path=current_vision_path,
            sensor_data=sensor_data,
            text_data=text_data,
            feedback_data=feedback_data,
            language_training_pair=language_training_pair,
            correction_text=correction_text,
        )

        print("--- GUI: Day Processed. Results: {} ---".format(results))
        print("--- GUI: Starting Bedtime Consolidation ---")
        coordinator.bedtime()
        print("--- GUI: Bedtime Consolidation Complete ---")

        log_output = captured_output.getvalue()
        sys.stdout = old_stdout
        return results, log_output

    except Exception as e:
        sys.stdout = old_stdout
        error_log = captured_output.getvalue()
        error_message = f"An error occurred: {str(e)}\nTraceback:\n{traceback.format_exc()}\nLog:\n{error_log}"
        print(error_message)
        return {"error": str(e), "details": traceback.format_exc()}, error_message

# Add a function to train the AI on a book (text file)
def train_on_book(book_file):
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    try:
        if book_file is None or not os.path.exists(book_file):
            print("No book file provided or file does not exist.")
            sys.stdout = old_stdout
            return {"result": "No file provided."}, captured_output.getvalue()
        with open(book_file, 'r', encoding='utf-8') as f:
            text = f.read()
        # Split into sentences or paragraphs (simple split by lines or periods)
        # Here, we use paragraphs (double newlines) if possible, else fallback to sentences
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if len(paragraphs) < 2:
            # fallback to sentences
            import re
            paragraphs = re.split(r'(?<=[.!?]) +', text)
        # Train on each pair of (current, next) as (input, target)
        pairs = []
        for i in range(len(paragraphs) - 1):
            pairs.append((paragraphs[i], paragraphs[i+1]))
        if not pairs:
            print("Book is too short for training.")
            sys.stdout = old_stdout
            return {"result": "Book is too short for training."}, captured_output.getvalue()
        print(f"Training on {len(pairs)} text pairs from book...")
        for pair in pairs:
            coordinator.temporal.learn([pair])
        print("Book training complete.")
        sys.stdout = old_stdout
        return {"result": f"Trained on {len(pairs)} text pairs from book."}, captured_output.getvalue()
    except Exception as e:
        sys.stdout = old_stdout
        import traceback
        error_log = captured_output.getvalue()
        error_message = f"An error occurred: {str(e)}\nTraceback:\n{traceback.format_exc()}\nLog:\n{error_log}"
        print(error_message)
        return {"error": str(e), "details": traceback.format_exc()}, error_message

# Define Gradio Inputs
vision_input_path_component = gr.Image(type="filepath", label="Vision Input Image", value=DEFAULT_IMAGE_PATH)
text_data_component = gr.Textbox(label="Text Data (Sentence/Question)", value="A small red block is on the table.", lines=2)
expected_response_component = gr.Textbox(label="Expected Response (Answer/Target Text)", value="The block is red.", lines=2)
sensor_data_component = gr.Textbox(
    label="Sensor Data (JSON list)",
    placeholder=f"e.g., a list of {coordinator.parietal.input_size} floats like [0.1, 0.2, ...]. If blank, random data is used.",
    lines=2
)

feedback_action_reward_component = gr.Radio([-1, 0, 1], label="Feedback: Action Reward", value=0, type="value")
feedback_spatial_error_component = gr.Textbox(
    label="Feedback: Spatial Error (JSON list)",
    placeholder=f"e.g., list of {coordinator.parietal.output_size} floats like [0.0, 0.1, -0.05]. If blank, random.",
    lines=1
)
feedback_memory_target_component = gr.Textbox(label="Feedback: Memory Target Text", lines=2, placeholder="Text AI should learn for input. If blank, uses input Text Data.")
feedback_vision_label_component = gr.Number(label="Feedback: True Vision Label (int)", value=0, precision=0)
feedback_motor_command_component = gr.Textbox(
    label="Feedback: True Motor Command (JSON list)",
    placeholder=f"e.g., list of {coordinator.cerebellum.output_size} floats like [0.5, -0.2, 0.0]. If blank, random.",
    lines=1
)
feedback_emotion_label_component = gr.Number(label="Feedback: True Emotion Label (int)", value=0, precision=0)
correction_component = gr.Textbox(label="Correct AI Output (if AI was wrong)", value="", lines=2, placeholder="If the AI output was incorrect, enter the correct response here.")

# Add Gradio file upload and button for book training
book_file_component = gr.File(label="Upload Book (Text File)")
train_book_button = gr.Button("Train AI on Book", variant="secondary")
book_train_result_component = gr.JSON(label="Book Training Result")
book_train_log_component = gr.Textbox(label="Book Training Log", lines=10, interactive=False, autoscroll=True)

# Define Gradio Outputs
results_component = gr.JSON(label="Processing Results")
log_component = gr.Textbox(label="Log Output", lines=20, interactive=False, autoscroll=True)

inputs_list = [
    vision_input_path_component,
    text_data_component,
    expected_response_component,  # New field
    sensor_data_component,
    feedback_action_reward_component,
    feedback_spatial_error_component,
    feedback_memory_target_component,
    feedback_vision_label_component,
    feedback_motor_command_component,
    feedback_emotion_label_component,
    correction_component,  # New field
]
outputs_list = [results_component, log_component]

with gr.Blocks(title="Baby AI Interactive Simulation") as demo:
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
        **How to Use & Teach the AI (From Scratch):**
        - Use the **Text Data** and **Expected Response** fields to teach the AI associations (Q&A, next-sentence, or dialogue pairs). For advanced training, upload structured Q&A or dialogue logs as books.
        - Use the **Book Training** section to upload a text file (book). The AI will learn associations between consecutive sentences or paragraphs. Books with dialogue, stories, or structured conversations work best for richer context.
        - If the AI's output is wrong, use the **Correct AI Output** field to reinforce the correct answer and help the AI learn interactively.
        - Upload images and provide labels for visual learning. You can also pair images with descriptive text for cross-modal association.
        - Use the reward and label fields to guide learning in all modalities. The system supports reinforcement-like feedback for actions and emotions.
        - For best results, start with simple examples and gradually increase complexity (curriculum learning). Mix narrative, dialogue, and factual text for more nuanced associations.
        - The AI's understanding is limited to the associations and patterns it has seen; it does not generalize beyond its training data. Provide clear, well-structured, and consistent data, and give feedback or corrections when possible.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Book Training")
            book_file_component.render()
            train_book_button.render()

    with gr.Row():
        book_train_result_component.render()
    with gr.Row():
        book_train_log_component.render()

    train_book_button.click(
        fn=train_on_book,
        inputs=[book_file_component],
        outputs=[book_train_result_component, book_train_log_component],
        api_name="train_on_book"
    )

if __name__ == "__main__":
    demo.launch()
