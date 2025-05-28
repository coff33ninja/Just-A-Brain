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

# Dynamically create emotion names
EMOTION_NAMES = [f"Emotion {i}" for i in range(coordinator.limbic.output_size)]
if coordinator.limbic.output_size == 3: # Example specific names if size is 3
    EMOTION_NAMES = ["Positive", "Neutral", "Negative"] # Or Happy, Neutral, Sad etc.
elif coordinator.limbic.output_size == 0: # Handle edge case
    EMOTION_NAMES = []

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
    motor_command_str, # This is feedback emotion_label, not the predicted one
    emotion_label, # This is the *feedback* emotion label
    correction_text,
    audio_input,  # New audio input
):
    # Capture stdout for logging
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    try:
        print("[DEBUG] Starting run_ai_day_interface")
        # Prepare inputs
        current_vision_path = vision_input_path if vision_input_path and os.path.exists(vision_input_path) else DEFAULT_IMAGE_PATH
        print(f"[DEBUG] Vision path resolved: {current_vision_path}")

        if audio_input is not None:
            print(f"Audio input received: {audio_input} (audio training not yet implemented)")

        if sensor_data_str:
            print("[DEBUG] Parsing sensor_data_str")
            sensor_data = parse_json_list(sensor_data_str, None)
            if sensor_data is None:
                 print(f"Using random sensor data as input string '{sensor_data_str}' was empty or invalid.")
                 sensor_data = np.random.randn(coordinator.parietal.input_size).tolist()
        else:
            print("No sensor data provided by user, generating random sensor data.")
            sensor_data = np.random.randn(coordinator.parietal.input_size).tolist()

        # Feedback data construction
        print("[DEBUG] Constructing feedback_data")
        feedback_data = {
            "action_reward": int(action_reward),
            "spatial_error": parse_json_list(spatial_error_str, np.random.rand(coordinator.parietal.output_size).tolist()),
            "memory_target": memory_target if memory_target else text_data,
            "vision_label": int(vision_label),
            "motor_command": parse_json_list(motor_command_str, np.random.rand(coordinator.cerebellum.output_size).tolist()),
            "emotion_label": int(emotion_label), # This is the *feedback* emotion label
        }

        # Prepare new language training data
        print("[DEBUG] Preparing language_training_pair")
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

        print("[DEBUG] Calling coordinator.process_day")
        results = coordinator.process_day(
            vision_input_path=current_vision_path,
            sensor_data=sensor_data,
            text_data=text_data,
            feedback_data=feedback_data,
            language_training_pair=language_training_pair,
            correction_text=correction_text,
        )
        print("[DEBUG] coordinator.process_day finished")

        # Get the AI's textual response from the result (after process_day)
        ai_textual_response_from_day = results.get("ai_textual_response") # Already included in results
        # if ai_textual_response_from_day is not None:
        #     print(f"[Memory] The AI's current answer for '{text_data}': {ai_textual_response_from_day}")
        # else:
        #     print(f"[Memory] The AI has no specific answer for '{text_data}' after this cycle.")

        # Check AI answer against expected response
        if expected_response:
            if ai_textual_response_from_day and ai_textual_response_from_day.strip().lower() == expected_response.strip().lower():
                print("[Check] The AI's answer matches the expected response. No correction needed.")
                # In GUI, we can't prompt, so just log that reinforcement is optional
                print("You may reinforce this answer by entering it in the Correction field if desired.")
            else:
                print("[Check] The AI's answer does NOT match the expected response.")
                print(f"AI's answer: {ai_textual_response_from_day if ai_textual_response_from_day is not None else '[No answer]'}")
                print(f"Expected: {expected_response}")
                print("Reinforcing correct answer...")
                coordinator.temporal.learn([(text_data, expected_response)])

        print("--- GUI: Day Processed. Results: {} ---".format(results))
        print("[DEBUG] Calling coordinator.bedtime")
        print("--- GUI: Starting Bedtime Consolidation ---")
        coordinator.bedtime()
        print("[DEBUG] coordinator.bedtime finished")
        print("--- GUI: Bedtime Consolidation Complete ---")

        # Prepare data for emotion bar plot
        emotion_plot_data = None
        if "emotion_probabilities" in results and EMOTION_NAMES:
            probs = results["emotion_probabilities"]
            if len(probs) == len(EMOTION_NAMES):
                emotion_plot_data = {name: prob for name, prob in zip(EMOTION_NAMES, probs)}
            else:
                print(f"Warning: Mismatch between emotion probabilities ({len(probs)}) and names ({len(EMOTION_NAMES)}). Skipping plot.")

        log_output = captured_output.getvalue()
        sys.stdout = old_stdout
        # Return results, log, and emotion plot data
        return results, log_output, emotion_plot_data

    except Exception as e:
        sys.stdout = old_stdout
        error_log = captured_output.getvalue()
        error_message = f"An error occurred: {str(e)}\nTraceback:\n{traceback.format_exc()}\nLog:\n{error_log}"
        print(error_message)
        # Ensure the number of return values matches outputs_list even in error
        return {"error": str(e), "details": traceback.format_exc()}, error_message, None

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

# Add a button to list all learned Q&A pairs in the GUI
def list_learned_qa_gui():
    qa_list = []
    for idx, seq in enumerate(coordinator.temporal.memory_db):
        for q, a in seq:
            qa_list.append({"Q": q, "A": a})
    if not qa_list:
        return [{"Q": "No Q&A pairs learned yet.", "A": ""}]
    return qa_list

def qa_list_gui():
    return list_learned_qa_gui()

# --- Functions for "Generate Example" buttons ---
def get_example_sensor_data():
    example_data = np.random.rand(coordinator.parietal.input_size).tolist()
    return json.dumps([round(x, 3) for x in example_data])

def get_example_spatial_error():
    example_data = np.random.rand(coordinator.parietal.output_size).tolist()
    return json.dumps([round(x, 3) for x in example_data])

def get_example_motor_command():
    example_data = np.random.rand(coordinator.cerebellum.output_size).tolist()
    return json.dumps([round(x, 3) for x in example_data])

# Define Gradio Inputs
vision_input_path_component = gr.Image(type="filepath", label="Vision Input Image", value=DEFAULT_IMAGE_PATH,
                                       ) # info="Upload an image for the AI to 'see'."
text_data_component = gr.Textbox(label="Text Data (Sentence, Question, or Paragraph)",
                                 value="A small red block is on the table.", lines=4,
                                 info="Main textual input for the AI. Can be a statement, question, or part of a narrative.")
expected_response_component = gr.Textbox(label="Expected Response (Answer, Next Sentence, or Target Text)",
                                         value="The block is red.", lines=4,
                                         info="If Text Data is a question, this is the answer. If a statement, this could be the next logical sentence.")
sensor_data_component = gr.Textbox(
    label="Sensor Data",
    placeholder=f"e.g., JSON list of {coordinator.parietal.input_size} floats like [0.1, 0.2, ...].",
    lines=2,
    info="Simulated sensory readings. Click 'Generate Example' or leave blank for random."
)
# Button for sensor data example
generate_sensor_example_button = gr.Button("Generate Example Sensor Data", size="sm", variant="secondary")


feedback_action_reward_component = gr.Radio([-1, 0, 1], label="Feedback: Action Reward", value=0, type="value",
                                            info="Reward signal for the AI's last general action (-1: penalty, 0: neutral, 1: reward).")
feedback_spatial_error_component = gr.Textbox(
    label="Feedback: Spatial Error",
    placeholder=f"e.g., JSON list of {coordinator.parietal.output_size} floats like [0.0, 0.1, -0.05].",
    lines=1,
    info="Error signal for spatial awareness/prediction. Click 'Generate Example' or leave blank for random."
)
generate_spatial_example_button = gr.Button("Generate Example Spatial Error", size="sm", variant="secondary")

feedback_memory_target_component = gr.Textbox(label="Feedback: Memory Target Text", lines=2,
                                              placeholder="Text AI should associate with the input Text Data. If blank, uses input Text Data itself.",
                                              info="Specific text the AI should learn/memorize in relation to the main Text Data.")
feedback_vision_label_component = gr.Number(label="Feedback: True Vision Label (int)", value=0, precision=0,
                                            info=f"Correct classification label for the vision input (0 to {coordinator.occipital.output_size - 1}).")
feedback_motor_command_component = gr.Textbox(
    label="Feedback: True Motor Command",
    placeholder=f"e.g., JSON list of {coordinator.cerebellum.output_size} floats like [0.5, -0.2, 0.0].",
    lines=1,
    info="Correct motor command sequence. Click 'Generate Example' or leave blank for random."
)
generate_motor_example_button = gr.Button("Generate Example Motor Command", size="sm", variant="secondary")

feedback_emotion_label_component = gr.Number(label="Feedback: True Emotion Label (int)", value=0, precision=0,
                                             info=f"Correct emotion label for the current context (0 to {coordinator.limbic.output_size - 1}).")
correction_component = gr.Textbox(label="Correct AI Output (Reinforcement)", value="", lines=3,
                                  placeholder="If the AI's textual response was incorrect, enter the correct response here to reinforce it.",
                                  info="Provide the correct text if the AI's generated response (from memory) was wrong.")

audio_input_component = gr.Audio(type="filepath", label="Audio Input (Experimental)",
                                 # info="Upload an audio file. Currently logged but not used for training." # Pylance: No parameter named "info"
                                 )

# Add Gradio file upload and button for book training
book_file_component = gr.File(label="Upload Book (Text File)")
train_book_button = gr.Button("Train AI on Book", variant="secondary")
book_train_result_component = gr.JSON(label="Book Training Result")
book_train_log_component = gr.Textbox(label="Book Training Log", lines=10, interactive=False, autoscroll=True)
# Define Gradio Outputs
results_component = gr.JSON(label="Processing Results (Raw)")
log_component = gr.Textbox(label="Log Output", lines=20, interactive=False, autoscroll=True)
emotion_plot_component = gr.BarPlot(label="Predicted Emotion Probabilities",
                                    # info="Visualization of the AI's predicted emotional state probabilities." # Pylance: No parameter named "info"
                                    )

# Add Q&A memory display components
qa_list_component = gr.Dataframe(headers=["Q", "A"], label="Learned Q&A Pairs", interactive=False)
show_qa_button = gr.Button("Show All Learned Q&A Pairs", variant="secondary")

inputs_list = [
    vision_input_path_component,
    text_data_component,
    expected_response_component,
    sensor_data_component,
    feedback_action_reward_component,
    feedback_spatial_error_component,
    feedback_memory_target_component,
    feedback_vision_label_component,
    feedback_motor_command_component,
    feedback_emotion_label_component,
    correction_component,
    audio_input_component,  # New audio input
]
outputs_list = [results_component, log_component, emotion_plot_component] # Added emotion_plot_component

with gr.Blocks(title="Baby AI Interactive Simulation") as demo:  # Revert to default theme by removing the theme argument
    gr.Markdown("# 🧠 Baby AI Interactive Simulation")
    gr.Markdown("Interact with the AI by providing inputs for a 'day' of experience and observe its learning. All AI model weights are saved after each day's consolidation. Use the tabs below to navigate different interaction modes.")

    with gr.Tabs():
        with gr.TabItem("☀️ Daily Interaction & Learning"):
            gr.Markdown("## Provide Daily Inputs and Feedback")
            gr.Markdown("Configure the AI's experience for one 'day', provide feedback, and observe the results.")

            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 🗣️ Language & Text Input")
                    text_data_component.render()
                    expected_response_component.render()
                    correction_component.render()
                with gr.Column(scale=1):
                    gr.Markdown("### 👁️ Visual Input")
                    vision_input_path_component.render()

            with gr.Accordion("🔬 Advanced Sensory & Feedback Controls", open=False):
                with gr.Row():
                    with gr.Column(scale=2): # Give more space to textbox
                        gr.Markdown("#### Sensor Data")
                        sensor_data_component.render()
                    with gr.Column(scale=1, min_width=180): # Ensure button fits
                        gr.Markdown(" ") # Spacer for alignment
                        generate_sensor_example_button.render()

                with gr.Row():
                    with gr.Column(scale=1): # Keep audio input in its own row or group if it's distinct
                        gr.Markdown("#### Audio Input (Experimental)")
                        audio_input_component.render()
                        # gr.Markdown("*Audio input is accepted but not yet used for training.*") # Covered by info

                gr.Markdown("#### Detailed Feedback Parameters")
                with gr.Row():
                    feedback_action_reward_component.render()
                    feedback_vision_label_component.render()
                    feedback_emotion_label_component.render()
                with gr.Row():
                    with gr.Column(scale=2):
                        feedback_spatial_error_component.render()
                    with gr.Column(scale=1, min_width=180):
                        gr.Markdown(" ")
                        generate_spatial_example_button.render()
                with gr.Row():
                    with gr.Column(scale=2):
                        feedback_motor_command_component.render()
                    with gr.Column(scale=1, min_width=180):
                        gr.Markdown(" ")
                        generate_motor_example_button.render()
                feedback_memory_target_component.render()

            process_button = gr.Button("Process One Day & Consolidate Brain State", variant="primary", scale=2)

            gr.Markdown("---")
            gr.Markdown("### 📊 AI Outputs & Logs for the Day")
            with gr.Row():
                with gr.Column(scale=1):
                    results_component.render()
                with gr.Column(scale=1):
                    emotion_plot_component.render() # Added emotion plot here
            with gr.Row():
                log_component.render()

            process_button.click(
                fn=run_ai_day_interface,
                inputs=inputs_list,
                outputs=outputs_list,
                api_name="process_day"
            )
            # Wire button clicks to example generators
            generate_sensor_example_button.click(fn=get_example_sensor_data, inputs=[], outputs=[sensor_data_component])
            generate_spatial_example_button.click(fn=get_example_spatial_error, inputs=[], outputs=[feedback_spatial_error_component])
            generate_motor_example_button.click(fn=get_example_motor_command, inputs=[], outputs=[feedback_motor_command_component])

        with gr.TabItem("📚 Book & Sequential Training"):
            gr.Markdown("## Train AI on Textual Sequences")
            gr.Markdown("Upload a text file (book, dialogue log, or Q&A log) to train the AI on sequences of sentences or paragraphs. The AI learns associations between consecutive text segments.")
            with gr.Row():
                with gr.Column(scale=1):
                    book_file_component.render()
                    train_book_button.render()
                with gr.Column(scale=2):
                    book_train_result_component.render()
                    book_train_log_component.render()

            train_book_button.click(
                fn=train_on_book,
                inputs=[book_file_component],
                outputs=[book_train_result_component, book_train_log_component],
                api_name="train_on_book"
            )

        with gr.TabItem("💡 Learned Q&A"):
            gr.Markdown("## Review Learned Question & Answer Pairs")
            gr.Markdown("Review the Q&A pairs the AI has learned. You can use this information to guide further training or corrections.")
            show_qa_button.render()
            qa_list_component.render()
            show_qa_button.click(fn=qa_list_gui, inputs=[], outputs=[qa_list_component])

    gr.Markdown("---")
    with gr.Accordion("📖 How to Use & Teach the AI (From Scratch)", open=False):
        gr.Markdown(
            """
        **How to Use & Teach the AI (From Scratch):**
        - **Daily Interaction Tab:**
            - **Language & Text Input:** Provide a sentence/question in "Text Data". If it's a question or you want to teach a specific follow-up, put the desired answer/next sentence in "Expected Response". Use "Correct AI Output" to reinforce if the AI's memory lookup (shown in logs) is wrong.
            - **Visual Input:** Upload an image.
            - **Advanced Controls (Accordion):**
                - **Sensor Data:** Input a JSON list of numbers (see placeholder for size) or click "Generate Example". Leave blank for random.
                - **Audio Input:** (Experimental) Upload audio; it's logged but not yet used for training.
                - **Detailed Feedback:**
                    - *Action Reward:* Give +1 for good overall AI behavior, -1 for bad, 0 for neutral.
                    - *Vision Label:* The correct category number for the image.
                    - *Emotion Label:* The correct emotion category number for the context.
                    - *Spatial Error:* JSON list (see placeholder) representing error in spatial tasks, or "Generate Example".
                    - *Motor Command:* Correct JSON list for motor output, or "Generate Example".
                    - *Memory Target:* Specific text to associate with "Text Data". Defaults to "Text Data" itself for self-association.
        - **Book & Sequential Training Tab:** Upload a plain text file. The AI learns by reading sentences/paragraphs sequentially, associating each with the one that follows. Good for learning narrative flow or dialogue patterns.
        - **Learned Q&A Tab:** See what direct question/answer pairs the AI has stored in its temporal lobe memory.
        - **General Tips:**
            - Start simple, then increase complexity (curriculum learning).
            - Mix input types (narrative, Q&A, visual).
            - The AI learns from patterns. Clear, consistent data and feedback are key. It doesn't "understand" like humans but learns associations.
            - All model weights are saved after each "Process One Day" click.
        """
    )
    gr.Markdown("---")

if __name__ == "__main__":
    demo.launch()
