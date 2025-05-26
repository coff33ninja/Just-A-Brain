import gradio as gr
from main import BrainCoordinator, DEFAULT_IMAGE_PATH # Ensure DEFAULT_IMAGE_PATH is imported
import os
import numpy as np

# Initialize a global BrainCoordinator instance
print("Initializing BrainCoordinator for Gradio app...")
coordinator = BrainCoordinator()
print("BrainCoordinator initialized.")

# Initialize Global Day Counter
day_index = 0

# Main AI processing function (remains largely the same)
def process_ai_day_interface(
    text_input, 
    image_input_path,
    sensor_input_1, 
    sensor_input_2, 
    sensor_input_3, 
    action_reward_input, 
    vision_label_input
):
    global day_index 
    log_messages = []
    current_day_for_log = day_index + 1
    log_messages.append(f"--- Day {current_day_for_log} ---")

    try:
        log_messages.append(f"[Day {current_day_for_log}] Preparing inputs...")
        current_vision_path = DEFAULT_IMAGE_PATH
        if image_input_path: 
            if os.path.exists(image_input_path):
                current_vision_path = image_input_path
                log_messages.append(f"[Day {current_day_for_log}] Using uploaded image: {os.path.basename(image_input_path)}")
            else:
                log_messages.append(f"[Day {current_day_for_log}] Warning: Uploaded image path '{image_input_path}' not found. Using default image.")
        else:
            log_messages.append(f"[Day {current_day_for_log}] No image uploaded. Using default image.")

        sensor_data = np.array([sensor_input_1, sensor_input_2, sensor_input_3])
        log_messages.append(f"[Day {current_day_for_log}] Sensor data: {sensor_data.tolist()}")

        text_data = text_input if text_input and text_input.strip() else "Default sample text"
        log_messages.append(f"[Day {current_day_for_log}] Text data: '{text_data}'")

        try:
            user_provided_vision_label = int(vision_label_input)
        except ValueError:
            log_messages.append(f"[Day {current_day_for_log}] Warning: Invalid vision label feedback '{vision_label_input}'. Defaulting to 0.")
            user_provided_vision_label = 0
        
        feedback_data = {
            "action_reward": action_reward_input,
            "spatial_error": np.random.rand(3).tolist(), 
            "memory_target": text_data, 
            "vision_label": user_provided_vision_label, 
            "motor_command": np.random.rand(3).tolist(), 
            "emotion_label": np.random.randint(0, getattr(coordinator.limbic, "output_size", 3))
        }
        log_messages.append(f"[Day {current_day_for_log}] Feedback data prepared. Action reward: {action_reward_input}, Vision label feedback: {user_provided_vision_label}.")

        log_messages.append(f"[Day {current_day_for_log}] Processing day...")
        results = coordinator.process_day(
            vision_input_path=current_vision_path,
            sensor_data=sensor_data,
            text_data=text_data,
            feedback_data=feedback_data
        )
        log_messages.append(f"[Day {current_day_for_log}] Day processed successfully.")

        log_messages.append(f"[Day {current_day_for_log}] Consolidating learning (bedtime)...")
        coordinator.bedtime()
        log_messages.append(f"[Day {current_day_for_log}] Consolidation complete.")

        day_index += 1

        ai_action = results.get('action', "N/A")
        ai_emotion = results.get('emotion', "N/A")
        ai_vision_prediction = results.get('vision_label', "N/A") 

        action_str = str(ai_action)
        emotion_str = str(ai_emotion)
        vision_pred_str = str(ai_vision_prediction)
        
        image_to_display_path = current_vision_path
        
        log_messages.append(f"[Day {current_day_for_log}] AI Action: {action_str}, Emotion: {emotion_str}, Vision Prediction: {vision_pred_str}")
        log_messages.append(f"--- End of Day {current_day_for_log} ---")
        
        final_log = "\n".join(log_messages)
        return action_str, emotion_str, vision_pred_str, image_to_display_path, final_log

    except Exception as e:
        error_message = f"[Day {current_day_for_log}] Error during AI processing: {str(e)}"
        log_messages.append(error_message)
        print(error_message) 
        final_log = "\n".join(log_messages)
        return "Error", "Error", "Error", DEFAULT_IMAGE_PATH, final_log

# Function to reset AI state
def reset_ai_state_interface():
    global coordinator, day_index
    print("Resetting AI Coordinator and day_index...") # Server-side log
    coordinator = BrainCoordinator() # Re-initialize
    day_index = 0
    print("AI state reset complete.")
    log_message = "AI state has been reset. Day count is now 0. All lobes and learning progress are cleared."
    # Return values to clear/update all 5 output fields
    return "N/A", "N/A", "N/A", None, log_message


# Define Gradio Interface using gr.Blocks
with gr.Blocks(title="Baby AI Interactive Day Processor", theme=gr.themes.Soft()) as iface:
    gr.Markdown("# Baby AI Interactive Day Processor")
    gr.Markdown("Provide inputs for the AI to process for a 'day'. The AI will learn based on the feedback provided. You can also reset the AI's learned state.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Inputs")
            text_input_comp = gr.Textbox(label="Text Description", info="Enter text for the AI to process.")
            image_input_comp = gr.Image(type="filepath", label="Upload Image", sources=["upload"])
            
            gr.Markdown("#### Sensor Values")
            sensor_input_1_comp = gr.Slider(minimum=-1, maximum=1, step=0.1, label="Sensor Value 1", value=0)
            sensor_input_2_comp = gr.Slider(minimum=-1, maximum=1, step=0.1, label="Sensor Value 2", value=0)
            sensor_input_3_comp = gr.Slider(minimum=-1, maximum=1, step=0.1, label="Sensor Value 3", value=0)
            
            gr.Markdown("#### Feedback for Learning")
            action_reward_input_comp = gr.Slider(minimum=-1, maximum=1, step=1, label="Action Reward (Feedback)", value=0, info="-1 for punishment, 0 for neutral, 1 for reward")
            vision_label_input_comp = gr.Number(label="Vision Label (Feedback for Image)", value=0, precision=0, info="Integer label you associate with the uploaded image.")
            
            with gr.Row():
                process_button = gr.Button("Process Day", variant="primary")
                reset_button = gr.Button("Reset AI State")

        with gr.Column(scale=1):
            gr.Markdown("### AI Outputs")
            action_output_comp = gr.Textbox(label="AI Action") 
            emotion_output_comp = gr.Textbox(label="AI Emotion") 
            vision_pred_output_comp = gr.Textbox(label="AI Vision Output (Predicted Label)") 
            image_display_comp = gr.Image(label="Processed/Input Image Display", type="filepath") 
            log_output_comp = gr.Textbox(label="Log / Messages", lines=10, interactive=False, autoscroll=True)

    # Define lists of input and output components
    inputs_list = [
        text_input_comp,
        image_input_comp,
        sensor_input_1_comp,
        sensor_input_2_comp,
        sensor_input_3_comp,
        action_reward_input_comp,
        vision_label_input_comp
    ]
    
    outputs_list = [
        action_output_comp,
        emotion_output_comp,
        vision_pred_output_comp,
        image_display_comp,
        log_output_comp
    ]

    # Connect buttons to functions
    process_button.click(
        fn=process_ai_day_interface,
        inputs=inputs_list,
        outputs=outputs_list
    )
    
    reset_button.click(
        fn=reset_ai_state_interface,
        inputs=[], # No direct inputs from UI fields for reset
        outputs=outputs_list
    )

if __name__ == "__main__":
    print("Launching Gradio interface with Blocks...")
    iface.launch()
