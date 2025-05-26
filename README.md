# Baby AI System

## System Design

    Modules: Six Python modules (frontal.py, parietal.py, temporal.py, occipital.py, cerebellum.py, limbic.py), each containing a class for its respective AI.
        - The Occipital Lobe (CNN), Temporal Lobe (RNN/LSTM), Parietal Lobe (Dense NN), and Cerebellum (Dense NN) are built with TensorFlow/Keras. The Limbic System currently uses a simpler two-layer numpy-based neural network.
        - FrontalLobeAI uses a Deep Q-Network (DQN) built with TensorFlow/Keras for decision-making, leveraging experience replay and target networks.
    Main Script: main.py initializes the AIs, routes inputs/outputs, and manages the "daytime" (task processing) and "bedtime" (consolidation) cycles.
    Functionality:
        Each AI processes tasks relevant to its brain region.
        Learning occurs via backpropagation for most modules and Q-learning for the Frontal lobe.
        Bedtime consolidation refines models by replaying experiences and saving updates to disk.
        Purpose emerges from environmental feedback, with no predefined goals.
    Dependencies: This project uses Python with NumPy, Pillow, Gradio, and TensorFlow. Install dependencies using the provided `requirements.txt` file. For simplicity, heavy frameworks like TensorFlow are avoided for other modules, but the system can be scaled up.
    File Structure:
    project/
    ├── main.py
    ├── app.py
    ├── frontal.py
    ├── parietal.py
    ├── temporal.py
    ├── occipital.py
    ├── cerebellum.py
    ├── limbic.py
    ├── requirements.txt
    └── data/  # Stores model weights and memory

## Implementation

The system provides a modular implementation for each AI, focusing on the "baby AI" concept where capabilities are learned and refined over time.
    - The Occipital Lobe uses a TensorFlow/Keras based Convolutional Neural Network (CNN) for advanced image processing. Input images for the Occipital Lobe are automatically resized to 64x64 pixels and normalized before being processed by its CNN.
    - The Temporal Lobe now features an RNN (LSTM/GRU based) model using TensorFlow/Keras for more advanced, sequence-aware text processing. Text input is tokenized, converted to sequences, and padded/truncated to a fixed length before being processed by its RNN. 
    - The Parietal Lobe now uses a TensorFlow/Keras based Dense feed-forward neural network for sensory integration and spatial awareness tasks. 
    - The Cerebellum now uses a TensorFlow/Keras based Dense feed-forward neural network for motor control and coordination, typically outputting commands scaled between -1 and 1 (e.g., using a `tanh` activation on its output layer). The Limbic System currently uses a two-layer numpy-based neural network.
    - FrontalLobeAI employs a Deep Q-Network (DQN) using TensorFlow/Keras to improve its decision-making based on rewards and experience replay.
    - Data input has been enhanced to support image files (for Occipital) and paired image-text data for cross-modal learning experiments.

## Explanation

    Modules: Each module defines a class with:
        - A model: The OccipitalLobeAI (CNN), TemporalLobeAI (RNN/LSTM), FrontalLobeAI (DQN), ParietalLobeAI (Dense NN), and CerebellumAI (Dense NN) are implemented with TensorFlow/Keras. The Limbic System currently uses a two-layer numpy-based neural network.
        - process_task: Handles inputs specific to the AI’s role.
        - learn: Updates weights via backpropagation (for numpy-based modules), Keras model training (`fit` for OccipitalLobeAI, TemporalLobeAI, ParietalLobeAI, and CerebellumAI), or Deep Q-Network training (including experience replay and target network updates) for FrontalLobeAI based on feedback and experiences.
        - consolidate: Replays experiences from memory at "bedtime," further refining weights and saving them. For Keras models like OccipitalLobeAI, FrontalLobeAI, TemporalLobeAI, and ParietalLobeAI, this primarily involves saving the learned model weights (and for DQN/Temporal/Parietal, performing more replay steps or saving memory).
        - save_model/load_model: Persists weights (and for TemporalLobeAI, structured memories and tokenizer state) to disk. TensorFlow/Keras models use their specific weight saving/loading mechanisms.
    main.py:
        - Initializes all AIs and coordinates tasks.
        - process_day: Manages data flow (including image paths and text descriptions from paired data), routes inputs to appropriate AIs, collects outputs, and applies feedback for learning.
        - bedtime: Triggers consolidation for all AIs.
    Learning and Growth:
        - AIs start with random weights (or initial Keras model states).
        - They learn from feedback during tasks (e.g., rewards, errors, target values) and consolidate nightly.
        - OccipitalLobeAI processes images and learns to classify them using its CNN.
        - TemporalLobeAI processes text using tokenization and an RNN (LSTM/GRU based). It learns to produce text embeddings and associate them with visual labels through its TensorFlow/Keras model.
        - FrontalLobeAI learns to choose actions using a Deep Q-Network (DQN), training on past experiences stored in a replay buffer.
    Storage: Models and memories are saved in the `data/` folder. While most modules save their models as JSON files, the Occipital, Frontal, Temporal, Parietal, and Cerebellum Lobes' TensorFlow/Keras model weights are saved in specific formats (e.g., `occipital_model.weights.h5`, `frontal_model.weights.h5`, `temporal_model.weights.h5`, `parietal_model.weights.h5`, `cerebellum_model.weights.h5`). The Temporal Lobe also saves its `Tokenizer` state, the Frontal Lobe saves its exploration epsilon, and the Parietal and Cerebellum Lobes save their replay memory, typically as JSON files.

## Running the Command-Line Simulation (main.py)

1.  Create the project folder structure as listed above.
2.  Ensure Python 3.x is installed.
3.  Install necessary Python libraries:
    ```bash
    pip install -r requirements.txt
    ```
4.  Run the main script:
    ```bash
    python main.py
    ```
    The script simulates activity cycles, processing data and consolidating learning each "night." Outputs show evolving actions, motor commands, and emotions. The `main.py` script also features an interactive command-line loop where you can choose to proceed to the next day, provide new input, or quit.

## Interactive Gradio Interface

For a more user-friendly and interactive experience, a Gradio web interface is available, allowing you to interact with the AI in real-time.

### Running the Gradio App

1.  Ensure all dependencies are installed. The `requirements.txt` file includes Gradio and TensorFlow:
    ```bash
    pip install -r requirements.txt
    ```
2.  Launch the Gradio app using:
    ```bash
    python app.py
    ```
3.  Open your web browser to the local URL provided by Gradio (typically `http://127.0.0.1:7860` or `http://localhost:7860`).

### Using the Interface

The interface allows you to simulate the AI's 'days' interactively:

**Inputs**:
*   **Text Description**: Textual input for the AI to process.
*   **Upload Image**: Visual input for the AI. You can upload an image file. (Images are resized to 64x64 for the Occipital Lobe).
*   **Sensor Values (1, 2, 3)**: Three sliders representing simplified sensor data inputs to the Parietal lobe.
*   **Action Reward (Feedback)**: A slider (-1, 0, or 1) to provide a reward or punishment. This primarily influences the Frontal Lobe's Q-learning process based on the outcome of the previous state and action (though in this UI, it's provided alongside the current day's input).
*   **Vision Label (Feedback for Image)**: A numerical label you associate with the uploaded image. This is used as a target for training the Occipital Lobe (image classification) and can influence Temporal Lobe associations.

**Controls**:
*   **`Process Day` Button**: Submits the current inputs to the AI. The AI processes these inputs, its learning algorithms run, and then it undergoes 'bedtime' consolidation. The output fields are updated with the AI's responses.
*   **`Reset AI State` Button**: Re-initializes the entire AI system (all AI modules, their learned weights, and internal memories) and resets the simulation day counter to zero. This is useful for starting fresh experiments.

**Outputs**:
*   **AI Action**: The action chosen/output by the Frontal Lobe.
*   **AI Emotion**: The emotion value output by the Limbic System.
*   **AI Vision Output (Predicted Label)**: The Occipital Lobe's predicted label for the input image.
*   **Processed/Input Image Display**: Shows the image that was used for the current day's processing.
*   **Log / Messages**: Provides detailed logs of the AI's internal processing steps, the values of inputs, feedback given, and the current simulation day count.

The Gradio interface utilizes the same underlying `BrainCoordinator` and AI modules as the command-line `main.py` script but facilitates a more direct and visual way to provide data and observe the AI's behavior and learning over time.

## Example Output
(This section can remain largely the same, as the high-level output format from `main.py` hasn't changed, though specific values will differ due to new algorithms.)
text
Day 1
Action: X, Motor: [...], Emotion: Y
Starting bedtime consolidation...
Consolidation complete.
...

## Customization and Next Steps

    Real Inputs: Continue to expand with more diverse and real-world data. Consider OpenCV for more advanced vision tasks or audio libraries for direct sound processing by the Temporal Lobe.
    Scaling Models: While most modules now have a hidden layer, further scaling (more layers, more neurons) or specialized architectures (e.g., convolutional layers for Occipital, recurrent layers for Temporal) can be explored.
    Advanced RL for Frontal Lobe: The FrontalLobeAI now uses a Deep Q-Network (DQN). This can be further enhanced with techniques like Prioritized Experience Replay, Dueling DQN architectures, or by exploring more complex state representations and reward functions. The current `done` signal (always True) and `next_state` (same as current state) in `main.py`'s simulation loop also limit long-term planning; refining this would enable more sophisticated learning.
    Purpose Emergence: Refine reward functions and feedback mechanisms to guide specialization and emergent behaviors more effectively.
    Visualization: Develop tools to visualize network states, learned features, or decision processes.
    Specific Use Case: Tailor inputs, outputs, and rewards to simulate specific scenarios (e.g., simple robotics, chatbot interaction).
    Refine Modules: Continue to add features like curiosity-driven exploration, more sophisticated memory models, or attention mechanisms.
