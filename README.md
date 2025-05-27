# Baby AI System

## System Design

- **Modules:** Six Python modules (`frontal.py`, `parietal.py`, `temporal.py`, `occipital.py`, `cerebellum.py`, `limbic.py`), each containing a class for its respective AI.
    - All AI modules—Occipital Lobe (CNN), Temporal Lobe (RNN/LSTM), Parietal Lobe (Dense NN), Cerebellum (Dense NN), Frontal Lobe (DQN), and Limbic System (Dense NN)—are built with TensorFlow/Keras, replacing their original NumPy-based implementations.
- **Main Script:** `main.py` initializes the AIs, routes inputs/outputs, and manages the "daytime" (task processing) and "bedtime" (consolidation) cycles.
- **Functionality:**
    - Each AI processes tasks relevant to its brain region.
    - Learning occurs via backpropagation for most modules and Q-learning for the Frontal lobe.
    - Bedtime consolidation refines models by replaying experiences and saving updates to disk.
    - Purpose emerges from environmental feedback, with no predefined goals.
- **Dependencies:** Python 3.8+ with NumPy, Pillow, Gradio, and TensorFlow. Install dependencies using the provided `requirements.txt` file.

### File Structure

```
project/
├── main.py
├── gui.py
├── frontal.py
├── parietal.py
├── temporal.py
├── occipital.py
├── cerebellum.py
├── limbic.py
├── requirements.txt
└── data/  # Stores model weights and memory
```

## Implementation

The system provides a modular implementation for each AI, focusing on the "baby AI" concept where capabilities are learned and refined over time.

- **Occipital Lobe:** TensorFlow/Keras-based Convolutional Neural Network (CNN) for image processing. Input images are resized to 64x64 pixels and normalized before processing.
- **Temporal Lobe:** RNN (LSTM/GRU) model using TensorFlow/Keras for sequence-aware text processing. Text is tokenized, converted to sequences, and padded/truncated to a fixed length.
- **Parietal Lobe:** TensorFlow/Keras-based Dense feed-forward neural network for sensory integration and spatial awareness.
- **Cerebellum:** TensorFlow/Keras-based Dense feed-forward neural network for motor control and coordination.
- **Limbic System:** TensorFlow/Keras Dense feed-forward neural network for emotion classification, typically taking text embeddings as input and outputting an emotion category using `softmax` activation.
- **Frontal Lobe:** Deep Q-Network (DQN) using TensorFlow/Keras for decision-making based on rewards and experience replay.
- **Data Input:** Supports image files (for Occipital) and paired image-text data for cross-modal learning experiments.

## Explanation

Each module defines a class with:
- **A model:** All AI modules—OccipitalLobeAI (CNN), TemporalLobeAI (RNN/LSTM), FrontalLobeAI (DQN), ParietalLobeAI (Dense NN), CerebellumAI (Dense NN), and LimbicSystemAI (Dense NN)—are implemented with TensorFlow/Keras, each tailored to its specific processing task (e.g., image classification, text processing, decision making, sensor integration, motor control, emotion classification).
- **process_task:** Handles inputs specific to the AI’s role.
- **learn:** Updates weights via Keras model training (`fit` for OccipitalLobeAI, TemporalLobeAI, ParietalLobeAI, CerebellumAI, and LimbicSystemAI), or Deep Q-Network training (including experience replay and target network updates) for FrontalLobeAI. The Limbic System's Keras training incorporates reward signals as `sample_weight` during learning and consolidation.
- **consolidate:** Replays experiences from memory at "bedtime," further refining weights and saving them. For Keras models, this involves saving learned model weights and, for DQN/Temporal/Parietal/Limbic, performing more replay steps or saving memory.
- **save_model/load_model:** Persists weights (and for TemporalLobeAI, structured memories and tokenizer state) to disk. TensorFlow/Keras models use their specific weight saving/loading mechanisms.

**main.py:**
- Initializes all AIs and coordinates tasks.
- **process_day:** Manages data flow (including image paths and text descriptions from paired data), routes inputs to appropriate AIs, collects outputs, and applies feedback for learning.
- **bedtime:** Triggers consolidation for all AIs.

**Learning and Growth:**
- AIs start with random weights (or initial Keras model states).
- They learn from feedback during tasks (e.g., rewards, errors, target values) and consolidate nightly.
- OccipitalLobeAI processes images and learns to classify them using its CNN.
- TemporalLobeAI processes text using tokenization and an RNN (LSTM/GRU). It learns to produce text embeddings and associate them with visual labels.
- FrontalLobeAI learns to choose actions using a Deep Q-Network (DQN), training on past experiences stored in a replay buffer.

**Storage:** Models and memories are saved in the `data/` folder. Model weights are saved in specific formats (e.g., `occipital_model.weights.h5`, ..., `limbic_model.weights.h5`). The Temporal Lobe also saves its `Tokenizer` state, the Frontal Lobe saves its exploration epsilon, and the Parietal, Cerebellum, and Limbic Lobes save their replay memory (including rewards for the Limbic System), typically as JSON files.

## Running the Command-Line Simulation (`main.py`)

1. Create the project folder structure as listed above.
2. Ensure Python 3.8+ is installed.
3. Install necessary Python libraries:
   ```pwsh
   pip install -r requirements.txt
   ```
4. Run the main script:
   ```pwsh
   python main.py
   ```
   The script simulates activity cycles, processing data and consolidating learning each "night." Outputs show evolving actions, motor commands, and emotions. The `main.py` script also features an interactive command-line loop where you can proceed to the next day, provide new input, or quit.

## Interactive Gradio Interface

A Gradio web interface is available for real-time interaction with the AI.

### Running the Gradio App

1. Ensure all dependencies are installed:
   ```pwsh
   pip install -r requirements.txt
   ```
2. Launch the Gradio app:
   ```pwsh
   python gui.py
   ```
3. Open your web browser to the local URL provided by Gradio (typically `http://127.0.0.1:7860` or `http://localhost:7860`).

### Using the Interface

The interface allows you to simulate the AI's 'days' interactively:

**Inputs:**
- **Text Description:** Textual input for the AI to process.
- **Upload Image:** Visual input for the AI. (Images are resized to 64x64 for the Occipital Lobe).
- **Sensor Values (1, 2, 3):** Three sliders representing simplified sensor data inputs to the Parietal lobe.
- **Action Reward (Feedback):** A slider (-1, 0, or 1) to provide a reward or punishment. This primarily influences the Frontal Lobe's Q-learning process.
- **Vision Label (Feedback for Image):** A numerical label for the uploaded image, used as a target for training the Occipital Lobe and influencing Temporal Lobe associations.

**Controls:**
- **Process Day Button:** Submits the current inputs to the AI. The AI processes these inputs, runs its learning algorithms, and undergoes 'bedtime' consolidation. Output fields are updated with the AI's responses.
- **Reset AI State Button:** Re-initializes the entire AI system (all modules, weights, and memories) and resets the simulation day counter to zero.

**Outputs:**
- **AI Action:** The action chosen/output by the Frontal Lobe.
- **AI Emotion:** The emotion value output by the Limbic System.
- **AI Vision Output (Predicted Label):** The Occipital Lobe's predicted label for the input image.
- **Processed/Input Image Display:** Shows the image used for the current day's processing.
- **Log / Messages:** Detailed logs of the AI's internal processing steps, input values, feedback, and the current simulation day count.

The Gradio interface uses the same underlying `BrainCoordinator` and AI modules as the command-line `main.py` script, providing a direct and visual way to provide data and observe the AI's behavior and learning over time.

## Example Output

```
Day 1
Action: X, Motor: [...], Emotion: Y
Starting bedtime consolidation...
Consolidation complete.
...
```

## Customization and Next Steps

- **Real Inputs:** Expand with more diverse and real-world data. Consider OpenCV for advanced vision tasks or audio libraries for direct sound processing by the Temporal Lobe.
- **Scaling Models:** Further scaling (more layers, more neurons) or specialized architectures (e.g., convolutional layers for Occipital, recurrent layers for Temporal) can be explored.
- **Advanced RL for Frontal Lobe:** Enhance the DQN with techniques like Prioritized Experience Replay, Dueling DQN architectures, or more complex state representations and reward functions. The current `done` signal (always True) and `next_state` (same as current state) in `main.py`'s simulation loop limit long-term planning; refining this would enable more sophisticated learning.
- **Purpose Emergence:** Refine reward functions and feedback mechanisms to guide specialization and emergent behaviors more effectively.
- **Visualization:** Develop tools to visualize network states, learned features, or decision processes.
- **Specific Use Case:** Tailor inputs, outputs, and rewards to simulate specific scenarios (e.g., simple robotics, chatbot interaction).
- **Refine Modules:** Add features like curiosity-driven exploration, more sophisticated memory models, or attention mechanisms.
