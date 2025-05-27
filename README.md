# Baby AI System

## Overview

Baby AI System is a modular, brain-inspired artificial intelligence framework. It features six specialized neural modules (Occipital, Temporal, Parietal, Cerebellum, Frontal, Limbic), each handling a distinct cognitive function, coordinated by a central script. The system is designed for extensibility, experimentation, and educational purposes.

For a detailed technical and architectural explanation, see the [Whitepaper](./Whitepaper.md).

## System Design

- **Modules:** Six Python modules (`frontal.py`, `parietal.py`, `temporal.py`, `occipital.py`, `cerebellum.py`, `limbic.py`), each containing a class for its respective AI. All modules are implemented with TensorFlow/Keras, replacing their original NumPy-based implementations.
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

## Quick Start

### Command-Line Simulation

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

### Interactive Gradio Interface

A Gradio web interface is available for real-time interaction with the AI.

1. Ensure all dependencies are installed:
   ```pwsh
   pip install -r requirements.txt
   ```
2. Launch the Gradio app:
   ```pwsh
   python gui.py
   ```
3. Open your web browser to the local URL provided by Gradio (typically `http://127.0.0.1:7860` or `http://localhost:7860`).

#### Using the Interface

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

## How Training Works: The Baby Analogy and Current Limitations

Like a human baby, this AI system learns through repeated exposure to sensory data and feedback. Each module specializes in a different aspect of cognition:
- **Vision (Occipital):** Learns to recognize objects from images.
- **Language/Association (Temporal):** Learns to associate simple words or short text with images or labels.
- **Sensation/Spatial (Parietal):** Learns to interpret sensor data and spatial relationships.
- **Action/Decision (Frontal):** Learns to make decisions based on experience and feedback.
- **Motor (Cerebellum):** Learns to produce coordinated outputs.
- **Emotion (Limbic):** Learns to associate feelings with experiences.

**Current Training Modalities:**
- The system can be trained on images (object/vision recognition) and simple text (labels or short descriptions for images).
- There is no true language learning (e.g., grammar, conversation, or complex text understanding) yet—only basic associations between words and images.
- No audio, video, or real-world sensor data is currently supported.
- Feedback is limited to simple rewards or labels provided by the user.

**How to Train:**
- Each "day" (cycle), the AI processes new data (image, text, sensor values), receives feedback (reward/label), and updates its memory.
- At "bedtime" (consolidation), it replays experiences to strengthen learning, similar to how sleep helps babies consolidate memories.
- You can further train the AI by providing more data, giving feedback, or triggering extra consolidation cycles (like extra practice or sleep).
- In the GUI, use the "Process Day" button for a single cycle, and (if available) a "Further Train" button to trigger additional consolidation.
- In the CLI, you can run more cycles or add new data for continued learning.

**Limitations and Future Directions:**
- The system cannot yet learn language in a human sense (no conversation, grammar, or context).
- It cannot learn from audio, video, or real-world interaction.
- Future work could add richer data types, curriculum learning, or more interactive training methods.

For more details on the architecture and future plans, see the [Whitepaper](./Whitepaper.md).

## Customization and Next Steps

- **Real Inputs:** Expand with more diverse and real-world data. Consider OpenCV for advanced vision tasks or audio libraries for direct sound processing by the Temporal Lobe.
- **Scaling Models:** Further scaling (more layers, more neurons) or specialized architectures (e.g., convolutional layers for Occipital, recurrent layers for Temporal) can be explored.
- **Advanced RL for Frontal Lobe:** Enhance the DQN with techniques like Prioritized Experience Replay, Dueling DQN architectures, or more complex state representations and reward functions. The current `done` signal (always True) and `next_state` (same as current state) in `main.py`'s simulation loop limit long-term planning; refining this would enable more sophisticated learning.
- **Purpose Emergence:** Refine reward functions and feedback mechanisms to guide specialization and emergent behaviors more effectively.
- **Visualization:** Develop tools to visualize network states, learned features, or decision processes.
- **Specific Use Case:** Tailor inputs, outputs, and rewards to simulate specific scenarios (e.g., simple robotics, chatbot interaction).
- **Refine Modules:** Add features like curiosity-driven exploration, more sophisticated memory models, or attention mechanisms.

---

For a comprehensive description of the architecture, learning mechanisms, and module details, please refer to the [Whitepaper](./Whitepaper.md).
