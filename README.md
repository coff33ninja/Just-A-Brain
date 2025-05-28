# Baby AI System

## Overview

Baby AI System is a modular, brain-inspired artificial intelligence framework. It features six specialized neural modules (Occipital, Temporal, Parietal, Cerebellum, Frontal, Limbic), each handling a distinct cognitive function, coordinated by a central script. The system is designed for extensibility, experimentation, and educational purposes.

For a detailed technical and architectural explanation, see the [Whitepaper](./Whitepaper.md).

## System Design

- **Modules:** Six Python modules (`frontal.py`, `parietal.py`, `temporal.py`, `occipital.py`, `cerebellum.py`, `limbic.py`), each containing a class for its respective AI. All six core AI modules (Occipital, Temporal, Parietal, Cerebellum, Frontal, and Limbic) are now implemented using TensorFlow/Keras.
- **Main Script:** `main.py` initializes the AIs, routes inputs/outputs, and manages the "daytime" (task processing) and "bedtime" (consolidation) cycles.
- **Functionality:**
    - Each AI processes tasks relevant to its brain region.
    - Learning occurs via backpropagation for most modules. The Frontal Lobe employs Q-learning, which now operates over defined multi-step episodes, allowing it to learn from sequences of states and actions for better long-term decision-making.
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
1. **Clone the repository (if you haven't already):**
   ```pwsh
   git clone <your-repository-url>  # Replace <your-repository-url> with the actual URL
   cd Just-A-Brain
   ```
   *(If you've downloaded the code as a ZIP, extract it and navigate to the project directory.)*

2. **Create and activate a virtual environment (recommended):**
   ```pwsh
   python -m venv .venv
   # On Windows PowerShell:
   .\.venv\Scripts\Activate.ps1
   # On Linux/macOS:
   # source .venv/bin/activate
   ```
3. **Ensure Python 3.8+ is installed.**
4. **Install necessary Python libraries:**
   ```pwsh
   pip install -r requirements.txt
   ```
5. Run the main script:
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

**AI Data Management Tab (if present):** This tab provides more granular control over resetting AI data.
- **Warning:** Actions in this tab will delete saved model weights and clear learned memories.
- You can typically reset data for individual AI lobes (Frontal, Temporal, etc.) or reset all AI data at once.

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

## How This AI Learns (From Scratch)

This AI does **not** use any pre-trained models or transformer architectures. All neural modules (vision, language, decision, etc.) are built from scratch and learn only from the data you provide during training. The system forms associations between inputs (images, text, sensor data) and feedback, gradually improving its predictions and responses through repeated exposure and feedback.

### Teaching the AI
- **Text/Language:** Use the 'Text Data' and 'Expected Response' fields to teach associations (e.g., Q&A, next-sentence, or dialogue pairs).
- **Images:** Upload images and provide labels to train visual recognition.
- **Books:** Use the 'Book Training' section to upload a text file (book). The AI will learn associations between consecutive sentences or paragraphs.
- **Correction:** If the AI's output is wrong, use the 'Correct AI Output' field to provide the right answer and reinforce learning.
- **Feedback:** Use the reward and label fields to guide the AI's learning for each modality.

### Limitations
- The AI's language and reasoning abilities are limited to the patterns and associations it has seen during training. It does not have deep understanding or generative capabilities like transformer-based models.
- For best results, provide clear, consistent, and well-structured training data.

## Customization and Next Steps

- **Real Inputs:** Expand with more diverse and real-world data. Consider OpenCV for advanced vision tasks or audio libraries for direct sound processing by the Temporal Lobe.
- **Scaling Models:** Further scaling (more layers, more neurons) or specialized architectures (e.g., convolutional layers for Occipital, recurrent layers for Temporal) can be explored.
- **Advanced RL for Frontal Lobe:** The Frontal Lobe's DQN now learns with a proper episodic structure, using meaningful `done` signals and `next_state` transitions, which provides a better foundation for long-term planning. Future enhancements can build on this by exploring techniques like Prioritized Experience Replay, Dueling DQN architectures, or developing more complex state representations and reward functions tied to specific tasks.
- **Purpose Emergence:** Refine reward functions and feedback mechanisms to guide specialization and emergent behaviors more effectively.
- **Visualization:** Develop tools to visualize network states, learned features, or decision processes.
- **Specific Use Case:** Tailor inputs, outputs, and rewards to simulate specific scenarios (e.g., simple robotics, chatbot interaction).
- **Refine Modules:** Add features like curiosity-driven exploration, more sophisticated memory models, or attention mechanisms.

---

For a comprehensive description of the architecture, learning mechanisms, and module details, please refer to the [Whitepaper](./Whitepaper.md).

## Using main.py: Interactive CLI & Book Training

The command-line interface (`main.py`) now supports richer training modalities, similar to the GUI:

### Starting the Simulation

```pwsh
python main.py
```

You will be prompted for each 'day' to:
- Enter a new image path (or press Enter to use the last/default)
- Enter new text data (sentence, question, or paragraph)
- Optionally enter an **expected response** (answer, next sentence, or target text) for Q&A or dialogue training
- Enter a visual label (integer)
- Optionally generate new random sensor data
- Optionally provide a path to an audio file (not yet used for training, but accepted for future support)

After the AI processes the day's data, you can:
- Enter a **correction** if the AI's output was wrong (this reinforces the correct answer for the given input)

You can also run for a fixed number of days or indefinitely, and choose to use scheduled data or provide new input each day.

### Book/Story/Sequential Training

You can train the AI on a text file (book, Q&A log, or story) using:

```pwsh
python main.py --book path/to/book.txt
```

The AI will learn associations between consecutive sentences or paragraphs in the file. This is useful for teaching dialogue, stories, or structured Q&A.

### Help

```pwsh
python main.py --help
```

### Notes
- Audio input is accepted as a file path but is not yet used for training (future feature).
- Correction and Q&A/expected response training are available in both CLI and GUI.
- For best results, provide clear, well-structured, and consistent data, and use correction/feedback to reinforce learning.

See the rest of this README and the Whitepaper for more details on architecture and training modalities.

## Q&A Memory, Explainability, and Feedback (New Features)

### CLI Features
- At any prompt, type `list` to see all learned Q&A pairs (facts) the AI has stored in its memory.
- When you enter a text input, the AI will show if it already knows an answer for it (from its memory).
- After the AI answers, it will compare its answer to your expected response and tell you if it was correct or not.
- The system will only reinforce (learn) if the answer was wrong, or if you explicitly choose to reinforce even when correct.
- You can still provide corrections as before.

### GUI Features
- Click the **Show All Learned Q&A Pairs** button to see a table of all Q&A pairs the AI has learned so far.
- When you submit a text input, the log will show if the AI already knows an answer for it.
- If you provide an expected response, the system will compare the AI's answer to it and log whether it was correct. If not, it will automatically reinforce the correct answer.
- You can use the **Correct AI Output** field to reinforce the correct answer if the AI was wrong, or to further strengthen correct associations.

### Why This Matters
- These features make the AI's learning process transparent and interactive.
- You can see exactly what the AI has learned, correct mistakes, and avoid over-reinforcing correct answers.
- This approach supports explainable AI and helps you guide the system's knowledge more effectively.
