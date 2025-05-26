# Baby AI System

## System Design

    Modules: Six Python modules (frontal.py, parietal.py, temporal.py, occipital.py, cerebellum.py, limbic.py), each containing a class for its respective AI.
        - Occipital, Temporal, Cerebellum, Limbic, and Parietal lobes now implement two-layer neural networks (input-hidden-output) with backpropagation for learning. Hidden layers typically use `tanh` activation.
        - FrontalLobeAI has been upgraded to use Q-learning for decision-making.
    Main Script: main.py initializes the AIs, routes inputs/outputs, and manages the "daytime" (task processing) and "bedtime" (consolidation) cycles.
    Functionality:
        Each AI processes tasks relevant to its brain region.
        Learning occurs via backpropagation for most modules and Q-learning for the Frontal lobe.
        Bedtime consolidation refines models by replaying experiences and saving updates to disk.
        Purpose emerges from environmental feedback, with no predefined goals.
    Dependencies: This project uses Python with NumPy and Pillow. Install dependencies using the provided `requirements.txt` file. For simplicity, heavy frameworks like TensorFlow are avoided, but the system can be scaled up.
    File Structure:
    project/
    ├── main.py
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
    - Most AI modules (Occipital, Temporal, Cerebellum, Limbic, Parietal) now feature two-layer neural networks, allowing for more complex pattern recognition and function approximation.
    - FrontalLobeAI employs Q-learning to improve its decision-making based on rewards.
    - Data input has been enhanced to support image files (for Occipital) and paired image-text data for cross-modal learning experiments.

## Explanation

    Modules: Each module defines a class with:
        - A model: Two-layer neural networks (input-hidden-output with biases) for most modules. FrontalLobeAI uses a Q-table approximated by a linear model.
        - process_task: Handles inputs specific to the AI’s role.
        - learn: Updates weights via backpropagation (for most modules) or Q-learning updates (FrontalLobeAI) based on feedback and experiences.
        - consolidate: Replays experiences from memory at "bedtime," further refining weights and saving them.
        - save_model/load_model: Persists weights (and for TemporalLobeAI, structured memories) to disk, with backward compatibility for older model formats.
    main.py:
        - Initializes all AIs and coordinates tasks.
        - process_day: Manages data flow (including image paths and text descriptions from paired data), routes inputs to appropriate AIs, collects outputs, and applies feedback for learning.
        - bedtime: Triggers consolidation for all AIs.
    Learning and Growth:
        - AIs start with random weights.
        - They learn from feedback during tasks (e.g., rewards, errors, target values) and consolidate nightly.
        - OccipitalLobeAI processes images and learns to classify them.
        - TemporalLobeAI processes text, learns text embeddings, and can associate text with visual labels.
        - FrontalLobeAI learns to choose actions using Q-learning.
    Storage: Models and memories are saved in the `data/` folder as JSON files.

## Running the System

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
    The script simulates activity cycles, processing data and consolidating learning each "night." Outputs show evolving actions, motor commands, and emotions.

## Example Output
(This section can remain largely the same, as the high-level output format hasn't changed, though specific values will differ due to new algorithms.)
text
Day 1
Action: X, Motor: [...], Emotion: Y
Starting bedtime consolidation...
Consolidation complete.
...

## Customization and Next Steps

    Real Inputs: Continue to expand with more diverse and real-world data. Consider OpenCV for more advanced vision tasks or audio libraries for direct sound processing by the Temporal Lobe.
    Scaling Models: While most modules now have a hidden layer, further scaling (more layers, more neurons) or specialized architectures (e.g., convolutional layers for Occipital, recurrent layers for Temporal) can be explored.
    Advanced RL for Frontal Lobe: The Q-learning in FrontalLobeAI can be enhanced (e.g., Deep Q-Networks (DQN) if TensorFlow/PyTorch were added, more complex state/reward definitions).
    Purpose Emergence: Refine reward functions and feedback mechanisms to guide specialization and emergent behaviors more effectively.
    Visualization: Develop tools to visualize network states, learned features, or decision processes.
    Specific Use Case: Tailor inputs, outputs, and rewards to simulate specific scenarios (e.g., simple robotics, chatbot interaction).
    Refine Modules: Continue to add features like curiosity-driven exploration, more sophisticated memory models, or attention mechanisms.
