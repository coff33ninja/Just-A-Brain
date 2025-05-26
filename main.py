# main.py (Central Coordinator)
import numpy as np
from frontal import FrontalLobeAI
from parietal import ParietalLobeAI
from temporal import TemporalLobeAI
from occipital import OccipitalLobeAI
from cerebellum import CerebellumAI
from limbic import LimbicSystemAI
import os


class BrainCoordinator:
    def __init__(self):
        os.makedirs("data", exist_ok=True)
        self.frontal = FrontalLobeAI()
        self.parietal = ParietalLobeAI()
        self.temporal = TemporalLobeAI()
        self.occipital = OccipitalLobeAI()
        self.cerebellum = CerebellumAI()
        self.limbic = LimbicSystemAI()

    def process_day(self, inputs):
        """Simulate a day's tasks."""
        # Example inputs: {vision, sensors, text, feedback}
        vision_data = inputs.get("vision", np.random.randn(25))  # Random image
        sensor_data = inputs.get("sensors", np.random.randn(20))  # Random sensors
        text_data = inputs.get("text", np.random.randn(15))  # Random text
        feedback = inputs.get(
            "feedback",
            {
                "action_reward": 1,
                "spatial_error": [0, 0, 0],
                "memory_target": text_data,
                "vision_label": 0,
                "motor_command": [0, 0, 0],
                "emotion_label": 0,
            },
        )

        # Process tasks
        vision_result = self.occipital.process_task(vision_data)
        spatial_result = self.parietal.process_task(sensor_data)
        memory_result = self.temporal.process_task(text_data)
        action = self.frontal.process_task(
            np.concatenate([vision_result, spatial_result, memory_result])
        )
        motor_command = self.cerebellum.process_task(sensor_data)
        emotion = self.limbic.process_task(text_data)

        # Learn from feedback
        self.occipital.learn(vision_data, feedback["vision_label"])
        self.parietal.learn(sensor_data, feedback["spatial_error"])
        self.temporal.learn(text_data, feedback["memory_target"])
        self.frontal.learn(
            np.concatenate([vision_result, spatial_result, memory_result]),
            action,
            feedback["action_reward"],
        )
        self.cerebellum.learn(sensor_data, feedback["motor_command"])
        self.limbic.learn(
            text_data, feedback["emotion_label"], feedback["action_reward"]
        )

        return {"action": action, "motor": motor_command, "emotion": emotion}

    def bedtime(self):
        """Consolidate all AIs' experiences."""
        print("Starting bedtime consolidation...")
        self.frontal.consolidate()
        self.parietal.consolidate()
        self.temporal.consolidate()
        self.occipital.consolidate()
        self.cerebellum.consolidate()
        self.limbic.consolidate()
        print("Consolidation complete.")


def main():
    coordinator = BrainCoordinator()
    # Simulate a day with random inputs (replace with real data)
    inputs = {
        "vision": np.random.randn(25),
        "sensors": np.random.randn(20),
        "text": np.random.randn(15),
        "feedback": {
            "action_reward": 1,
            "spatial_error": [0, 0, 0],
            "memory_target": np.random.randn(10),
            "vision_label": 0,
            "motor_command": [0, 0, 0],
            "emotion_label": 0,
        },
    }
    for day in range(3):  # Simulate 3 days
        print(f"Day {day + 1}")
        result = coordinator.process_day(inputs)
        print(
            f"Action: {result['action']}, Motor: {result['motor']}, Emotion: {result['emotion']}"
        )
        coordinator.bedtime()


if __name__ == "__main__":
    main()
