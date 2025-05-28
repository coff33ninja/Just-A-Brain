# occipital.py (Visual Processing)
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input  # type: ignore
import numpy as np
from collections import deque # For memory buffer
import os # type: ignore
from PIL import Image  # Still needed for image loading and preprocessing


class OccipitalLobeAI:
    def __init__(
        self, model_path="data/occipital_model.weights.h5"
    ):  # Changed extension
        self.input_shape = (64, 64, 3)  # Height, Width, Channels (color images)
        self.output_size = 5  # Number of object labels/classes
        self.model_path = model_path
        self.memory = deque(maxlen=100) # Store up to 100 recent experiences
        self.consolidation_batch_size = 16 # Batch size for consolidation training

        self.model = self._build_model()
        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        self.load_model()  # Load weights if they exist

    def _build_model(self):
        model = Sequential(
            [
            Input(shape=self.input_shape, name="input_layer"), # Add Input layer
                Conv2D(
                    32,
                    (3, 3),
                    activation="relu",
                # input_shape removed from here
                name="conv2d_1"
                ),
                MaxPooling2D((2, 2), name="maxpool_1"),
                Conv2D(64, (3, 3), activation="relu", name="conv2d_2"),
                MaxPooling2D((2, 2), name="maxpool_2"),
                Flatten(name="flatten"),
                Dense(64, activation="relu", name="dense_1"),
                # Dropout(0.5), # Optional for regularization, ensure it's imported if used
                Dense(
                    self.output_size, activation="softmax", name="output_dense"
                ),  # Softmax for multi-class
            ]
        )
        # model.summary() # You can uncomment this to print summary when an instance is created
        return model

    def save_model(self):
        print(f"Saving Occipital Lobe model weights to {self.model_path}...")
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        try:
            if not self.model_path.endswith(".weights.h5"):
                print(f"Occipital Lobe: Warning: model_path '{self.model_path}' does not end with '.weights.h5'. Keras save_weights prefers this extension.")
            self.model.save_weights(self.model_path)
            print("Occipital Lobe model weights saved.")
        except Exception as e:
            print(f"Error saving Occipital Lobe model weights: {e}")

    def load_model(self):
        if os.path.exists(self.model_path):
            print(f"Loading Occipital Lobe model weights from {self.model_path}...")
            try:
                self.model.load_weights(self.model_path)
                print("Occipital Lobe model weights loaded successfully.")
            except Exception as e:
                print(
                    f"Error loading Occipital Lobe model weights: {e}. Model remains initialized with new weights."
                )
        else:
            print(
                f"No pre-trained weights found at {self.model_path} for Occipital Lobe. Model is initialized with new weights."
            )

    def _preprocess_image(self, image_path):
        """
        Loads and preprocesses an image to be suitable for the CNN.
        Returns a numpy array (batch_size, height, width, channels) or None if error.
        """
        try:
            if not os.path.exists(image_path):
                print(f"Occipital Lobe: Error: Image file not found at {image_path}")
                return None
            img = Image.open(image_path).convert("RGB")  # Ensure 3 channels
            img = img.resize((self.input_shape[0], self.input_shape[1]))
            img_array = np.array(img) / 255.0  # Normalize to [0, 1]
            return np.expand_dims(img_array, axis=0)  # Add batch dimension
        except FileNotFoundError:
            print(f"Occipital Lobe: Error: Image file not found at {image_path}")
            return None
        except Exception as e:
            print(f"Occipital Lobe: Error processing image {image_path}: {e}")
            return None

    def process_task(self, image_path):
        # Ensure os and np are imported in the file
        print(
            f"Occipital Lobe: Processing task for image {os.path.basename(image_path)}..."
        )
        processed_image_batch = self._preprocess_image(image_path)

        if processed_image_batch is None:
            # Error message already printed by _preprocess_image
            print(
                f"Occipital Lobe: Preprocessing failed for {os.path.basename(image_path)}. Cannot predict."
            )
            return -1  # Indicator of failure

        try:
            print("Occipital Lobe: Predicting with CNN model...")
            predictions = self.model.predict(processed_image_batch, verbose=0)
            # predictions is typically a 2D array, e.g., [[0.1, 0.7, 0.2]] for batch size 1
            # We need the index of the highest score in the first (and only) item of the batch.
            predicted_label = np.argmax(predictions[0])

            # Optional: Log the raw scores for debugging
            scores_str = ", ".join([f"{score:.3f}" for score in predictions[0]])
            print(
                f"Occipital Lobe: Raw prediction scores for {os.path.basename(image_path)}: [{scores_str}]"
            )

            print(
                f"Occipital Lobe: Predicted label for {os.path.basename(image_path)} is {predicted_label}."
            )
            return int(predicted_label)  # Ensure it's a standard int
        except Exception as e:
            print(
                f"Occipital Lobe: Error during model prediction for {os.path.basename(image_path)}: {e}"
            )
            return -1  # Indicator of failure

    def learn(self, image_path, label):
        print(
            f"Occipital Lobe: learn called for image {os.path.basename(image_path)} with label {label}"
        )
        processed_image_batch = self._preprocess_image(image_path)

        if processed_image_batch is None:
            # Error already logged by _preprocess_image
            print(
                f"Occipital Lobe: Preprocessing failed for image {image_path}, skipping training."
            )
            return

        try:
            # Ensure label is an integer and prepare for Keras
            valid_label = int(label)
        except ValueError:
            print(
                f"Occipital Lobe: Invalid label '{label}'. Must be an integer. Skipping training."
            )
            return

        if not (0 <= valid_label < self.output_size):
            print(
                f"Occipital Lobe: Label {valid_label} is out of range (expected 0 to {self.output_size - 1}). Skipping training."
            )
            return

        target_label_array = np.array([valid_label])

        print(
            f"Occipital Lobe: Training on image {os.path.basename(image_path)} with label {valid_label}..."
        )
        try:
            history = self.model.fit(
                processed_image_batch, target_label_array, epochs=1, verbose=0
            )
            # Safely get loss and accuracy from history
            loss = history.history.get("loss", [float("nan")])[0]
            accuracy = history.history.get("accuracy", [float("nan")])[0]
            print(
                f"Occipital Lobe: Training complete. Loss: {loss:.4f}, Accuracy: {accuracy:.4f}"
            )
            # Add to memory after successful training
            self.memory.append((image_path, valid_label))
            print(f"Occipital Lobe: Added ({os.path.basename(image_path)}, {valid_label}) to memory. Memory size: {len(self.memory)}")
        except Exception as e:
            print(f"Occipital Lobe: Error during model training: {e}")

    def consolidate(self):
        print("Occipital Lobe: Starting consolidation...")
        if not self.memory:
            print("Occipital Lobe: Memory is empty. Nothing to consolidate beyond saving.")
            self.save_model()
            return

        print(f"Occipital Lobe: Consolidating {len(self.memory)} experiences from memory.")

        # Prepare data for batch training
        images_to_train = []
        labels_to_train = []

        for img_path, lbl in list(self.memory): # Iterate over a snapshot
            processed_img_batch = self._preprocess_image(img_path)
            if processed_img_batch is not None:
                images_to_train.append(processed_img_batch[0]) # Get the image array from the batch
                labels_to_train.append(int(lbl))
            else:
                print(f"Occipital Lobe: Skipping {img_path} in consolidation due to preprocessing error.")

        if not images_to_train:
            print("Occipital Lobe: No valid images to train on after preprocessing memory. Saving model.")
            self.save_model()
            return

        images_np = np.array(images_to_train)
        labels_np = np.array(labels_to_train)

        print(f"Occipital Lobe: Training consolidation batch of size {images_np.shape[0]}...")
        try:
            consolidation_epochs = 1 # Could be configurable
            history = self.model.fit(
                images_np,
                labels_np,
                epochs=consolidation_epochs,
                batch_size=min(self.consolidation_batch_size, len(images_np)), # Ensure batch_size <= num_samples
                verbose=0
            )
            loss = history.history.get('loss', [float('nan')])[0]
            accuracy = history.history.get('accuracy', [float('nan')])[0]
            print(f"Occipital Lobe: Consolidation training complete. Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        except Exception as e:
            print(f"Occipital Lobe: Error during consolidation training: {e}")

        self.save_model()
        print("Occipital Lobe: Consolidation complete (replayed experiences and saved model).")


# Example usage (optional, for testing within the file)
if __name__ == "__main__":
    dummy_image_dir = "data/images"
    dummy_image_path = os.path.join(dummy_image_dir, "test_dummy_occipital.png")
    os.makedirs(dummy_image_dir, exist_ok=True)

    if not os.path.exists(dummy_image_path):
        try:
            Image.new("RGB", (64, 64), color="blue").save(
                dummy_image_path
            )  # Changed color for visibility
            print(f"Created dummy image at {dummy_image_path}")
        except Exception as e:
            print(f"Could not create dummy image: {e}")

    print("\n--- Testing OccipitalLobeAI Initialization ---")
    test_model_path = "data/test_temp_occipital_cnn.weights.h5"
    occipital_ai = OccipitalLobeAI(model_path=test_model_path)

    # Train the model a bit to get non-random predictions
    if os.path.exists(dummy_image_path):
        print("\n--- Initial training pass (to make predictions meaningful) ---")
        for i in range(occipital_ai.output_size):  # Train on each label once
            occipital_ai.learn(dummy_image_path, i)
        print("--- Initial training pass complete ---")

    print("\n--- Testing process_task method ---")
    if os.path.exists(dummy_image_path):
        print("\nTesting process_task with valid image:")
        predicted_label = occipital_ai.process_task(dummy_image_path)
        print(f"process_task final returned label: {predicted_label}")
    else:
        print(
            f"Skipping process_task test as dummy image {dummy_image_path} not found."
        )

    print("\nTesting process_task with invalid image path:")
    invalid_image_path = "data/images/non_existent_image.png"
    predicted_label_invalid = occipital_ai.process_task(invalid_image_path)
    print(
        f"process_task with invalid path final returned label: {predicted_label_invalid}"
    )

    # Clean up dummy model file
    if os.path.exists(test_model_path):
        os.remove(test_model_path)
        print(f"\nCleaned up dummy model weights file: {test_model_path}")

    print("\nOccipital Lobe AI (CNN) process_task method test script finished.")
