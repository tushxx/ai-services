import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)


# ============================================================
# CONSTANTS
# ============================================================

IMAGE_SIZE = (48, 48)

# Alphabetical order required by the question
CLASS_NAMES = [
    "hole",
    "oil_stain",
    "ok",
    "weave_fault"
]


# ============================================================
# 1. IMAGE PREPARATION
# ============================================================

def preprocess_images(input_dir, output_path, labelled):
    """
    Prepare images for CNN training/testing/prediction.

    labelled=True:
        - Reads images from category folders
        - Creates integer labels according to alphabetical
          folder order
        - Saves images + labels

    labelled=False:
        - Reads images directly from input_dir
        - Keeps filenames
        - Saves images + image_ids
    """

    input_dir = Path(input_dir)
    output_path = Path(output_path)

    images = []
    labels = []
    image_ids = []

    if labelled:
        # Category folders in alphabetical order
        category_dirs = sorted(
            [p for p in input_dir.iterdir() if p.is_dir()],
            key=lambda p: p.name
        )

        # Process each category
        for label, category_dir in enumerate(category_dirs):

            # Images in sorted filename order
            image_files = sorted(
                [
                    p for p in category_dir.iterdir()
                    if p.is_file()
                ],
                key=lambda p: p.name
            )

            for image_file in image_files:
                try:
                    with Image.open(image_file) as img:
                        img = img.convert("L")
                        img = img.resize(IMAGE_SIZE)

                        arr = np.asarray(
                            img,
                            dtype=np.float32
                        ) / 255.0

                        # Add channel dimension
                        arr = np.expand_dims(arr, axis=-1)

                        images.append(arr)
                        labels.append(label)

                except Exception:
                    # Ignore files that are not valid images
                    continue

    else:
        # Unlabelled images directly inside input directory
        image_files = sorted(
            [
                p for p in input_dir.iterdir()
                if p.is_file()
            ],
            key=lambda p: p.name
        )

        for image_file in image_files:
            try:
                with Image.open(image_file) as img:
                    img = img.convert("L")
                    img = img.resize(IMAGE_SIZE)

                    arr = np.asarray(
                        img,
                        dtype=np.float32
                    ) / 255.0

                    arr = np.expand_dims(arr, axis=-1)

                    images.append(arr)
                    image_ids.append(image_file.name)

            except Exception:
                continue

    # Convert lists to NumPy arrays
    images = np.asarray(images, dtype=np.float32)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save compressed NPZ
    if labelled:
        labels = np.asarray(labels, dtype=np.int64)

        np.savez_compressed(
            output_path,
            images=images,
            labels=labels
        )

        return images, labels

    else:
        image_ids = np.asarray(image_ids)

        np.savez_compressed(
            output_path,
            images=images,
            image_ids=image_ids
        )

        return images, image_ids


# ============================================================
# 2. NETWORK DEFINITION
# ============================================================

def build_model(input_shape, num_classes):
    """
    Build and compile the CNN.
    """

    model = Sequential([
        Input(shape=input_shape),

        Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same"
        ),
        MaxPooling2D((2, 2)),

        Conv2D(
            64,
            (3, 3),
            activation="relu",
            padding="same"
        ),
        MaxPooling2D((2, 2)),

        Conv2D(
            128,
            (3, 3),
            activation="relu",
            padding="same"
        ),
        MaxPooling2D((2, 2)),

        Flatten(),

        Dense(128, activation="relu"),
        Dropout(0.3),

        Dense(
            num_classes,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# 3. NETWORK TRAINING
# ============================================================

def train_model(processed_path, model_path, epochs):
    """
    Load prepared training data, train CNN and save it.
    """

    data = np.load(processed_path)

    images = data["images"]
    labels = data["labels"]

    # Derive number of classes from labels
    num_classes = len(np.unique(labels))

    # Build model using actual image shape
    model = build_model(
        input_shape=images.shape[1:],
        num_classes=num_classes
    )

    # Save validation data from the END of the prepared batch.
    # Keras validation_split takes the last samples before
    # training shuffling occurs.
    model.fit(
        images,
        labels,
        epochs=epochs,
        validation_split=0.2,
        shuffle=True,
        batch_size=32,
        verbose=0
    )

    # Native Keras format
    model_path = Path(model_path)
    model_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    model.save(model_path)

    return model


# ============================================================
# 4. NETWORK EVALUATION
# ============================================================

def evaluate_model(model_path, processed_path):
    """
    Load stored model and evaluate it on prepared test data.

    Returns only classification accuracy.
    """

    model = tf.keras.models.load_model(model_path)

    data = np.load(processed_path)

    images = data["images"]
    labels = data["labels"]

    # Do not train/refit the model
    _, accuracy = model.evaluate(
        images,
        labels,
        verbose=0
    )

    return float(accuracy)


# ============================================================
# 5. BATCH PREDICTION
# ============================================================

def predict_new_data(
    model_path,
    input_dir,
    processed_path,
    output_path
):
    """
    Predict classes for a fresh batch of unlabelled images.
    """

    # Load stored model
    model = tf.keras.models.load_model(model_path)

    # Prepare unlabelled images
    images, image_ids = preprocess_images(
        input_dir,
        processed_path,
        labelled=False
    )

    # Generate predictions
    probabilities = model.predict(
        images,
        verbose=0
    )

    # Numerical class indices
    predicted_indices = np.argmax(
        probabilities,
        axis=1
    )

    # Convert indices to category names
    predicted_classes = [
        CLASS_NAMES[index]
        for index in predicted_indices
    ]

    # Exactly two columns
    prediction_table = pd.DataFrame({
        "image_id": image_ids,
        "predicted_class": predicted_classes
    })

    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save CSV
    prediction_table.to_csv(
        output_path,
        index=False
    )

    return prediction_table


# ============================================================
# MAIN BLOCK
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    train_dir = "data/train"
    test_dir = "data/test"
    predict_dir = "data/predict"

    processed_dir = "processed_data"
    artifacts_dir = "artifacts"
    output_dir = "output"

    train_processed = (
        f"{processed_dir}/train_processed.npz"
    )

    test_processed = (
        f"{processed_dir}/test_processed.npz"
    )

    predict_processed = (
        f"{processed_dir}/predict_processed.npz"
    )

    model_path = (
        f"{artifacts_dir}/textile_defect_model.keras"
    )

    output_path = (
        f"{output_dir}/textile_predictions.csv"
    )

    # --------------------------------------------------------
    # 1. Preprocess training images
    # --------------------------------------------------------

    train_images, train_labels = preprocess_images(
        train_dir,
        train_processed,
        labelled=True
    )

    print(
        f"Processed training images: "
        f"{train_images.shape}"
    )

    # --------------------------------------------------------
    # 2. Preprocess held-back test images
    # --------------------------------------------------------

    test_images, test_labels = preprocess_images(
        test_dir,
        test_processed,
        labelled=True
    )

    print(
        f"Processed test images: "
        f"{test_images.shape}"
    )

    # --------------------------------------------------------
    # 3. Build / define network
    # --------------------------------------------------------

    model = build_model(
        input_shape=train_images.shape[1:],
        num_classes=len(np.unique(train_labels))
    )

    print("Model built and compiled.")

    # --------------------------------------------------------
    # 4. Train and save model
    # --------------------------------------------------------

    model = train_model(
        train_processed,
        model_path,
        epochs=10
    )

    print("Model trained and saved.")

    # --------------------------------------------------------
    # 5. Evaluate stored model
    # --------------------------------------------------------

    accuracy = evaluate_model(
        model_path,
        test_processed
    )

    print(f"Test accuracy: {accuracy}")

    # --------------------------------------------------------
    # 6. Predict unreviewed batch
    # --------------------------------------------------------

    predictions = predict_new_data(
        model_path,
        predict_dir,
        predict_processed,
        output_path
    )

    print(
        f"Predictions written: "
        f"{len(predictions)}"
    )