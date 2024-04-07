import os
import numpy as np
import tensorflow as tf
from train import load_data
from vit import ViT

if __name__ == "__main__":
    # Dataset path
    dataset_path = r"C:\Datasets\pneumonia\chest_xray"
    # Load test data
    (_, _), (_, _), (test_images, test_labels) = load_data(dataset_path)
    
    # Load the trained model
    model = tf.keras.models.load_model("files/model.h5")

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(test_images, test_labels, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Predict probabilities for each class
    probabilities = model.predict(test_images)

    # Convert probabilities to class predictions
    predictions = np.argmax(probabilities, axis=1)

    # Print some example predictions
    print("Example predictions:")
    for i in range(10):
        print(f"Predicted class: {predictions[i]}, True class: {test_labels[i]}")
