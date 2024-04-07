import os
import numpy as np
from glob import glob
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from vit import ViT

def load_data(dataset_path, split=0.1):
    train_path = os.path.join(dataset_path, "train")
    test_path = os.path.join(dataset_path, "test")
    val_path = os.path.join(dataset_path, "val")

    # Load images and labels for training set
    train_images_normal = np.array(glob(os.path.join(train_path, "Normal", "*.jpg")))
    train_images_pneumonia = np.array(glob(os.path.join(train_path, "Pneumonia", "*.jpg")))
    train_labels_normal = np.zeros(len(train_images_normal), dtype=np.int32)
    train_labels_pneumonia = np.ones(len(train_images_pneumonia), dtype=np.int32)

    # Load images and labels for testing set
    test_images_normal = np.array(glob(os.path.join(test_path, "Normal", "*.jpg")))
    test_images_pneumonia = np.array(glob(os.path.join(test_path, "Pneumonia", "*.jpg")))
    test_labels_normal = np.zeros(len(test_images_normal), dtype=np.int32)
    test_labels_pneumonia = np.ones(len(test_images_pneumonia), dtype=np.int32)

    # Load images and labels for validation set
    val_images_normal = np.array(glob(os.path.join(val_path, "Normal", "*.jpg")))
    val_images_pneumonia = np.array(glob(os.path.join(val_path, "Pneumonia", "*.jpg")))
    val_labels_normal = np.zeros(len(val_images_normal), dtype=np.int32)
    val_labels_pneumonia = np.ones(len(val_images_pneumonia), dtype=np.int32)

    # Concatenate images and labels
    train_images = np.concatenate((train_images_normal, train_images_pneumonia), axis=0)
    train_labels = np.concatenate((train_labels_normal, train_labels_pneumonia), axis=0)
    test_images = np.concatenate((test_images_normal, test_images_pneumonia), axis=0)
    test_labels = np.concatenate((test_labels_normal, test_labels_pneumonia), axis=0)
    val_images = np.concatenate((val_images_normal, val_images_pneumonia), axis=0)
    val_labels = np.concatenate((val_labels_normal, val_labels_pneumonia), axis=0)

    # Shuffle data
    train_images, train_labels = shuffle(train_images, train_labels, random_state=42)
    test_images, test_labels = shuffle(test_images, test_labels, random_state=42)
    val_images, val_labels = shuffle(val_images, val_labels, random_state=42)

    # Split data into train, validation, and test sets
    train_data = (train_images, train_labels)
    test_data = (test_images, test_labels)
    val_data = (val_images, val_labels)

    return train_data, val_data, test_data

if __name__ == "__main__":
    # Dataset path
    dataset_path = r"C:\Datasets\pneumonia\chest_xray"
    # Load data
    try:
        (train_images, train_labels), (val_images, val_labels), (test_images, test_labels) = load_data(dataset_path)
    except FileNotFoundError as e:
        print(e)
        exit(1)
