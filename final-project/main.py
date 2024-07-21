import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

# The Path to Dataset
dataset_path = 'garbage-dataset'

# Categories and Respective Image Counts
categories = {
    "metal": 1869,
    "glass": 4097,
    "biological": 985,
    "paper": 2727,
    "battery": 945,
    "trash": 834,
    "cardboard": 2341,
    "shoes": 1977,
    "clothes": 5325,
    "plastic": 2542
}

# Function to Load and Preprocess Grayscale Images
def load_and_preprocess_grayscale_images(category, count):
    images = []
    labels = []
    category_path = os.path.join(dataset_path, category)
    for i, img_name in enumerate(os.listdir(category_path)):
        if i >= count:
            break
        img_path = os.path.join(category_path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            # resize image
            img = cv2.resize(img, (64, 64))
            # convert to grayscale
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # normalize each image by scaling pixel values to the range [0, 1]
            gray_img = gray_img / 255.0
            images.append(gray_img)
            labels.append(category)
    return images, labels

# Function to Load and Preprocess RGB Images
def load_and_preprocess_rgb_images(category, count):
    images = []
    labels = []
    category_path = os.path.join(dataset_path, category)
    for i, img_name in enumerate(os.listdir(category_path)):
        if i >= count:
            break
        img_path = os.path.join(category_path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            # resize image
            img = cv2.resize(img, (64, 64))
            # normalize each image by scaling pixel values to the range [0, 1]
            img = img / 255.0
            images.append(img)
            labels.append(category)
    return images, labels

# Load and Preprocess All Grayscale Images
all_images = []
all_labels = []
for category, count in categories.items():
    images, labels = load_and_preprocess_grayscale_images(category, count)
    all_images.extend(images)
    all_labels.extend(labels)

# Convert Lists to numpy Arrays
all_images = np.array(all_images)
all_labels = np.array(all_labels)

# Reshape images for the Neural Network
all_images = all_images.reshape((all_images.shape[0], 64, 64, 1))

# Encode labels to integers
label_encoder = LabelEncoder()
all_labels_encoded = label_encoder.fit_transform(all_labels)
all_labels_encoded = to_categorical(all_labels_encoded)

# Split the data into training, validation, and testing datasets
X_train, X_temp, y_train, y_temp = train_test_split(all_images, all_labels_encoded, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

# Function to Create a CNN Model
def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to Train and Plot the Model
def train_and_plot(model, model_name, X_train, y_train, X_val, y_val):
    checkpoint = ModelCheckpoint(f'best_model_{model_name}.keras', monitor='val_accuracy', save_best_only=True, mode='max')
    
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint]
    )
    
    # Plotting Error and Accuracy Curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Training and Validation Loss ({model_name})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Training and Validation Accuracy ({model_name})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

    return history

# Training the CNN on Grayscale Images
input_shape = (64, 64, 1)
num_classes = len(categories)

cnn_model_grayscale = create_cnn_model(input_shape, num_classes)
history_grayscale = train_and_plot(cnn_model_grayscale, 'grayscale', X_train, y_train, X_val, y_val)

# Load and Preprocess All RGB Images
all_images_rgb = []
all_labels_rgb = []
for category, count in categories.items():
    images, labels = load_and_preprocess_rgb_images(category, count)
    all_images_rgb.extend(images)
    all_labels_rgb.extend(labels)

# Convert Lists to numpy Arrays
all_images_rgb = np.array(all_images_rgb)
all_labels_rgb = np.array(all_labels_rgb)

# Encode labels to integers
all_labels_rgb_encoded = label_encoder.fit_transform(all_labels_rgb)
all_labels_rgb_encoded = to_categorical(all_labels_rgb_encoded)

# Split the data into training, validation, and testing datasets
X_train_rgb, X_temp_rgb, y_train_rgb, y_temp_rgb = train_test_split(all_images_rgb, all_labels_rgb_encoded, test_size=0.2, random_state=42)
X_val_rgb, X_test_rgb, y_val_rgb, y_test_rgb = train_test_split(X_temp_rgb, y_temp_rgb, test_size=0.2, random_state=42)

# Training the CNN on RGB Images
input_shape_rgb = (64, 64, 3)

cnn_model_rgb = create_cnn_model(input_shape_rgb, num_classes)
history_rgb = train_and_plot(cnn_model_rgb, 'rgb', X_train_rgb, y_train_rgb, X_val_rgb, y_val_rgb)

# Load the Best Model
best_model_path = 'best_model_rgb.keras' if max(history_rgb.history['val_accuracy']) > max(history_grayscale.history['val_accuracy']) else 'best_model_grayscale.keras'
best_model = tf.keras.models.load_model(best_model_path)

# Test the Best Model
X_test = X_test_rgb if best_model_path == 'best_model_rgb.keras' else X_test
y_test = y_test_rgb if best_model_path == 'best_model_rgb.keras' else y_test

y_pred = best_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Confusion Matrix and F1 Score
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')

print("Confusion Matrix:")
print(conf_matrix)
print("Average F1 Score:")
print(f1)
