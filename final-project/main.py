import cv2
import os
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# Define image directories for each class
image_dirs = {
    'metal': 'garbage-dataset/metal',
    'glass': 'garbage-dataset/glass',
    'biological': 'garbage-dataset/biological',
    'paper': 'garbage-dataset/paper',
    'battery': 'garbage-dataset/battery',
    'trash': 'garbage-dataset/trash',
    'cardboard': 'garbage-dataset/cardboard',
    'shoes': 'garbage-dataset/shoes',
    'clothes': 'garbage-dataset/clothes',
    'plastic': 'garbage-dataset/plastic'
}


image_size = (64, 64)
max_images_per_class = 10  # Increased to 10 images per class

# Resize images function
def load_and_resize_images(image_paths, max_images):
    images = []
    for img_path in image_paths[:max_images]:
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, image_size)
            images.append(img)
    return images

# Load and resize images for all classes
data = {}
for label, dir_path in image_dirs.items():
    image_paths = glob(os.path.join(dir_path, '*.jpg'))
    data[label] = load_and_resize_images(image_paths, max_images_per_class)

def convert_to_grayscale(images):
    return [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

# Convert images to grayscale
grayscale_data = {label: convert_to_grayscale(imgs) for label, imgs in data.items()}

def normalize_images(images):
    return [img / 255.0 for img in images]

# Normalize images
normalized_data = {label: normalize_images(imgs) for label, imgs in grayscale_data.items()}

X, y = [], []

for label, images in normalized_data.items():
    X.extend(images)
    y.extend([label] * len(images))

X = np.array(X)
y = np.array(y)

# Flatten the images for SVM input
X_flat = X.reshape(len(X), -1)

# Ensure test size is sufficient for all classes
min_test_size = 10  # Minimum test size to accommodate all classes
test_size = max(0.2 * len(X_flat), min_test_size) / len(X_flat)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_flat, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded)

# Experiment 1: Train SVM on grayscale images
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Test the SVM model
y_pred_svm = svm_model.predict(X_test)

# Confusion Matrix and Classification Report for SVM
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
print("Confusion Matrix for SVM:")
print(conf_matrix_svm)

class_report_svm = classification_report(y_test, y_pred_svm, zero_division=0)
print("Classification Report for SVM:")
print(class_report_svm)

f1_svm = f1_score(y_test, y_pred_svm, average='weighted')
print(f"Average F1 Score for SVM: {f1_svm}")

# Split training set into training and validation sets for neural networks
X_train_nn, X_val_nn, y_train_nn, y_val_nn = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

# Reshape for neural networks
X_train_nn = X_train_nn.reshape(-1, 64, 64, 1)
X_val_nn = X_val_nn.reshape(-1, 64, 64, 1)
X_test_nn = X_test.reshape(-1, 64, 64, 1)

# One-hot encode labels
num_classes = len(image_dirs)
y_train_nn_encoded = tf.keras.utils.to_categorical(y_train_nn, num_classes=num_classes)
y_val_nn_encoded = tf.keras.utils.to_categorical(y_val_nn, num_classes=num_classes)
y_test_nn_encoded = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

# Experiment 2: Build and train two different neural networks
# Model 1
model_1 = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model_1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history_1 = model_1.fit(X_train_nn, y_train_nn_encoded, epochs=10, validation_data=(X_val_nn, y_val_nn_encoded))

# Model 2
model_2 = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model_2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history_2 = model_2.fit(X_train_nn, y_train_nn_encoded, epochs=10, validation_data=(X_val_nn, y_val_nn_encoded))

# Plotting function
def plot_history(history, model_name):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title(f'Training and Validation Accuracy - {model_name}')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title(f'Training and Validation Loss - {model_name}')

    plt.show()

# Plot the training history of both models
plot_history(history_1, 'Model 1')
plot_history(history_2, 'Model 2')

# Evaluate Model 1
test_loss_1, test_acc_1 = model_1.evaluate(X_test_nn, y_test_nn_encoded)
print(f'Model 1 - Test accuracy: {test_acc_1}, Test loss: {test_loss_1}')

# Evaluate Model 2
test_loss_2, test_acc_2 = model_2.evaluate(X_test_nn, y_test_nn_encoded)
print(f'Model 2 - Test accuracy: {test_acc_2}, Test loss: {test_loss_2}')

# Save the best model
best_model = model_1 if test_acc_1 > test_acc_2 else model_2
best_model.save("best_model.h5")

# Reload the best model for testing
loaded_model = tf.keras.models.load_model("best_model.h5")

# Test the best model
y_pred_test = loaded_model.predict(X_test_nn)
y_pred_test_classes = np.argmax(y_pred_test, axis=1)

# Calculate confusion matrix
conf_matrix_test = confusion_matrix(np.argmax(y_test_nn_encoded, axis=1), y_pred_test_classes)
print("Confusion Matrix - Test Data:")
print(conf_matrix_test)

# Classification Report for NN
class_report_nn = classification_report(y_test, y_pred_svm, zero_division=0)
print("Classification Report for NN:")
print(class_report_nn)

# Calculate average F1 score
f1_score_avg = f1_score(np.argmax(y_test_nn_encoded, axis=1), y_pred_test_classes, average='weighted')
print("Average F1 Score - Test Data:", f1_score_avg)

# Experiment 3: Train CNN on Grayscale and RGB Images
# Function to load and resize RGB images
def load_and_resize_images_rgb(image_paths, max_images):
    images = []
    for img_path in image_paths[:max_images]:
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, image_size)
            images.append(img)
    return images

# Load and resize RGB images for all classes
data_rgb = {}
for label, dir_path in image_dirs.items():
    image_paths = glob(os.path.join(dir_path, '*.jpg'))
    data_rgb[label] = load_and_resize_images_rgb(image_paths, max_images_per_class)

# Normalize RGB images
normalized_data_rgb = {label: normalize_images(imgs) for label, imgs in data_rgb.items()}

X_rgb, y_rgb = [], []

for label, images in normalized_data_rgb.items():
    X_rgb.extend(images)
    y_rgb.extend([label] * len(images))

X_rgb = np.array(X_rgb)
y_rgb = np.array(y_rgb)

# Encode labels for RGB images
y_rgb_encoded = label_encoder.transform(y_rgb)

# Split the RGB dataset into training, validation, and testing sets
X_train_rgb, X_test_rgb, y_train_rgb, y_test_rgb = train_test_split(X_rgb, y_rgb_encoded, test_size=test_size, random_state=42, stratify=y_rgb_encoded)
X_train_rgb, X_val_rgb, y_train_rgb, y_val_rgb = train_test_split(X_train_rgb, y_train_rgb, test_size=0.2, random_state=42, stratify=y_train_rgb)

# One-hot encode labels for RGB images
y_train_rgb_encoded = tf.keras.utils.to_categorical(y_train_rgb, num_classes=num_classes)
y_val_rgb_encoded = tf.keras.utils.to_categorical(y_val_rgb, num_classes=num_classes)
y_test_rgb_encoded = tf.keras.utils.to_categorical(y_test_rgb, num_classes=num_classes)

# CNN Model for Grayscale Images
model_cnn_gray = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model_cnn_gray.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history_cnn_gray = model_cnn_gray.fit(X_train_nn, y_train_nn_encoded, epochs=10, validation_data=(X_val_nn, y_val_nn_encoded))

# CNN Model for RGB Images
model_cnn_rgb = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model_cnn_rgb.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history_cnn_rgb = model_cnn_rgb.fit(X_train_rgb, y_train_rgb_encoded, epochs=10, validation_data=(X_val_rgb, y_val_rgb_encoded))

# Plotting function for CNN models
def plot_cnn_history(history, model_name):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title(f'Training and Validation Accuracy - {model_name}')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title(f'Training and Validation Loss - {model_name}')

    plt.show()

# Plot the training history of CNN models
plot_cnn_history(history_cnn_gray, 'CNN Grayscale Model')
plot_cnn_history(history_cnn_rgb, 'CNN RGB Model')

# Evaluate CNN Grayscale Model
test_loss_cnn_gray, test_acc_cnn_gray = model_cnn_gray.evaluate(X_test_nn, y_test_nn_encoded)
print(f'CNN Grayscale Model - Test accuracy: {test_acc_cnn_gray}, Test loss: {test_loss_cnn_gray}')

# Evaluate CNN RGB Model
test_loss_cnn_rgb, test_acc_cnn_rgb = model_cnn_rgb.evaluate(X_test_rgb, y_test_rgb_encoded)
print(f'CNN RGB Model - Test accuracy: {test_acc_cnn_rgb}, Test loss: {test_loss_cnn_rgb}')

# Confusion Matrix and Classification Report for CNN Grayscale Model
y_pred_cnn_gray = model_cnn_gray.predict(X_test_nn)
y_pred_cnn_gray_classes = np.argmax(y_pred_cnn_gray, axis=1)

conf_matrix_cnn_gray = confusion_matrix(np.argmax(y_test_nn_encoded, axis=1), y_pred_cnn_gray_classes)
print("Confusion Matrix for CNN Grayscale Model:")
print(conf_matrix_cnn_gray)

class_report_cnn_gray = classification_report(np.argmax(y_test_nn_encoded, axis=1), y_pred_cnn_gray_classes, zero_division=0)
print("Classification Report for CNN Grayscale Model:")
print(class_report_cnn_gray)

f1_score_avg_cnn_gray = f1_score(np.argmax(y_test_nn_encoded, axis=1), y_pred_cnn_gray_classes, average='weighted')
print("Average F1 Score for CNN Grayscale Model:", f1_score_avg_cnn_gray)

# Confusion Matrix and Classification Report for CNN RGB Model
y_pred_cnn_rgb = model_cnn_rgb.predict(X_test_rgb)
y_pred_cnn_rgb_classes = np.argmax(y_pred_cnn_rgb, axis=1)

conf_matrix_cnn_rgb = confusion_matrix(y_test_rgb, y_pred_cnn_rgb_classes)
print("Confusion Matrix for CNN RGB Model:")
print(conf_matrix_cnn_rgb)

class_report_cnn_rgb = classification_report(y_test_rgb, y_pred_cnn_rgb_classes, zero_division=0)
print("Classification Report for CNN RGB Model:")
print(class_report_cnn_rgb)

f1_score_avg_cnn_rgb = f1_score(y_test_rgb, y_pred_cnn_rgb_classes, average='weighted')
print("Average F1 Score for CNN RGB Model:", f1_score_avg_cnn_rgb)

# Load the best CNN model
best_cnn_model = model_cnn_gray if test_acc_cnn_gray > test_acc_cnn_rgb else model_cnn_rgb
best_cnn_model.save("best_cnn_model.h5")

# Reload the best CNN model for testing
loaded_cnn_model = tf.keras.models.load_model("best_cnn_model.h5")

# Test the best CNN model
if loaded_cnn_model.input_shape[-1] == 1:
    # Grayscale model
    X_test_cnn = X_test_nn
else:
    # RGB model
    X_test_cnn = X_test_rgb

y_pred_test_cnn = loaded_cnn_model.predict(X_test_cnn)
y_pred_test_cnn_classes = np.argmax(y_pred_test_cnn, axis=1)

# Calculate confusion matrix for best CNN model
conf_matrix_test_cnn = confusion_matrix(np.argmax(y_test_nn_encoded, axis=1), y_pred_test_cnn_classes)
print("Confusion Matrix - Best CNN Model:")
print(conf_matrix_test_cnn)

# Classification Report for best CNN model
class_report_test_cnn = classification_report(np.argmax(y_test_nn_encoded, axis=1), y_pred_test_cnn_classes, zero_division=0)
print("Classification Report - Best CNN Model:")
print(class_report_test_cnn)

# Calculate average F1 score for best CNN model
f1_score_avg_test_cnn = f1_score(np.argmax(y_test_nn_encoded, axis=1), y_pred_test_cnn_classes, average='weighted')
print("Average F1 Score - Best CNN Model:", f1_score_avg_test_cnn)
