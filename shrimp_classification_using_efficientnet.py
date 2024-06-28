import os
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import efficientnet.tfkeras as efn
from distutils.dir_util import copy_tree, remove_tree
from PIL import Image
from random import randint
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing.image import ImageDataGenerator


from google.colab import drive
drive.mount('/content/drive')


# Define directory paths
base_dir = "/content/drive/MyDrive/Colab Notebooks/Augmented/datasets"
test_dir = os.path.join(base_dir, "test")
train_dir = os.path.join(base_dir, "train")
work_dir = "/content/dataset/"  # Adjust this path according to your preference


# Create the working directory
if os.path.exists(work_dir):
    shutil.rmtree(work_dir)
os.makedirs(work_dir)

# Copy data from train and test directories to the working directory

# Define categories
categories = ["Bagda", "Golda"]

# Iterate over each category
for category in categories:
    # Ensure the destination directory exists before copying files
    destination_dir = os.path.join(work_dir, "train", category)
    os.makedirs(destination_dir, exist_ok=True)

    # Copy files from the corresponding source directory
    source_dir = os.path.join(train_dir, category)
    copy_tree(source_dir, destination_dir)


print("Working Directory Contents:", os.listdir(work_dir))

# Performing Image Augmentation to have more data samples
ZOOM = [.99, 1.01]
BRIGHT_RANGE = [0.8, 1.2]
HORZ_FLIP = True
FILL_MODE = "constant"
DATA_FORMAT = "channels_last"
DIM = (224, 224)  # Assuming this as the input dimension

# Create an ImageDataGenerator for augmentation
work_dr = ImageDataGenerator(rescale=1./255, brightness_range=BRIGHT_RANGE, zoom_range=ZOOM, data_format=DATA_FORMAT, fill_mode=FILL_MODE, horizontal_flip=HORZ_FLIP)

# Flow training images in batches using the ImageDataGenerator
train_data_gen = work_dr.flow_from_directory(directory=work_dir, target_size=DIM, batch_size=32, class_mode='categorical', shuffle=True)


def construct_efficientnet_model():
    """Constructing an EfficientNet model for performing the classification task."""
    base_model = efn.EfficientNetB0(weights='imagenet', include_top=False, input_shape=(DIM[0], DIM[1], 3))
    base_model.trainable = False

    model = Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(2, activation='softmax')
    ])

    return model

# Define other parameters for our EfficientNet model
model = construct_efficientnet_model()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()



# Fit the training data to the model and validate it using the validation data
EPOCHS = 100
history = model.fit(train_data_gen, epochs=EPOCHS)

# Save the model
model.save('efficientnet_model.h5')



#Retrieve the data from the ImageDataGenerator iterator
train_data, train_labels = train_data_gen.next()

#Splitting the dataset
train_data, test_data, train_labels, test_labels = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

#Evaluate the model
test_scores = model.evaluate(test_data, test_labels, verbose=0)
print("Testing Accuracy: %.2f%%" % (test_scores[1] * 100))


# Retrieve the data from the ImageDataGenerator iterator for test data
test_data, test_labels = train_data_gen.next()

# Predict on test data
pred_labels = model.predict(test_data)

# Since the labels are softmax arrays, we need to round them off to have them in the form of 0s and 1s,
# similar to the test_labels
def roundoff(arr):
    """To round off according to the argmax of each predicted label array."""
    arr[np.argwhere(arr != arr.max())] = 0
    arr[np.argwhere(arr == arr.max())] = 1
    return arr

# Rounding off the predicted labels
for labels in pred_labels:
    labels = roundoff(labels)

target_names = ['Golda', 'Bagda']

# Printing the classification report
print(classification_report(test_labels.argmax(axis=1), pred_labels.argmax(axis=1), target_names=target_names))




# Since the labels are softmax arrays, we need to round off to have it in the form of 0s and 1s,
# similar to the test_labels
def roundoff(arr):
    """To round off according to the argmax of each predicted label array."""
    arr[np.argwhere(arr != arr.max())] = 0
    arr[np.argwhere(arr == arr.max())] = 1
    return arr

for labels in pred_labels:
    labels = roundoff(labels)

target_names = ['Golda', 'Bagda']

# Plot the confusion matrix to understand the classification in detail
pred_ls = np.argmax(pred_labels, axis=1)
test_ls = np.argmax(test_labels, axis=1)

conf_arr = confusion_matrix(test_ls, pred_ls)

plt.figure(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

ax = sns.heatmap(conf_arr, cmap='Greens', annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names)

plt.title('Shrimp Classification')
plt.xlabel('Prediction')
plt.ylabel('Truth')
plt.show(ax)
