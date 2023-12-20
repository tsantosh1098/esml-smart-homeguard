# package import
import os
import cv2
import tqdm
import time
import shutil
import json
import random
import pathlib
from pathlib import PosixPath
import itertools
import collections
import numpy as np
from IPython import display
from urllib import request

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras
from tensorflow.keras.applications import VGG16, InceptionResNetV2, MobileNetV2
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.utils import to_categorical


# InceptionResNetV2
# Input shape to be used by the model
pInputShape = (229, 229, 3)
pNumFrames  = 30
pModelName  = "InceptionResNetV2"
pStrategyForFrame = "smart_select"
pBatchSize  = 2
pFeatureExtractorOutShape = 1536



# MobileNetV2
# Input shape to be used by the model
pInputShape = (224, 224, 3)
pNumFrames  = 30
pModelName  = "MobileNetV2"
pStrategyForFrame = "smart_select"
pBatchSize  = 2
pFeatureExtractorOutShape = 1280

main_path = "/s/chopin/l/grad/tskumar/Documents/ESML/datasets/dataset3"
train_data_path = "/s/chopin/l/grad/tskumar/Documents/ESML/datasets/dataset3/train"
test_data_path = "/s/chopin/l/grad/tskumar/Documents/ESML/datasets/dataset3/test"

def FeatureExtractor(model_name, input_shape):
    if model_name == "InceptionResNetV2":
        # use pretrained model as a feature extractor
        pretrained_model = InceptionResNetV2(
            input_shape = input_shape,
            include_top = False,
            weights='imagenet',  # Use pre-trained weights
        )
        
        # Exclude the last two dense layers
        output_layer = pretrained_model.get_layer('conv_7b_ac').output
        feature_extractor = Model(inputs = pretrained_model.input, outputs = output_layer)
        return feature_extractor

    elif model_name == "VGG16":
        # use pretrained model as a feature extractor
        pretrained_model = InceptionResNetV2(
            input_shape= input_shape,
            include_top=False,
            weights='imagenet',  # Use pre-trained weights
        )
        
        # Exclude the last two dense layers
        output_layer = pretrained_model.get_layer('fc2').output
        feature_extractor = Model(inputs = pretrained_model.input, outputs = output_layer)
        return feature_extractor

    elif model_name == "InceptionV3":
        # use pretrained model as a feature extractor
        pretrained_model = InceptionV3(
            input_shape= input_shape,
            include_top=False,
            weights='imagenet',  # Use pre-trained weights
        )
        
        # Exclude the last two dense layers
        output_layer = pretrained_model.get_layer('conv_7b_ac').output
        feature_extractor = Model(inputs = pretrained_model.input, outputs = output_layer)
        return feature_extractor

    elif model_name == "MobileNetV2":
        # use pretrained model as a feature extractor
        pretrained_model = MobileNetV2(
            input_shape= input_shape,
            include_top=False,
            weights='imagenet',  # Use pre-trained weights
        )
        
        # Exclude the last two dense layers
        output_layer = pretrained_model.get_layer('out_relu').output  # 7x7x1280
        feature_extractor = Model(inputs = pretrained_model.input, outputs = output_layer)
        return feature_extractor

    else:
        print("Provided model name is invalid")
        return None


def convert_to_one_hot(labels, num_classes=None):
    """
    Convert integer labels to one-hot encoded labels.

    Parameters:
    - labels: Array-like. Integer labels.
    - num_classes: int, optional. The total number of classes. If not provided, it will be inferred from the maximum label value.

    Returns:
    - One-hot encoded labels as a numpy array.
    """
    return tf.expand_dims(to_categorical(labels, num_classes=num_classes), axis = 0)

######################################################################################################################################################
feature_extractor = FeatureExtractor(pModelName, pInputShape)
feature_extractor.summary()

# Freeze the pre-trained layers
for layer in feature_extractor.layers:
    layer.trainable = True


# check whether we GPU is available or not
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# define number of classes and files per class
NUM_CLASSES = len(os.listdir(train_data_path))
print("Number of classes: ", NUM_CLASSES)


# Get total number of video's across train and test
mainPath = pathlib.Path(main_path)
video_count_train = len(list(mainPath.glob('train/*/*.mp4')))
video_count_test = len(list(mainPath.glob('test/*/*.mp4')))
video_total = video_count_train + video_count_test
print(f"Total train videos: {video_count_train}")
print(f"Total test videos: {video_count_test}")
print(f"Total videos: {video_total}")
######################################################################################################################################################



######################################################################################################################################################
# custom frame selection class, to select the best frames from the given video
from frame_generator import FrameGenerator

subset_paths = {}
subset_paths['train'] = PosixPath(os.path.join(main_path, 'train'))
subset_paths['test'] = PosixPath(os.path.join(main_path, 'test'))

fg = FrameGenerator(path = subset_paths['train'], n_frames = pNumFrames, output_size = pInputShape[:2],
                    frame_step = 5, selection_strategy = pStrategyForFrame, training = True)

frames, label = next(fg())

print(f"Shape: {frames.shape}")
print(f"Label: {label}")
######################################################################################################################################################




######################################################################################################################################################
# Create the training set
output_signature = (tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32),
                    tf.TensorSpec(shape = (), dtype = tf.int16))
train_ds = tf.data.Dataset.from_generator(FrameGenerator(path = subset_paths['train'], n_frames = pNumFrames, output_size = pInputShape[:2], frame_step = 5, selection_strategy = pStrategyForFrame, training = True),
                                         output_signature = output_signature)


for frames, labels in train_ds.take(10):
    print(labels)

# Create the validation set
val_ds = tf.data.Dataset.from_generator(FrameGenerator(path = subset_paths['test'], n_frames = pNumFrames, output_size = pInputShape[:2], frame_step = 5, selection_strategy = pStrategyForFrame, training = False),
                                        output_signature = output_signature)

# Print the shapes of the data
train_frames, train_labels = next(iter(train_ds))
print(f'Shape of training set of frames: {train_frames.shape}')
print(f'Shape of training labels: {train_labels.shape}')

val_frames, val_labels = next(iter(val_ds))
print(f'Shape of validation set of frames: {val_frames.shape}')
print(f'Shape of validation labels: {val_labels.shape}')
######################################################################################################################################################


AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(10).prefetch(buffer_size = AUTOTUNE)
val_ds = val_ds.cache().shuffle(10).prefetch(buffer_size = AUTOTUNE)

train_ds = train_ds.batch(pBatchSize)
val_ds = val_ds.batch(pBatchSize)

train_frames, train_labels = next(iter(train_ds))
print(f'Shape of training set of frames: {train_frames.shape}')
print(f'Shape of training labels: {train_labels.shape}')

val_frames, val_labels = next(iter(val_ds))
print(f'Shape of validation set of frames: {val_frames.shape}')
print(f'Shape of validation labels: {val_labels.shape}')



def weighted_categorical_crossentropy(y_true, y_pred, class_weights):
    # Calculate categorical crossentropy
    cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

    # print(cce)
    # Apply class weights
    # weighted_cce = tf.reduce_mean(tf.multiply(cce, class_weights))

    return cce  #weighted_cce


def get_the_class_weights():
    class_weights = []
    for directory in os.listdir(train_data_path):
        print("weights: class", directory)
        class_weights.append(len(os.listdir(os.path.join(train_data_path, directory))))

    total_count  = sum(class_weights)
    class_weights = (total_count - np.array(class_weights))/total_count
    return tf.constant(class_weights)


class_weights = get_the_class_weights()
print(class_weights)
######################################################################################################################################################


METRICS = [
    #   keras.metrics.BinaryCrossentropy(name='cross entropy'),  # same as model's loss
    #   keras.metrics.MeanSquaredError(name='Brier score'),
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]


from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Flatten 
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger


input_shape = (pBatchSize, 30, 224, 224, 3)


model = tf.keras.Sequential([
    Conv3D(16, (3, 3, 3), input_shape = input_shape[1:], activation='relu'),
    MaxPooling3D((2, 2, 2)),
    Conv3D(32, (3, 3, 3), activation='relu'),
    MaxPooling3D((2, 2, 2)),
    Conv3D(64, (3, 3, 3), activation='relu'),
    MaxPooling3D((2, 2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')  # Two classes for anomaly detection
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=["accuracy"])

print(model.summary())

# Define callbacks
checkpoint_callback = ModelCheckpoint("model_epoch_{epoch:02d}.h5", save_freq='epoch')  # Save model at the end of each epoch
csv_logger = CSVLogger("training_history.csv", append=True)  # Save training history to CSV file

model.fit(train_ds, 
          epochs = 10,
          validation_data = val_ds,
          batch_size = 2,
          callbacks=[checkpoint_callback, csv_logger])
        #   callbacks = tf.keras.callbacks.EarlyStopping(patience = 2, monitor = 'val_loss'))
# # Print the model summary
# model.summary()