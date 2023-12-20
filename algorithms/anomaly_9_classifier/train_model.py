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
from tensorflow.keras.applications import VGG16, InceptionResNetV2, MobileNetV2, ResNet50
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.utils import to_categorical
import seaborn as sns

# InceptionResNetV2
# # Input shape to be used by the model
# pInputShape = (229, 229, 3)
# pNumFrames  = 30
# pModelName  = "InceptionResNetV2"
# pStrategyForFrame = "smart_select"
# pBatchSize  = 2
# pFeatureExtractorOutShape = 1536


# # MobileNetV2
# # Input shape to be used by the model
# pInputShape = (224, 224, 3)
# pNumFrames  = 30
# pModelName  = "MobileNetV2"
# pStrategyForFrame = "smart_select"
# pBatchSize  = 2
# pFeatureExtractorOutShape = 1280


# ResNet50
# # Input shape to be used by the model
pInputShape = (224, 224, 3)
pNumFrames  = 30
pModelName  = "ResNet50"
pStrategyForFrame = "smart_select"
pBatchSize  = 8
pFeatureExtractorOutShape = 2048


main_path = "/s/chopin/l/grad/tskumar/Documents/ESML/datasets/dataset1"
train_data_path = "/s/chopin/l/grad/tskumar/Documents/ESML/datasets/dataset1/train"
test_data_path = "/s/chopin/l/grad/tskumar/Documents/ESML/datasets/dataset1/test"


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

from frame_generator import FeatureExtractor

feature_extractor = FeatureExtractor(pModelName, pInputShape)
# feature_extractor.summary()

# Freeze the pre-trained layers
for layer in feature_extractor.layers:
    layer.trainable = False


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

fg = FrameGenerator(feature_extractor, path = subset_paths['train'], n_frames = pNumFrames, output_size = pInputShape[:2],
                    frame_step = 5, selection_strategy = pStrategyForFrame, training = True)

frames, label = next(fg())

print(f"Shape: {frames.shape}")
print(f"Label: {label}")
######################################################################################################################################################




######################################################################################################################################################
# Create the training set
output_signature = (tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32),
                    tf.TensorSpec(shape = (), dtype = tf.int16))
train_ds = tf.data.Dataset.from_generator(FrameGenerator(feature_extractor, path = subset_paths['train'], n_frames = pNumFrames, output_size = pInputShape[:2], frame_step = 5, selection_strategy = pStrategyForFrame, training = True),
                                         output_signature = output_signature)


for frames, labels in train_ds.take(10):
    print(frames.shape, labels)

# Create the validation set
val_ds = tf.data.Dataset.from_generator(FrameGenerator(feature_extractor, path = subset_paths['test'], n_frames = pNumFrames, output_size = pInputShape[:2], frame_step = 5, selection_strategy = pStrategyForFrame, training = False),
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
train_ds = train_ds.shuffle(10).prefetch(buffer_size = AUTOTUNE)
val_ds = val_ds.shuffle(10).prefetch(buffer_size = AUTOTUNE)

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
      # keras.metrics.TruePositives(name='tp'),
      # keras.metrics.FalsePositives(name='fp'),
      # keras.metrics.TrueNegatives(name='tn'),
      # keras.metrics.FalseNegatives(name='fn'), 
    #   keras.metrics.CategoricalAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]


from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Flatten 
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from keras import layers
import einops
from keras.models import Model



class Conv2Plus1D(keras.layers.Layer):
  def __init__(self, filters, kernel_size, padding, **kwargs):
    """
      A sequence of convolutional layers that first apply the convolution operation over the
      spatial dimensions, and then the temporal dimension. 
    """
    super(Conv2Plus1D, self).__init__(**kwargs)
    self.filters = filters
    self.kernel_size = kernel_size
    self.padding = padding
    self.seq = keras.Sequential([  
        # Spatial decomposition
        layers.Conv3D(filters=self.filters,
                      kernel_size=(1, self.kernel_size[1], self.kernel_size[2]),
                      padding=self.padding),
        # Temporal decomposition
        layers.Conv3D(filters=self.filters, 
                      kernel_size=(self.kernel_size[0], 1, 1),
                      padding=self.padding)
        ])

  def call(self, x):
    return self.seq(x)
  
  def get_config(self):
    config = super().get_config()
    config.update({
        "filters": self.filters,
        "kernel_size": self.kernel_size,
        "padding": self.padding,
    })
    return config


class ResidualMain(keras.layers.Layer):
  """
    Residual block of the model with convolution, layer normalization, and the
    activation function, ReLU.
  """
  def __init__(self, filters, kernel_size, **kwargs):
    super(ResidualMain, self).__init__(**kwargs)
    self.filters = filters
    self.kernel_size = kernel_size
    self.seq = keras.Sequential([
        Conv2Plus1D(filters=filters,
                    kernel_size=kernel_size,
                    padding='same'),
        layers.LayerNormalization(),
        layers.ReLU(),
        Conv2Plus1D(filters=filters, 
                    kernel_size=kernel_size,
                    padding='same'),
        layers.LayerNormalization()
    ])

  def call(self, x):
    return self.seq(x)

  def get_config(self):
    config = super().get_config()
    config.update({
        "filters": self.filters,
        "kernel_size": self.kernel_size,
    })
    return config

class Project(keras.layers.Layer):
  """
    Project certain dimensions of the tensor as the data is passed through different 
    sized filters and downsampled. 
  """
  def __init__(self, units, **kwargs):
    super(Project, self).__init__(**kwargs)
    self.units = units
    self.seq = keras.Sequential([
        layers.Dense(self.units),
        layers.LayerNormalization()
    ])

  def call(self, x):
    return self.seq(x)

  def get_config(self):
    config = super().get_config()
    config.update({
        "units": self.units,
    })
    return config

def add_residual_block(input, filters, kernel_size):
  """
    Add residual blocks to the model. If the last dimensions of the input data
    and filter size does not match, project it such that last dimension matches.
  """
  out = ResidualMain(filters, 
                     kernel_size)(input)

  res = input
  # Using the Keras functional APIs, project the last dimension of the tensor to
  # match the new filter size
  if out.shape[-1] != input.shape[-1]:
    res = Project(out.shape[-1])(res)

  return layers.add([res, out])


class ResizeVideo(keras.layers.Layer):
  def __init__(self, height, width, **kwargs):
    super(ResizeVideo, self).__init__(**kwargs)
    self.height = height
    self.width = width

  def call(self, video):
    """
      Use the einops library to resize the tensor.  

      Args:
        video: Tensor representation of the video, in the form of a set of frames.

      Return:
        A downsampled size of the video according to the new height and width it should be resized to.
    """
    # b stands for batch size, t stands for time, h stands for height, 
    # w stands for width, and c stands for the number of channels.
    old_shape = einops.parse_shape(video, 'b t h w c')
    images = einops.rearrange(video, 'b t h w c -> (b t) h w c')
    images = keras.layers.Resizing(self.height, self.width)(images)
    videos = einops.rearrange(
        images, '(b t) h w c -> b t h w c',
        t = old_shape['t'])
    return videos

  def get_config(self):
    config = super().get_config()
    config.update({
        "height": self.height,
        "width": self.width,
    })
    return config


input_shape = (pBatchSize, 30, 224, 224, 3)
HEIGHT = input_shape[2]
WIDTH = input_shape[3]


input = layers.Input(shape=(input_shape[1:]))
x = input

x = Conv2Plus1D(filters=16, kernel_size=(3, 7, 7), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = ResizeVideo(HEIGHT // 2, WIDTH // 2)(x)

# Block 1
x = add_residual_block(x, 16, (3, 3, 3))
x = ResizeVideo(HEIGHT // 4, WIDTH // 4)(x)

# Block 2
x = add_residual_block(x, 32, (3, 3, 3))
x = ResizeVideo(HEIGHT // 8, WIDTH // 8)(x)

# Block 3
x = add_residual_block(x, 64, (3, 3, 3))
x = ResizeVideo(HEIGHT // 16, WIDTH // 16)(x)

# Block 4
x = add_residual_block(x, 128, (3, 3, 3))

x = layers.GlobalAveragePooling3D()(x)
x = layers.Flatten()(x)
x = layers.Dense(NUM_CLASSES)(x)

model = keras.Model(input, x)

cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)


from keras.models import load_model


# Compile the model
model.compile(loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              optimizer = keras.optimizers.Adam(learning_rate = 0.0001), metrics=['accuracy'])

# print(model.summary())


# Define callbacks
checkpoint_callback = ModelCheckpoint("model_epoch_{epoch:02d}.h5", save_freq='epoch')  # Save model at the end of each epoch
csv_logger = CSVLogger("training_history.csv", append=True)  # Save training history to CSV file


# # Load the model
# model = load_model('path/to/your/model.h5', custom_objects={'YourCustomLayer': YourCustomLayer})
model = load_model('./model_epoch_249.h5', custom_objects = {'Conv2Plus1D' : Conv2Plus1D, 'Project' : Project, 'ResizeVideo' : ResizeVideo, 'ResidualMain' : ResidualMain})
print("model ", model.summary())

# history = model.fit(train_ds, 
#           epochs = 250,
#           validation_data = val_ds,
#           batch_size = pBatchSize,
#           callbacks=[checkpoint_callback, csv_logger])
        #   callbacks = tf.keras.callbacks.EarlyStopping(patience = 2, monitor = 'val_loss'))
# # Print the model summary
# model.summary()


import matplotlib.pyplot as plt
import numpy as np

# def plot_history(history, save_path=None):
#     """
#     Plotting training and validation learning curves.

#     Args:
#       history: model history with all the metric measures
#       save_path: path to save the plot as a PNG file (optional)
#     """
#     fig, (ax1, ax2) = plt.subplots(2)

#     fig.set_size_inches(18.5, 10.5)

#     # Plot loss
#     ax1.set_title('Loss')
#     ax1.plot(history.history['loss'], label='train')
#     ax1.plot(history.history['val_loss'], label='test')
#     ax1.set_ylabel('Loss')

#     # Determine upper bound of y-axis
#     max_loss = max(history.history['loss'] + history.history['val_loss'])

#     ax1.set_ylim([0, np.ceil(max_loss)])
#     ax1.set_xlabel('Epoch')
#     ax1.legend(['Train', 'Validation'])

#     # Plot accuracy
#     ax2.set_title('Accuracy')
#     ax2.plot(history.history['accuracy'], label='train')
#     ax2.plot(history.history['val_accuracy'], label='test')
#     ax2.set_ylabel('Accuracy')
#     ax2.set_ylim([0, 1])
#     ax2.set_xlabel('Epoch')
#     ax2.legend(['Train', 'Validation'])

#     if save_path:
#         plt.savefig(save_path)
#     else:
#         plt.show()

# # Replace 'path/to/save_plots.png' with the desired file path to save the plots
# plot_history(history, save_path='./save_plots.png')


print("evalute_model")
# model.evaluate(val_ds, return_dict=True)


def get_actual_predicted_labels(dataset): 
  """
    Create a list of actual ground truth values and the predictions from the model.

    Args:
      dataset: An iterable data structure, such as a TensorFlow Dataset, with features and labels.

    Return:
      Ground truth and predicted values for a particular dataset.
  """
  actual = [labels for _, labels in dataset.unbatch()]
  predicted = model.predict(dataset)

  actual = tf.stack(actual, axis=0)
  predicted = tf.concat(predicted, axis=0)
  predicted = tf.argmax(predicted, axis=1)

  return actual, predicted


def plot_confusion_matrix(actual, predicted, labels, ds_type):
  cm = tf.math.confusion_matrix(actual, predicted)
  ax = sns.heatmap(cm, annot=True, fmt='g')
  sns.set(rc={'figure.figsize':(12, 12)})
  sns.set(font_scale=1.4)
  ax.set_title('Confusion matrix of action recognition for ' + ds_type)
  ax.set_xlabel('Predicted Action')
  ax.set_ylabel('Actual Action')
  plt.xticks(rotation=90)
  plt.yticks(rotation=0)
  ax.xaxis.set_ticklabels(labels)
  ax.yaxis.set_ticklabels(labels)
  plt.savefig(f"./{ds_type}_cf.png")


labels = list(fg.class_ids_for_name.keys())


# print("training")
# actual, predicted = get_actual_predicted_labels(train_ds)
# plot_confusion_matrix(actual, predicted, labels, 'training')

print("testing")

actual, predicted = get_actual_predicted_labels(val_ds)
plot_confusion_matrix(actual, predicted, labels, 'test')


def calculate_classification_metrics(y_actual, y_pred, labels):
  """
    Calculate the precision and recall of a classification model using the ground truth and
    predicted values. 

    Args:
      y_actual: Ground truth labels.
      y_pred: Predicted labels.
      labels: List of classification labels.

    Return:
      Precision and recall measures.
  """
  cm = tf.math.confusion_matrix(y_actual, y_pred)
  tp = np.diag(cm) # Diagonal represents true positives
  precision = dict()
  recall = dict()
  for i in range(len(labels)):
    col = cm[:, i]
    fp = np.sum(col) - tp[i] # Sum of column minus true positive is false negative

    row = cm[i, :]
    fn = np.sum(row) - tp[i] # Sum of row minus true positive, is false negative

    precision[labels[i]] = tp[i] / (tp[i] + fp) # Precision 

    recall[labels[i]] = tp[i] / (tp[i] + fn) # Recall

  return precision, recall


precision, recall = calculate_classification_metrics(actual, predicted, labels) # Test dataset





