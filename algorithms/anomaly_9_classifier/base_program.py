import tqdm
import random
import pathlib
import itertools
import collections

import os
import cv2
import numpy as np
# import remotezip as rz

import tensorflow as tf

# Some modules to display an animation using imageio.
import imageio
from IPython import display
from urllib import request
import shutil
# from tensorflow_docs.vis import embed


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


NUM_CLASSES = 2
FILES_PER_CLASS = 1000

main_path = "./datasets/base_dataset"
# files_for_class = get_files_per_class(main_path)
# classes = list(files_for_class.keys())

classes = os.listdir(os.path.join(main_path, 'train'))


print('Num classes:', len(classes))
# print('Num videos for class[0]:', len(files_for_class[classes[0]]))



## Create a new function called select_subset_of_classes that selects a subset of the classes present within the dataset and a particular number of files per class:


# def select_subset_of_classes(files_for_class, classes, files_per_class):
#     """ Create a dictionary with the class name and a subset of the files in that class.

#     Args:
#       files_for_class: Dictionary of class names (key) and files (values).
#       classes: List of classes.
#       files_per_class: Number of files per class of interest.

#     Returns:
#       Dictionary with class as key and list of specified number of video files in that class.
#     """
#     files_subset = dict()

#     for class_name in classes:
#         class_files = files_for_class[class_name]
#         files_subset[class_name] = class_files[:files_per_class]

#     return files_subset



# files_subset = select_subset_of_classes(files_for_class, classes[:NUM_CLASSES], FILES_PER_CLASS)
# print(list(files_subset.keys()))



### Define helper functions that split the videos into training, validation, and test sets.

# def split_train_test_val(classes, files_for_class, main_path, splits):
#     """ split the data into various parts, such as training, validation, and test.

#     Args:
#       zip_url: A URL with a ZIP file with the data.
#       num_classes: Number of labels.
#       splits: Dictionary specifying the training, validation, test, etc. (key) division of data 
#               (value is number of files per split).
#       download_dir: Directory to download data to.

#     Return:
#       Mapping of the directories containing the subsections of data.
#     """

#     for cls in classes:
#         random.shuffle(files_for_class[cls])

#     # Only use the number of classes you want in the dictionary
#     files_for_class = {x: files_for_class[x] for x in classes}

#     dirs = {'train': [], 'test': [], 'val': []}
    
    # select_idx = 0
    # for split_name, split_count in splits.items():
    #     dst_path = os.path.join(main_path, split_name)
    #     print(dst_path)
    #     if not os.path.exists(dst_path):
    #         os.makedirs(dst_path)
    #         print(f"Folder created.")
    #     else:
    #         print(f"Folder already exists.")
    
    #     for cls in classes:
    #         destination_path = os.path.join(dst_path, cls)
    #         print(destination_path)
    #         if not os.path.exists(destination_path):
    #             os.makedirs(destination_path)
    #             print(f"Folder created.")
    #         else:
    #             print(f"Folder already exists.")

    #         for idx in range(split_count):
    #             current_idx = select_idx + idx
    #             file_name = files_for_class[cls][current_idx].split('/')[-1]
#                 shutil.move(files_for_class[cls][current_idx], os.path.join(destination_path, file_name))
#                 dirs[split_name].append(os.path.join(destination_path, file_name))
#         select_idx += split_count

#     return dirs

# main_path = "./datasets/base_dataset"
# subset_paths = split_train_test_val(classes, files_for_class, main_path, splits = {"train": 700, "val": 150, "test": 150})


# custom_subset_paths = subset_paths.copy()


mainPath = pathlib.Path(main_path)
video_count_train = len(list(mainPath.glob('train/*/*.mp4')))
video_count_val = len(list(mainPath.glob('val/*/*.mp4')))
video_count_test = len(list(mainPath.glob('test/*/*.mp4')))
video_total = video_count_train + video_count_val + video_count_test
print(f"Total videos: {video_total}")


def format_frames(frame, output_size):
    """
    Pad and resize an image from a video.

    Args:
      frame: Image that needs to resized and padded. 
      output_size: Pixel size of the output frame image.

    Return:
      Formatted frame with padding of specified output size.
    """
    try:
        frame = tf.image.convert_image_dtype(frame, tf.float32)
        frame = tf.image.resize_with_pad(frame, *output_size)
        return frame
        
    except tf.errors.InvalidArgumentError as e:
        # Handle the specific exception raised when the conversion fails
        print(f"Error converting image data type: {e}")
        # You may want to log the error, raise a custom exception, or take other actions
    except Exception as e:
        # Handle other exceptions that might occur
        print(f"An unexpected error occurred: {e}")
        # You can customize this block based on your specific needs


def frames_from_video_file(video_path, n_frames, output_size = (224,224), frame_step = 5):
    """
    Creates frames from each video file present for each category.

    Args:
      video_path: File path to the video.
      n_frames: Number of frames to be created per video file.
      output_size: Pixel size of the output frame image.

    Return:
      An NumPy array of frames in the shape of (n_frames, height, width, channels).
    """
    # Read each video frame by frame
    result = []
    src = cv2.VideoCapture(str(video_path))  

    video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

    need_length = 1 + (n_frames - 1) * frame_step

    if need_length > video_length:
        start = 0
    else:
        max_start = video_length - need_length
        start = random.randint(0, max_start + 1)

    src.set(cv2.CAP_PROP_POS_FRAMES, start)
    # ret is a boolean indicating whether read was successful, frame is the image itself
    ret, frame = src.read()
    result.append(format_frames(frame, output_size))

    for _ in range(n_frames - 1):
        for _ in range(frame_step):
            ret, frame = src.read()
        if ret:
            frame = format_frames(frame, output_size)
            result.append(frame)
        else:
            result.append(np.zeros_like(result[0]))
    src.release()
    result = np.array(result)[..., [2, 1, 0]]

    return result


# video_path = subset_paths['train'][0]
# sample_video = frames_from_video_file(video_path, n_frames = 5)
# print(sample_video.shape)


class FrameGenerator:
    def __init__(self, path, n_frames, training = False):
        """ Returns a set of frames with their associated label. 

          Args:
            path: Video file paths.
            n_frames: Number of frames. 
            training: Boolean to determine if training dataset is being created.
        """
        print("path", path)
        self.path = path
        self.n_frames = n_frames
        self.training = training
        self.class_names = sorted(set(p.name for p in self.path.iterdir() if p.is_dir()))
        self.class_ids_for_name = dict((name, idx) for idx, name in enumerate(self.class_names))

    def get_files_and_class_names(self):
        video_paths = list(self.path.glob('*/*.mp4'))
        classes = [p.parent.name for p in video_paths] 
        return video_paths, classes

    def __call__(self):
        video_paths, classes = self.get_files_and_class_names()

        pairs = list(zip(video_paths, classes))

        if self.training:
            random.shuffle(pairs)

        for path, name in pairs:
            video_frames = frames_from_video_file(path, self.n_frames) 
            label = self.class_ids_for_name[name] # Encode labels
            yield video_frames, label



from pathlib import PosixPath

subset_paths = {}
main_path = "./datasets/base_dataset"
subset_paths['train'] = PosixPath(os.path.join(main_path, 'train'))
subset_paths['test'] = PosixPath(os.path.join(main_path, 'test'))
subset_paths['val'] = PosixPath(os.path.join(main_path, 'val'))

fg = FrameGenerator(subset_paths['train'], 5, training=True)

frames, label = next(fg())

print(f"Shape: {frames.shape}")
print(f"Label: {label}")



# Create the training set
output_signature = (tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32),
                    tf.TensorSpec(shape = (), dtype = tf.int16))
train_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['train'], 30, training=True),
                                          output_signature = output_signature)


for frames, labels in train_ds.take(10):
    print(labels)


# Create the validation set
val_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['val'], 30),
                                        output_signature = output_signature)




# Print the shapes of the data
train_frames, train_labels = next(iter(train_ds))
print(f'Shape of training set of frames: {train_frames.shape}')
print(f'Shape of training labels: {train_labels.shape}')

val_frames, val_labels = next(iter(val_ds))
print(f'Shape of validation set of frames: {val_frames.shape}')
print(f'Shape of validation labels: {val_labels.shape}')




AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(10).prefetch(buffer_size = AUTOTUNE)
val_ds = val_ds.cache().shuffle(10).prefetch(buffer_size = AUTOTUNE)



train_ds = train_ds.batch(2)
val_ds = val_ds.batch(2)

train_frames, train_labels = next(iter(train_ds))
print(f'Shape of training set of frames: {train_frames.shape}')
print(f'Shape of training labels: {train_labels.shape}')

val_frames, val_labels = next(iter(val_ds))
print(f'Shape of validation set of frames: {val_frames.shape}')
print(f'Shape of validation labels: {val_labels.shape}')



net = tf.keras.applications.EfficientNetB0(include_top = False, input_shape=(224, 224, 3))
net.trainable = False

import tensorflow as tf



# model = tf.keras.Sequential([
#     # tf.keras.layers.Rescaling(scale=255),
#     tf.keras.layers.InputLayer(input_shape=(10, 224, 224, 3)),
#     tf.keras.layers.TimeDistributed(net),
#     tf.keras.layers.Dense(2),
#     tf.keras.layers.GlobalAveragePooling3D()
# ])

# model.compile(optimizer = 'adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
#               metrics=['accuracy'])

# print("summary ", model.summary())


import tensorflow as tf
from tensorflow.keras.layers import Input, TimeDistributed, GlobalAveragePooling3D, Dense
from tensorflow.keras.applications import EfficientNetB0

# Define the input shape
input_shape = (2, 30, 224, 224, 3)  # 60 frames, each of shape (224, 224, 3)

# Create an EfficientNetB0 model
# effnet_b0 = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling=None)

# Set the EfficientNetB0 model to be non-trainable
# effnet_b0.trainable = False

tf.config.run_functions_eagerly(True)
# Create the main model
# model = tf.keras.Sequential([
#     Input(shape=input_shape),
#     TimeDistributed(effnet_b0),
#     GlobalAveragePooling3D(),
#     Dense(2, activation='softmax')  # Two classes for anomaly detection
# ])

from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Flatten 
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

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
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

# Define callbacks
checkpoint_callback = ModelCheckpoint("model_epoch_{epoch:02d}.h5", save_freq='epoch')  # Save model at the end of each epoch
csv_logger = CSVLogger("training_history.csv", append=True)  # Save training history to CSV file

model.fit(train_ds, 
          epochs = 10,
          validation_data = val_ds,
          batch_size=2,
          callbacks=[checkpoint_callback, csv_logger])
        #   callbacks = tf.keras.callbacks.EarlyStopping(patience = 2, monitor = 'val_loss'))
# # Print the model summary
# model.summary()


