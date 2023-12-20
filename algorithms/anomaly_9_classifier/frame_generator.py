import tensorflow as tf
import cv2
import random
import numpy as np

from tensorflow.keras.applications import VGG16, InceptionResNetV2, MobileNetV2, ResNet50, InceptionV3
from keras.models import Model
from tensorflow.keras.utils import to_categorical

class FrameGenerator:
    def __init__(self, feature_extractor, path, n_frames, output_size, frame_step, selection_strategy, training = False):
        """ Returns a set of frames with their associated label. 

          Args:
            path: Video file paths.
            n_frames: Number of frames.
            selection_strategy: Provide which startegy to use for frame extraction
            we have support for 3 different stategies: 1. random, 2. dynamic and 3. smart_select
            training: Boolean to determine if training dataset is being created.
        """
        # print("path", path)
        self.path = path
        self.feature_extractor = feature_extractor
        self.n_frames = n_frames
        self.output_size = output_size
        self.frame_step = frame_step
        self.selection_strategy = selection_strategy
        self.training = training
        self.class_names = sorted(set(p.name for p in self.path.iterdir() if p.is_dir()))
        self.class_ids_for_name = dict((name, idx) for idx, name in enumerate(self.class_names))

        print("self.class_names ", self.class_names)
        print("self.class_ids_for_name ", self.class_ids_for_name)


    # Function that receive the video frame and output the padded and resized frame.
    def format_frames(self, frame):
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
            frame = tf.image.resize_with_pad(frame, *self.output_size)
            return frame
            
        except tf.errors.InvalidArgumentError as e:
            # Handle the specific exception raised when the conversion fails
            print(f"Error converting image data type: {e}")
        except Exception as e:
            # Handle other exceptions that might occur
            print(f"An unexpected error occurred: {e}")


    # Function that convert the received input video and output constant number of frames.
    def frames_from_video_file(self, video_path):
        """
        Creates frames from each video file present for each category.

        Args:
        video_path: File path to the video.
        selection_strategy: Provide which startegy to use for frame extraction
        n_frames: Number of frames to be created per video file.
        output_size: Pixel size of the output frame image.

        Return:
        An NumPy array of frames in the shape of (n_frames, height, width, channels).
        """
        # Read each video frame by frame
        result = []
        src = cv2.VideoCapture(str(video_path))

        src.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # ret is a boolean indicating whether read was successful, frame is the image itself
        for _ in range(self.n_frames):
            ret, frame = src.read()
            ret, frame = src.read()
            if ret:
                frame = self.format_frames(frame)
                result.append(frame)
            else:
                frame = np.zeros((256, 256, 3), dtype = np.uint8)
                frame = self.format_frames(frame)
                result.append(frame)
            src.release()

        return np.array(result)[..., [2, 1, 0]]

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
            video_frames = self.frames_from_video_file(video_path = path)
            # output = self.feature_extractor(video_frames)
            # output = tf.keras.layers.GlobalAveragePooling2D()(output)
            label = self.class_ids_for_name[name] # Encode labels
            yield video_frames, label


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

    elif model_name == "ResNet50":
        # use pretrained model as a feature extractor
        pretrained_model = ResNet50(
            input_shape= input_shape,
            include_top=False,
            weights='imagenet',  # Use pre-trained weights
        )
        
        # Exclude the last two dense layers
        output_layer = pretrained_model.get_layer('conv5_block3_out').output  # 7x7x2048
        feature_extractor = Model(inputs = pretrained_model.input, outputs = output_layer)
        return feature_extractor        
    else:
        print("Provided model name is invalid")
        return None
    
