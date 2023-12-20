# import required packages
import tensorflow as tf
import cv2
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import PosixPath

import os
import cv2
import shutil
import json


# Function to display extracted files
def display_extracted_frames(num_rows, num_cols, list_frames):

    """
    Display function to show extracted frames from the video

    Args:
      num_rows: Number of rows to construct the subplot.
      num_cols: Number of columns to construct the subplot.
      list_frames: List of frames to be displayed.

    Return:
      Display the subplot
    """
    # Create a Matplotlib figure and subplots
    _, axes = plt.subplots(num_rows, num_cols, figsize=(16, 16))  # Adjust figsize as needed

    # Flatten the 2D array of subplots to make it easier to iterate through
    axes = axes.flatten()

    # Iterate through the image paths and display each image on a subplot
    for count, frame in enumerate(list_frames):
        if count < num_rows * num_cols:
            # Display the image on the current subplot
            axes[count].imshow(frame)
            axes[count].set_title(f'Image {count + 1}')
            # Turn off axis labels
            axes[count].axis('off')

    # Adjust the layout for better visualization
    plt.tight_layout()

    # Show the subplots
    plt.show()


def frames_to_video(frames, output_path, fps):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        video_writer.write(frame)

    video_writer.release()


def nth_max_score(scores, n):
    # sort the list in descending order
    sorted_scores = sorted(scores, reverse=True)

    # Check if n is within the valid range
    if 1 <= n <= len(sorted_scores):
        return sorted_scores[n - 1]
    else:
        return f"Invalid value of n. It should be between 1 and {len(sorted_scores)}."
        
def select_frame_on_score_smart_approach(src, video_length, frame_step, n_frames):
    results = []
    scores = []
    previous_heatmap = None

    src.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, previous_frame = src.read()
    video_length = int(video_length)
    need_length = (frame_step * (n_frames + 1))
    if need_length <= video_length:
        loop_end, loop_incrementer = video_length, frame_step
    else:
        loop_end, loop_incrementer = video_length, 1
    
    # ret is a boolean indicating whether read was successful, frame is the image itself
    for _ in range(0, max(loop_end, n_frames), loop_incrementer):
        for _ in range(loop_incrementer):
            ret, frame = src.read()
        if not ret:
            frame = np.zeros_like(previous_frame)

        # Load two images
        image1 = previous_frame
        image2 = frame

        # print("image", image1.shape, image2.shape)
        
        # Convert images to grayscale
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)*255
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)*255

        # Compute absolute difference between the two images
        diff = cv2.absdiff(gray1, gray2)

        # Apply thresholding
        _, binary_mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        non_zero_pixels_count = np.sum(binary_mask)
        
        # Create a heatmap using a color scale
        # heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_HOT)
        
        # if previous_heatmap is None:
            # previous_heatmap = heatmap

        results.append(frame)
        scores.append(non_zero_pixels_count)
                
        # previous_heatmap = heatmap
        previous_frame   = frame

    src.release()

    threshold_score = nth_max_score(scores, n_frames)
    final_selected_frames = []
    for idx, score in enumerate(scores):
        if score >= threshold_score:
            final_selected_frames.append(results[idx])
    return final_selected_frames[:n_frames]
    


# Function that convert the received input video and output constant number of frames.
def frames_from_video_file(video_path):
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

    video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)
    print("video_length ", video_length)

    return select_frame_on_score_smart_approach(src, video_length, frame_step = 5, n_frames = 60)


# src_path = "/home/armnn/Music/ESML/project/datasets/SPHAR_Dataset/video"
# dst_path = "/home/armnn/Music/ESML/project/datasets/SPHAR_Dataset/pre_video"


src_path = "/home/armnn/Music/ESML/project/datasets/SPHAR_Dataset/v"
dst_path = "/home/armnn/Music/ESML/project/datasets/SPHAR_Dataset/pre_v"

count = 0
all_cnt = 0
dict_record = {}
for directory in os.listdir(src_path):
    print("directory ", directory)
    start_time = time.time()
    for video_file in os.listdir(os.path.join(src_path, directory)):
        src_file = os.path.join(src_path, directory, video_file)
        result = frames_from_video_file(src_file)
        print("frames count:", len(result))
        output_path = os.path.join(dst_path, directory, video_file)
        directory_path = os.path.join(dst_path, directory)
        # Check if the directory exists
        if not os.path.exists(directory_path):
            # Create the directory if it doesn't exist
            os.makedirs(directory_path)
            # print(f"Directory '{directory_path}' created.")
        else:
            pass
            # print(f"Directory '{directory_path}' already exists.")

        fps = 30
        frames_to_video(result, output_path, fps)
        count += 1
        all_cnt += 1
        print("count ", all_cnt)
    end_time = time.time()
    
    dict_record[directory] = ((end_time - start_time), count)
    count = 0


json_file_path = "./sumary_of_preprocess_and_store.json"
with open(json_file_path, 'w') as json_file:
    json.dump(dict_record, json_file, indent=4)
    
