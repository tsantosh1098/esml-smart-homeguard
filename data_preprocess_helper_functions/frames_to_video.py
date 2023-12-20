import os
import cv2
import shutil


def frames_to_video(frames, output_path, fps):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        video_writer.write(frame)

    video_writer.release()


# src_path = "/home/armnn/Music/ESML/project/datasets/Smart_City_CCTV_Violence_Detection_Dataset/SCVD/videos"
# dst_path = "/home/armnn/Music/ESML/project/datasets/Smart_City_CCTV_Violence_Detection_Dataset/SCVD/processed_video"

# src_path = "/home/armnn/Music/ESML/project/datasets/Large_Scale_Multi_Camera_Detection_Dataset"
# dst_path = "/home/armnn/Music/ESML/project/datasets/Large_Scale_Multi_Camera_Detection_Dataset1"


src_path = "/home/armnn/Music/ESML/project/datasets/human_entering_door_processed"
dst_path = "/home/armnn/Music/ESML/project/datasets/human_entering_door_processed111"

count = 0
for directory in os.listdir(src_path):
    for sub_directory in os.listdir(os.path.join(src_path, directory)):
        image_folder = os.path.join(src_path, directory, sub_directory)
        image_files = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
        # image_files = os.listdir()
        print("sub_directory ", sub_directory)
        image_files.sort()
        reuslts = []
        count += 1
        print("count ", count)
        for img_name in image_files:
            img_path = os.path.join(src_path, directory, sub_directory, img_name)
            reuslts.append(cv2.imread(img_path))
        
        output_path = os.path.join(dst_path, sub_directory + ".mp4")
        fps = 30
        frames_to_video(reuslts, output_path, fps)
