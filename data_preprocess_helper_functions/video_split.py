import os
import cv2
import shutil



# src_path = "/home/armnn/Music/ESML/project/datasets/SPHAR_Dataset/videos/normal_data"
# dst_path = "/home/armnn/Music/ESML/project/datasets/SPHAR_Dataset/processed_videos/normal_data"

src_path = "/home/armnn/Music/ESML/project/datasets/SPHAR_Dataset/videos/anomaly_data"
dst_path = "/home/armnn/Music/ESML/project/datasets/SPHAR_Dataset/processed_videos/anomaly_data"


list_directory = os.listdir(src_path)


def frame_count(video_path):
    # Read each video frame by frame
    src = cv2.VideoCapture(str(video_path))
    return src.get(cv2.CAP_PROP_FRAME_COUNT)



def frames_to_video(frames, output_path, fps):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        video_writer.write(frame)

    video_writer.release()



def frames_from_video_file(source_path, destination_path):
        # Read each video frame by frame
        result = []
        src = cv2.VideoCapture(str(source_path))

        video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)
        print("video_length ", video_length)

        num_splits = (video_length // (180*30))
        print("num_splits", num_splits)

        src.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # ret is a boolean indicating whether read was successful, frame is the image itself
        results = []
        split_count = 1
        for index in range(1, int(video_length)):
            ret, frame = src.read()
            results.append(frame)
            if index % (180*30) == 0:
                output_path = destination_path[:-4] + "_" + str(split_count) + "_.mp4"
                fps = 30
                frames_to_video(results, output_path, fps)
                results = []
                split_count += 1

        if len(results) != 0:
            output_path = destination_path[:-4] + "_" + str(split_count) + "_.mp4"
            fps = 30
            frames_to_video(results, output_path, fps)
            results = []

        src.release()



count = 0
stats_of_video_data = {}
for directory in list_directory:
    directory_path = os.path.join(src_path, directory)
    stats_of_video_data[directory] = []
    for video_file in os.listdir(directory_path):
        video_path = os.path.join(directory_path, video_file)
        count += 1
        print("count ", count)
        # duration = get_video_duration(video_path)
        video_length = frame_count(video_path)
        print(video_path)
        source_path = video_path
        destination_path = os.path.join(dst_path, directory, video_file)

        if video_length > 180*30:
            frames_from_video_file(source_path, destination_path)
            os.remove(source_path)        
print(count)


