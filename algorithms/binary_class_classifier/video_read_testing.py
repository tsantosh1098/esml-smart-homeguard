
import os
import cv2

src_path = "/s/chopin/l/grad/tskumar/Documents/ESML/datasets/dataset2/train/neutral/"

count = 0
for file_name in os.listdir(src_path):
    file_dir_name = file_name[:-4]

    srcpath = os.path.join(src_path, file_name)

    # Read each video frame by frame

    src = cv2.VideoCapture(str(srcpath))

    src.set(cv2.CAP_PROP_POS_FRAMES, 0)
    results = []
    for _ in range(60):
        ret, frame = src.read()
        if ret:
            results.append(frame)
    src.release()
    count += 1
    print("count", count, file_name, len(results))