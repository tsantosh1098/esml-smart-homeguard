src_path = "/s/chopin/l/grad/tskumar/Documents/ESML/datasets/dataset2"
dst_path = "/s/chopin/l/grad/tskumar/Documents/ESML/datasets/dataset3"



import os
import cv2
import shutil
import random


train_per_class = 500
val_per_class   = 150

for directory in os.listdir(src_path):
    print(directory)
    src_folder_path = os.path.join(src_path, directory)
    dst_folder_path = os.path.join(dst_path, directory)

    # If not, create the folder
    if not os.path.exists(dst_folder_path):
        os.makedirs(dst_folder_path)

    num_of_files = train_per_class if "train" == directory else val_per_class
    print("num_of_files ", num_of_files)

    for sub_directory in os.listdir(src_folder_path):
        print(sub_directory)
        src_folder_path = os.path.join(src_path, directory, sub_directory)
        dst_folder_path = os.path.join(dst_path, directory, sub_directory)

        # If not, create the folder
        if not os.path.exists(dst_folder_path):
            os.makedirs(dst_folder_path)

        list_files = os.listdir(src_folder_path)
        random.shuffle(list_files)

        if num_of_files < len(list_files):
            list_files = list_files[:num_of_files]


        for file_name in list_files:
            print(file_name)
            srcpath = os.path.join(src_path, directory, sub_directory, file_name)
            dstpath = os.path.join(dst_path, directory, sub_directory)

            # If not, create the folder
            if not os.path.exists(dstpath):
                os.makedirs(dstpath)

            shutil.copy(srcpath, dstpath)