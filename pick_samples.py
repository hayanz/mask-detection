import os
import random
import shutil

# the samples are from "UTKFace Large Scale Face Dataset"
# https://susanqq.github.io/UTKFace/
dir1 = "faces_dataset/part1/"  # directory of the pictures
dir2 = "faces_dataset/part2/"  # directory of the pictures
picked_dir = "faces_dataset/picked/"

pic_type = ".jpg"
count = 1  # number of samples to use
count_half = count / 2
start_idx = 16  # starting index for counting the samples


def move_files(target, start=0, end=count, add_num=0):
    for i in range(start + add_num, end + add_num):
        chosen = random.choice(os.listdir(target))
        chosen_path = "%s%s" % (target, chosen)
        shutil.move(chosen_path, picked_dir)  # move the file
        previous = "%s%s" % (picked_dir, chosen)
        new_name = "%s%s" % (picked_dir, str(i) + pic_type)
        os.rename(previous, new_name)  # rename for distinguish the files
        print("[%d] Successfully picked!" % i)  # for debugging


if __name__ == "__main__":
    move_files(dir1, end=count_half, add_num=start_idx)  # pick samples from 'dir1'
    move_files(dir2, start=count_half, add_num=start_idx)  # pick samples from 'dir2'
