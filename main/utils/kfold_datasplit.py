# Project: squeezeDetOnKeras
# Filename: train_val_split
# Author: Christopher Ehmann
# Date: 12.12.17
# Organisation: searchInk
# Email: christopher@searchink.com

import random
import numpy as np
import argparse
from sklearn.cross_validation import KFold
import os



def kfold_datasplit(img_file = "images.txt", gt_file="labels.txt"
                    , k = 5, percent_train = 80, percent_val = 20, percent_test = 0):
    """Given a two files containing the list of images and ground truth , create a k fold datasplit and save in txt
    
    Keyword Arguments:
        img_file {str} -- file name containing image paths (default: {"images.txt})
        gt_file {str} -- file name containing ground truth paths (default: {"labels.txt"})
        k {int} -- number of splits (default: {5})
        percent_train {int} -- Percent of training set (default: {80})
        percent_val {int} -- Percent of validation set(default: {20})
        percent_test {int} --  Percent of training set (default: {0})
    """



    if not (percent_train + percent_test +  percent_val) == 100:
        print("Percentages have to sum to 100")


    with open(img_file) as imgs:
        img_names = imgs.read().splitlines()
    imgs.close()
    with open(gt_file) as gts:
        gt_names = gts.read().splitlines()
    gts.close()


    shuffle = 1

    if shuffle:
        # permutate images
        shuffled = list(zip(img_names, gt_names))
        random.shuffle(shuffled)
        img_names, gt_names = zip(*shuffled)




    kf = KFold(len(img_names), n_folds=k)


    i = 0


    ensure_dir("./splits/")

    for train_idx, val_idx in kf:

        path = "./splits/split{}/".format(i)

        ensure_dir(path)

        with open( os.path.join(path, "img_train.txt".format(i)), 'w') as img_train:
            img_train.write("\n".join([ img_names[j] for j in train_idx]) )

        #img_train.close()

        with open(os.path.join(path, "gt_train.txt".format(i)), 'w') as gt_train:
            gt_train.write("\n".join([ gt_names[j] for j in train_idx]) )

        #gt_train.close()

        with open(os.path.join(path, "img_val.txt".format(i)), 'w') as img_val:
            img_val.write("\n".join([ img_names[j] for j in val_idx]) )

        #img_train.close()

        with open(os.path.join(path, "gt_val.txt".format(i)), 'w') as gt_val:
            gt_val.write("\n".join([ gt_names[j] for j in val_idx]) )

        i += 1


    print("Imgs and gts splitted")

            #gt_val.close()




def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    print(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Split images and gt in train, val and test set. ')
    parser.add_argument("--train",type=int, help="Percentage of train DEFAULT: 80 ")
    parser.add_argument("--val", type=int, help="Percentage of val DEFAULT: 20 ")
    parser.add_argument("--test", type=int, help="Percentage of test. DEFAULT: 0 ")
    parser.add_argument("--img", help="File with image names. DEFAULT: /home/cehmann/data/Dataset/Frankstahl/training/img.txt")
    parser.add_argument("--gt", help="File with gt names. DEFAULT: /home/cehmann/data/Dataset/Frankstahl/training/gt.txt")
    parser.add_argument("--k" , type=int, help="Number of folds. DEFAULT: 5")
    args = parser.parse_args()

    percent_train = 80
    percent_val = 20
    percent_test = 0
    k = 5

    if args.train is not None:
        percent_train = args.train

    if args.test is not None:
        percent_test= args.test

    if args.val is not None:
        percent_val = args.val



    if args.k is not None:
        k = args.k


    img_file = "/home/cehmann/data/Dataset/Frankstahl/training/img.txt"
    gt_file = "/home/cehmann/data/Dataset/Frankstahl/training/gt.txt"


    if args.img is not None:
        img_file = args.img

    if args.gt is not None:
        gt_file = args.gt


    kfold_datasplit(img_file, gt_file, k, percent_train, percent_val, percent_test)