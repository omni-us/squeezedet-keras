# Project: squeezeDetOnKeras
# Filename: train_val_split
# Author: Christopher Ehmann
# Date: 12.12.17
# Organisation: searchInk
# Email: christopher@searchink.com

import random
import numpy as np
import argparse




def train_eval_split(img_file =  "images.txt" ,
                    gt_file =  "labels.txt"
                    , percent_train = 80, percent_val = 20, percent_test = 0):
    """Given a two files containing the list of images and ground truth , create a single datasplit
    
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



    n_train = int(np.floor(len(img_names) * percent_train/100))

    n_val =  int(np.floor(len(img_names) * (percent_train + percent_val)/100))


    assert len(img_names) == len(gt_names)

    with open("img_train.txt", 'w') as img_train:
        img_train.write("\n". join(img_names[0:n_train]) )

    #img_train.close()

    with open("gt_train.txt", 'w') as gt_train:
        gt_train.write("\n". join(gt_names[0:n_train]) )

    #gt_train.close()

    with open("img_val.txt", 'w') as img_val:
        img_val.write("\n". join(img_names[n_train:n_val]) )

    #img_train.close()

    with open("gt_val.txt", 'w') as gt_val:
        gt_val.write("\n". join(gt_names[n_train:n_val]) )

    with open("img_test.txt", 'w') as img_test:
        img_test.write("\n".join(img_names[n_val:]))

    # img_train.close()

    with open("gt_test.txt", 'w') as gt_test:
        gt_test.write("\n".join(gt_names[n_val:]))



    print("Imgs and gts splitted")

            #gt_val.close()






if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Split images and gt in train, val and test set. ')
    parser.add_argument("--train",type=int, help="Percentage of train DEFAULT: 80 ")
    parser.add_argument("--val", type=int, help="Percentage of val DEFAULT: 20 ")
    parser.add_argument("--test",type=int, help="Percentage of test. DEFAULT: 0 ")
    parser.add_argument("--img", help="File with image names. DEFAULT: images.txt")
    parser.add_argument("--gt", help="File with gt names. DEFAULT: labels.txt")

    args = parser.parse_args()

    percent_train = 80
    percent_val = 20
    percent_test = 0


    if args.train is not None:
        percent_train = args.train

    if args.test is not None:
        percent_test= args.test

    if args.val is not None:
        percent_val = args.val




    img_file = "images.txt"
    gt_file = "labels.txt"


    if args.img is not None:
        img_file = args.img

    if args.gt is not None:
        gt_file = args.gt



    train_eval_split(img_file, gt_file, percent_train, percent_val, percent_test)


