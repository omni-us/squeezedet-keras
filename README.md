# SqueezeDet on Keras #

## _SqueezeDet:_ Unified, Small, Low Power Fully Convolutional Neural Networks for Real-Time Object Detection for Autonomous Driving
By Bichen Wu, Alvin Wan, Forrest Iandola, Peter H. Jin, Kurt Keutzer (UC Berkeley & DeepScale)

This repository contains a Keras implementation of SqueezeDet, a convolutional neural network based object detector described in this paper: https://arxiv.org/abs/1612.01051. If you find this work useful for your research, please consider citing:

    @inproceedings{squeezedet,
        Author = {Bichen Wu and Forrest Iandola and Peter H. Jin and Kurt Keutzer},
        Title = {SqueezeDet: Unified, Small, Low Power Fully Convolutional Neural Networks for Real-Time Object Detection for Autonomous Driving},
        Journal = {arXiv:1612.01051},
        Year = {2016}
    }

### Installation ###

Please have a look at our [Installation Guide](https://github.com/omni-us/squeezedet-keras/blob/master/Install.md)

### How do I run it? ###

I will show an example on the KITTI dataset. If you have any
doubts, most scripts run with the **-h** flag give you the 
arguments you can pass

* Download the KITTI training example from [here](http://www.cvlibs.net/download.php?file=data_object_image_2.zip) and [here](http://www.cvlibs.net/download.php?file=data_object_label_2.zip)

* Unzip them 


	`unzip data_object_image_2.zip`

	`unzip data_object_label_2.zip`


     You should get a folder called **training**.


* Inside the repository folder create a folder for the experiment. If you don't mind
	or dont want to type .. all the time you can do it in the **scripts** folder

	`cd path/to/squeezeDet`

	`mkdir experiments`

	`mkdir experiments/kitti`

	`cd experiments/kitti`

* SqueezeDet takes a list of images with full paths to the images and the same for labels. It's the same for training and evaluation. Create a list of full path names of images and labels:

	`find  /path/to/training/image_2/ -name "*png" | sort > images.txt`

	`find /path/to/training/label_2/ -name "*txt" | sort > labels.txt`

* Create a training test split


	`python ../../main/utils/train_val_split.py`

	You should get img_train.txt, gt_train.txt, img_val.txt gt_val.txt, img_test.txt, gt_test.txt . Testing set is empty
	by default.


* Create a config file

	`python ../../main/config/create_config.py`

	Depending on the GPU change the batch size inside **squeeze.config** and other parameters like learning rate.


* Run training, this starts with pre-trained weights from imagenet

	`python ../../scripts/train.py --init ../../main/model/imagenet.h5`

* In another shell, to run evaluation

	 - If you have no second GPU or none at all:

	   `python ../../scripts/eval.py --gpu ""`

	- Otherwise:
	 

	  `python ../../scripts/eval.py `

	  This will run evaluation in parallel on the second GPU.

* To run training on multiple GPUS:

	 `python ../../scripts/train.py --gpus 2 --init ../../main/model/imagenet.h5`

	 To run on the first 2 GPUS. Then you have to run evaluation on the third or CPU, if you have it. 


* **scripts/scheduler.py** allows you to run multiple trainings
after another. Check out the dummy **scripts/schedule.config** for an example. Run this with


	 `python ../../scripts/scheduler.py --schedule ../../scripts/schedule.config --train ../../scripts/train.py --eval ../../scripts/eval.py 
`



### Tensorboard visualization

For tensoboard visualization you can can run:


`tensorboard --logdir log`

Open in your brower  **localhost:6006** or the IP where you ran the training. On the first page you can see the losses, sublosses and metrics like mean average precision and f1 scores.

![Image not found](images/scalar.png?raw=true "Scalars")


On the second page, you find visualizations of a couple of validation images with their ground truth bounding boxes and how the predictions change over the course of the training.


![Image not found](images/visualization.png?raw=true "Scalars")

The third page gives you a nice view over the network graph.

![Image not found](images/graph.png?raw=true "Graph")






