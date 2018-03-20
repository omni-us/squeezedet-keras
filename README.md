# README #


## Keras/tensorflow implementation of SqueezeDet in Python 3 ##

## How do I get set up? ###

* Clone repository

	`git clone `

* Install pip3 and virtualenv, creater a virtualenv inside the repository folder

	`sudo apt-get install python3-pip`

	`pip3 install virtualenv`

	`cd squeezedet`

	`virtualenv -p python3 env`
	
* Add the project to the **env/bin/activate** of the virtualenv so the modules can be found

	
	`export PYTHONPATH=path/to/squeezedet`

* Start virtualenv

	`source env/bin/activate`
	
* Install dependencies

	`pip install -r requirements.txt`
	
* Optionally: If you want to use GPU install CUDA 8.0 with CUDNN 6.0. 

	* Get newest NVIDIA drivers for you GPU. 
	* I only tested on Ubuntu 16.04. For this I recommend getting the base installer run file from [here](https://developer.nvidia.com/cuda-80-ga2-download-archive) . Run this and do not install the drivers when you have to choose. If you install from paket manager other CUDA version might be overwritten.
	* Create a NVDIA developer account [here](https://developer.nvidia.com/cudnn). Download the CUDNN 6.0 files for CUDA 8.0. I recommend storing them seperatly and adding them in the .bashrc to your LD_LIBRARY_PATH as suggested [here](http://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html)



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

	`python ../../main/utils/create_config.py`

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




