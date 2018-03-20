# Installation guide #

### Requirements ###

* python3
* pip
* virtualenv

## Next steps ###

* Clone repository

	`git clone https://github.com/omni-us/squeezedet-keras.git`

* Install requirements with pip

	`cd squeezedet-keras`

	`virtualenv -p python3 venv`

* Start virtualenv

	`source venv/bin/activate`
	
* Install dependencies

	`pip install -r requirements.txt`
	
* Add the project to the **venv/bin/activate** of the virtualenv so the modules can be found

	`export PYTHONPATH=path/to/squeezedet`
	
* Optionally: If you want to use GPU install CUDA 8.0 with CUDNN 6.0. 

	* Get newest NVIDIA drivers for you GPU. 
	* I only tested on Ubuntu 16.04. For this I recommend getting the base installer run file from [here](https://developer.nvidia.com/cuda-80-ga2-download-archive) . Run this and do not install the drivers when you have to choose. If you install from paket manager other CUDA version might be overwritten.
	* Create a NVDIA developer account [here](https://developer.nvidia.com/cudnn). Download the CUDNN 6.0 files for CUDA 8.0. I recommend storing them seperatly and adding them in the .bashrc to your LD_LIBRARY_PATH as suggested [here](http://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html)
