# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""Model configuration for pascal dataset"""

import numpy as np
from easydict import EasyDict as edict
import json
import argparse


def squeezeDet_config(name):
    """Specify the parameters to tune below."""
    cfg = edict()

    #we only care about these, others are omiited
    cfg.CLASS_NAMES = ['cyclist', 'pedestrian', 'car']

    # number of categories to classify
    cfg.CLASSES = len(cfg.CLASS_NAMES)

    # classes to class index dict
    cfg.CLASS_TO_IDX = dict(zip(cfg.CLASS_NAMES, range(cfg.CLASSES)))


    # Probability to keep a node in dropout
    cfg.KEEP_PROB = 0.5

    # a small value used to prevent numerical instability
    cfg.EPSILON = 1e-16

    # threshold for safe exponential operation
    cfg.EXP_THRESH = 1.0


    #image properties
    cfg.IMAGE_WIDTH           = 1248
    cfg.IMAGE_HEIGHT          = 384
    cfg.N_CHANNELS            = 3

    #batch sizes
    cfg.BATCH_SIZE            = 4
    cfg.VISUALIZATION_BATCH_SIZE = 16

    #SGD + Momentum parameters
    cfg.WEIGHT_DECAY          = 0.001
    cfg.LEARNING_RATE         = 0.01
    cfg.MAX_GRAD_NORM         = 1.0
    cfg.MOMENTUM              = 0.9

    #coefficients of loss function
    cfg.LOSS_COEF_BBOX        = 5.0
    cfg.LOSS_COEF_CONF_POS    = 75.0
    cfg.LOSS_COEF_CONF_NEG    = 100.0
    cfg.LOSS_COEF_CLASS       = 1.0


    #thesholds for evaluation
    cfg.NMS_THRESH            = 0.4
    cfg.PROB_THRESH           = 0.005
    cfg.TOP_N_DETECTION       = 64
    cfg.IOU_THRESHOLD         = 0.5
    cfg.FINAL_THRESHOLD       = 0.0

    
    cfg.ANCHOR_SEED = np.array([[  36.,  37.], [ 366., 174.], [ 115.,  59.],
                                [ 162.,  87.], [  38.,  90.], [ 258., 173.],
                                [ 224., 108.], [  78., 170.], [  72.,  43.]])



    cfg.ANCHOR_PER_GRID       = len(cfg.ANCHOR_SEED)

    cfg.ANCHORS_HEIGHT = 24
    cfg.ANCHORS_WIDTH = 78

    return cfg


def create_config_from_dict(dictionary = {}, name="squeeze.config"):
    """Creates a config and saves it
    
    Keyword Arguments:
        dictionary {dict} -- [description] (default: {{}})
        name {str} -- [description] (default: {"squeeze.config"})
    """

    cfg = squeezeDet_config(name)

    for key, value in dictionary.items():

        cfg[key] = value

    save_dict(cfg, name)

#save a config files to json
def save_dict(dict, name="squeeze.config"):

    #change np arrays to lists for storing
    for key, val, in dict.items():

        if type(val) is np.ndarray:

            dict[key] = val.tolist()

    with open( name, "w") as f:
        json.dump(dict, f, sort_keys=True, indent=0 )  ### this saves the array in .json format


def load_dict(path):
    """Loads a dictionary from a given path name
    
    Arguments:
        path {[type]} -- string of path
    
    Returns:
        [type] -- [description]
    """

    with open(path, "r") as f:
        cfg = json.load(f)  ### this loads the array from .json format


    #changes lists back
    for key, val, in cfg.items():

        if type(val) is list:
            cfg[key] = np.array(val)

    #cast do easydict
    cfg = edict(cfg)

    #create full anchors from seed
    cfg.ANCHOR_BOX, cfg.N_ANCHORS_HEIGHT, cfg.N_ANCHORS_WIDTH = set_anchors(cfg)
    cfg.ANCHORS = len(cfg.ANCHOR_BOX)

    #if you added a class in the config manually, but were to lazy to update
    cfg.CLASSES = len(cfg.CLASS_NAMES)
    cfg.CLASS_TO_IDX = dict(zip(cfg.CLASS_NAMES, range(cfg.CLASSES)))



    return cfg


#compute the anchors for the grid from the seed
def set_anchors(cfg):
  H, W, B = cfg.ANCHORS_HEIGHT, cfg.ANCHORS_WIDTH, cfg.ANCHOR_PER_GRID


  anchor_shapes = np.reshape(
      [cfg.ANCHOR_SEED] * H * W,
      (H, W, B, 2)
  )
  center_x = np.reshape(
      np.transpose(
          np.reshape(
              np.array([np.arange(1, W+1)*float(cfg.IMAGE_WIDTH)/(W+1)]*H*B),
              (B, H, W)
          ),
          (1, 2, 0)
      ),
      (H, W, B, 1)
  )
  center_y = np.reshape(
      np.transpose(
          np.reshape(
              np.array([np.arange(1, H+1)*float(cfg.IMAGE_HEIGHT)/(H+1)]*W*B),
              (B, W, H)
          ),
          (2, 1, 0)
      ),
      (H, W, B, 1)
  )
  anchors = np.reshape(
      np.concatenate((center_x, center_y, anchor_shapes), axis=3),
      (-1, 4)
  )

  return anchors, H, W




if __name__ == "__main__":
    
  # parse arguments
  parser = argparse.ArgumentParser(description='Creates config file for squeezeDet training')
  parser.add_argument("--name", help="Name of the config file. DEFAULT: squeeze.config")
  args = parser.parse_args()

  name = "squeeze.config"


  if args.name:
      name = args.name

  cfg = squeezeDet_config(name=name)


  save_dict(cfg, name)
