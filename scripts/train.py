# Project: squeezeDetOnKeras
# Filename: train
# Author: Christopher Ehmann
# Date: 08.12.17
# Organisation: searchInk
# Email: christopher@searchink.com


from main.model.squeezeDet import  SqueezeDet
from main.model.dataGenerator import generator_from_data_path
import keras.backend as K
from keras import optimizers
import tensorflow as tf
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.models import load_model
from main.model.multi_gpu_model_checkpoint import  ModelCheckpointMultiGPU
import argparse
import os
import gc
from keras.utils import multi_gpu_model
import pickle
from main.config.create_config import load_dict
from main.model.loss import losses

#global variables can be set by optional arguments
#TODO: Makes proper variables in train() instead of global arguments.


def train(img_file = "img_train.txt",
        gt_file = "gt_train.txt",
        log_dir_name = './log',
        init_file = None,
        EPOCHS = 300,
        STEPS = None,
        OPTIMIZER = "SGD",
        CUDA_VISIBLE_DEVICES = "0",
        GPUS = 1,
        NOREDUCELRONPLATEAU = 0,
        VERBOSE=False,
        CONFIG = "squeeze.config",
        INITFROMCONFIG=0):

    """Def trains a Keras model of SqueezeDet and stores the checkpoint after each epoch
    """


    #create subdirs for logging of checkpoints and tensorboard stuff
    checkpoint_dir = log_dir_name +"/checkpoints"
    tb_dir = log_dir_name +"/tensorboard"



    #delete old checkpoints and tensorboard stuff
    if tf.gfile.Exists(checkpoint_dir):
        tf.gfile.DeleteRecursively(checkpoint_dir)

    if tf.gfile.Exists(tb_dir):
        tf.gfile.DeleteRecursively(tb_dir)

    tf.gfile.MakeDirs(tb_dir)
    tf.gfile.MakeDirs(checkpoint_dir)



    #open files with images and ground truths files with full path names
    with open(img_file) as imgs:
        img_names = imgs.read().splitlines()
    imgs.close()
    with open(gt_file) as gts:
        gt_names = gts.read().splitlines()
    gts.close()


    #create config object
    cfg = load_dict(CONFIG)


    #add stuff for documentation to config
    cfg.img_file = img_file
    cfg.gt_file = gt_file
    cfg.images = img_names
    cfg.gts = gt_names
    cfg.init_file = init_file
    cfg.EPOCHS = EPOCHS
    cfg.OPTIMIZER = OPTIMIZER
    cfg.CUDA_VISIBLE_DEVICES = CUDA_VISIBLE_DEVICES
    cfg.GPUS = GPUS
    cfg.NOREDUCELRONPLATEAU = NOREDUCELRONPLATEAU




    #set gpu
    if GPUS < 2:

        os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES

    else:

        gpus = ""
        for i in range(GPUS):
            gpus +=  str(i)+","
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus


    #scale batch size to gpus
    cfg.BATCH_SIZE = cfg.BATCH_SIZE * GPUS

    #compute number of batches per epoch
    nbatches_train, mod = divmod(len(img_names), cfg.BATCH_SIZE)

    if STEPS is not None:
        nbatches_train = STEPS

    cfg.STEPS = nbatches_train

    #print some run info
    print("Number of images: {}".format(len(img_names)))
    print("Number of epochs: {}".format(EPOCHS))
    print("Number of batches: {}".format(nbatches_train))
    print("Batch size: {}".format(cfg.BATCH_SIZE))

    #tf config and session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    K.set_session(sess)



    #callbacks
    cb = []


    #set optimizer
    #multiply by number of workers do adjust for increased batch size
    if OPTIMIZER == "adam":
        opt = optimizers.Adam(lr=cfg.LEARNING_RATE,  clipnorm=cfg.MAX_GRAD_NORM)
        cfg.LR= 0.001 
    if OPTIMIZER == "rmsprop":
        opt = optimizers.RMSprop(lr=cfg.LEARNING_RATE ,  clipnorm=cfg.MAX_GRAD_NORM)
        cfg.LR= 0.001 

    if OPTIMIZER == "adagrad":
        opt = optimizers.Adagrad(lr=cfg.LEARNING_RATE ,  clipnorm=cfg.MAX_GRAD_NORM)
        cfg.LR = 1

    #use default is nothing is given
    else:


        # create sgd with momentum and gradient clipping
        opt = optimizers.SGD(lr=cfg.LEARNING_RATE , decay=0, momentum=cfg.MOMENTUM,
                             nesterov=False, clipnorm=cfg.MAX_GRAD_NORM)

        cfg.LR = cfg.LEARNING_RATE  * GPUS


        print("Learning rate: {}".format(cfg.LEARNING_RATE))




    #save config file to log dir
    with open( log_dir_name  +'/config.pkl', 'wb') as f:
        pickle.dump(cfg, f, pickle.HIGHEST_PROTOCOL)


    #add tensorboard callback
    tbCallBack = TensorBoard(log_dir=tb_dir, histogram_freq=0,
                             write_graph=True, write_images=True)

    cb.append(tbCallBack)

    #if flag was given, add reducelronplateu callback
    if not NOREDUCELRONPLATEAU:

        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1,verbose=1,
                                      patience=5, min_lr=0.0)

        cb.append(reduce_lr)


    #if you just want to load the hdf5 file containing weights and architecture
    if not INITFROMCONFIG:
        print("Architecture and weights initialized from {}".format(init_file))

        model = load_model(init_file)


    else:

        print("Initialized architecture from config.")
        #instantiate model with random weights
        model = SqueezeDet(cfg).model

        #if file was provided load possible weights
        if not init_file is None:

            from main.model.modelLoading import load_only_possible_weights
            print("Weights initialized by name from {}".format(init_file))

            load_only_possible_weights(model, init_file, verbose=VERBOSE)

            
    #print keras model summary
    if VERBOSE:
        print(model.summary())

    

    ls = losses(cfg)

    #create train generator
    train_generator = generator_from_data_path(img_names, gt_names, config=cfg)

    #make model parallel if specified
    if GPUS > 1:

        #use multigpu model checkpoint
        ckp_saver = ModelCheckpointMultiGPU(checkpoint_dir + "/model.{epoch:02d}-{loss:.2f}.hdf5", monitor='loss', verbose=0,
                                    save_best_only=False,
                                    save_weights_only=False, mode='auto', period=1)

        cb.append(ckp_saver)


        print("Using multi gpu support with {} GPUs".format(GPUS))

        # make the model parallel
        parallel_model = multi_gpu_model(model, gpus=GPUS)
        parallel_model.compile(optimizer=opt,
                              loss=[ls.total_loss], metrics=[ls.loss_without_regularization,
                               ls.bbox_loss, ls.class_loss, ls.conf_loss])



        #actually do the training
        parallel_model.fit_generator(train_generator, epochs=EPOCHS,
                                        steps_per_epoch=nbatches_train, callbacks=cb)


    else:

        # add a checkpoint saver
        ckp_saver = ModelCheckpoint(checkpoint_dir + "/model.{epoch:02d}-{loss:.2f}.hdf5", monitor='loss', verbose=0,
                                    save_best_only=False,
                                    save_weights_only=False, mode='auto', period=1)
        cb.append(ckp_saver)


        print("Using single GPU")
        #compile model from squeeze object, loss is not a function of model directly
        model.compile(optimizer=opt,
                              loss=[ls.total_loss], metrics=[ls.loss_without_regularization,
                               ls.bbox_loss, ls.class_loss, ls.conf_loss])

        #actually do the training
        model.fit_generator(train_generator, epochs=EPOCHS,
                                        steps_per_epoch=nbatches_train, callbacks=cb)

    try:
        gc.collect()

    except:
        pass
if __name__ == "__main__":



    #parse arguments
    parser = argparse.ArgumentParser(description='Train squeezeDet model.')
    parser.add_argument("--img", help="file of full path names for the training images. DEFAULT: img_train.txt", default="img_train.txt")
    parser.add_argument("--epochs", type=int, help="number of epochs. DEFAULT: 300", default=300)
    parser.add_argument("--config",   help="Dictionary of all the hyperparameters.  DEFAULT: squeeze.config", default="squeeze.config")

    parser.add_argument("--gt", help="file of full path names for the corresponding training gts. DEFAULT: gt_train.txt", default="gt_train.txt")
    parser.add_argument("--steps",  type=int, help="steps per epoch. DEFAULT: #imgs/ batch size")
    parser.add_argument("--optimizer",  help="Which optimizer to use. DEFAULT: SGD with Momentum and lr decay OPTIONS: \
    sgd, adam, adagrad, rmsprop", default="sgd", choices=["sgd", "adam", "adagrad", "rmsprop"])
    parser.add_argument("--logdir", help="dir with checkpoints and loggings. DEFAULT: ./log", default="./log")
   
    parser.add_argument("--gpu",  help="which gpu to use. DEFAULT: 0", default="0")
    parser.add_argument("--gpus", type=int, \
     help="number of GPUS to use when using multi gpu support. Overwrites gpu flag. DEFAULT: 1", default=1)
    parser.add_argument("--init",  help="keras checkpoint to start training from. If argument is not given \
    training starts from random init.")
    parser.add_argument("--no-reduced-lr",help="If added, disables automatics reduction of learning rate.",action='store_true')
    parser.add_argument("--verbose",  help="If added, prints additional information.",action='store_true')
    parser.add_argument("--init-from-config",  help="If added, instead of loading the hdf5 file, it creates a model \
    specified in the config, and loads the possible weights.",action='store_true')

    args = parser.parse_args()


    args_vars = vars(args)

    #print(args_vars)

    

    train(img_file = args.img,
        gt_file = args.gt,
        log_dir_name = args.logdir,
        init_file = args.init,
        EPOCHS = args.epochs,
        STEPS = args.steps,
        OPTIMIZER = args.optimizer,
        CUDA_VISIBLE_DEVICES = args.gpu,
        GPUS = args.gpus,
        NOREDUCELRONPLATEAU = args.no_reduced_lr,
        VERBOSE= args.config,
        INITFROMCONFIG=args.init_from_config)
