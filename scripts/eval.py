# Project: squeezeDetOnKeras
# Filename: eval
# Author: Christopher Ehmann
# Date: 08.12.17
# Organisation: searchInk
# Email: christopher@searchink.com


from main.model.squeezeDet import  SqueezeDet
from main.model.dataGenerator import generator_from_data_path, visualization_generator_from_data_path
import keras.backend as K
from keras import optimizers
import tensorflow as tf
from main.model.evaluation import evaluate
from main.model.visualization import  visualize
import os
import time
import numpy as np
import argparse
from keras.utils import multi_gpu_model
from main.config.create_config import load_dict

#default values for some variables
#TODO: uses them as proper parameters instead of global variables
img_file = "img_val.txt"
gt_file = "gt_val.txt"
img_file_test = "img_test.txt"
gt_file_test = "gt_test.txt"
log_dir_name = "./log"
checkpoint_dir = './log/checkpoints'
tensorboard_dir = './log/tensorboard_val'
tensorboard_dir_test = './log/tensorboard_test'
TIMEOUT = 20
EPOCHS = 100
CUDA_VISIBLE_DEVICES = "1"
steps = None
GPUS = 1
STARTWITH = None
CONFIG = "squeeze.config"
TESTING = False



def eval():
    """
    Checks for keras checkpoints in a tensorflow dir and evaluates losses and given metrics. Also creates visualization and
    writes everything to tensorboard.
    """

    #create config object
    cfg = load_dict(CONFIG)


    #open files with images and ground truths files with full path names
    with open(img_file) as imgs:
        img_names = imgs.read().splitlines()
    imgs.close()
    with open(gt_file) as gts:
        gt_names = gts.read().splitlines()
    gts.close()

    #if multigpu support, adjust batch size
    if GPUS > 1:
        cfg.BATCH_SIZE = GPUS * cfg.BATCH_SIZE


    #compute number of batches per epoch
    nbatches_valid, mod = divmod(len(gt_names), cfg.BATCH_SIZE)

    #if a number for steps was given
    if steps is not None:
        nbatches_valid = steps

    #set gpu to use if no multigpu


    #hide the other gpus so tensorflow only uses this one
    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES


    #tf config and session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    K.set_session(sess)


    #Variables to visualize losses (as metrics) for tensorboard
    loss_var = tf.Variable(
        initial_value=0, trainable=False,
        name='val_loss', dtype=tf.float32
    )

    loss_without_regularization_var = tf.Variable(
        initial_value=0, trainable=False,
        name='val_loss_without_regularization', dtype=tf.float32
    )

    conf_loss_var = tf.Variable(
        initial_value=0, trainable=False,
        name='val_conf_loss', dtype=tf.float32
    )
    class_loss_var = tf.Variable(
        initial_value=0, trainable=False,
        name='val_class_loss', dtype=tf.float32
    )

    bbox_loss_var = tf.Variable(
        initial_value=0, trainable=False,
        name='val_bbox_loss', dtype=tf.float32
    )

    #create placeholders for metrics. Variables get assigned these.
    loss_placeholder = tf.placeholder(loss_var.dtype, shape=())
    loss_without_regularization_placeholder = tf.placeholder(loss_without_regularization_var.dtype, shape=())
    conf_loss_placeholder = tf.placeholder(conf_loss_var.dtype, shape=())
    class_loss_placeholder = tf.placeholder(class_loss_var.dtype, shape=())
    bbox_loss_placeholder = tf.placeholder(bbox_loss_var.dtype, shape=())


    #we have to create the assign ops here and call the assign ops with a feed dict, otherwise memory leak
    loss_assign_ops = [ loss_var.assign(loss_placeholder),
                        loss_without_regularization_var.assign(loss_without_regularization_placeholder),
                        conf_loss_var.assign(conf_loss_placeholder),
                        class_loss_var.assign(class_loss_placeholder),
                        bbox_loss_var.assign(bbox_loss_placeholder) ]

    tf.summary.scalar("loss", loss_var)
    tf.summary.scalar("loss_without_regularization", loss_without_regularization_var)
    tf.summary.scalar("conf_loss", conf_loss_var)
    tf.summary.scalar("class_loss", class_loss_var)
    tf.summary.scalar("bbox_loss", bbox_loss_var)


    #variables for images to visualize
    images_with_boxes = tf.Variable(  initial_value = np.zeros((cfg.VISUALIZATION_BATCH_SIZE, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, 3)), name="image", dtype=tf.float32)
    update_placeholder = tf.placeholder(images_with_boxes.dtype, shape=images_with_boxes.get_shape())
    update_images = images_with_boxes.assign(update_placeholder)

    tf.summary.image("images", images_with_boxes, max_outputs=cfg.VISUALIZATION_BATCH_SIZE )

    #variables for precision recall and mean average precision
    precisions = []
    recalls = []
    APs = []
    f1s= []

    #placeholders as above
    precision_placeholders = []
    recall_placeholders = []
    AP_placeholders = []
    f1_placeholders = []
    prmap_assign_ops = []


    #add variables, placeholders and assign ops for each class
    for i, name in enumerate(cfg.CLASS_NAMES):

        print("Creating tensorboard plots for " + name)

        precisions.append( tf.Variable(
            initial_value=0, trainable=False,
            name="precision/" +name , dtype=tf.float32
        ))
        recalls.append( tf.Variable(
            initial_value=0, trainable=False,
            name="recall/" +name , dtype=tf.float32
        ))

        f1s.append( tf.Variable(
            initial_value=0, trainable=False,
            name="f1/" +name , dtype=tf.float32
        ))

        APs.append( tf.Variable(
            initial_value=0, trainable=False,
            name="AP/" +name , dtype=tf.float32
        ))

        precision_placeholders.append(
            tf.placeholder(dtype=precisions[i].dtype,
                           shape=precisions[i].shape))

        recall_placeholders.append(
            tf.placeholder(dtype=recalls[i].dtype,
                           shape=recalls[i].shape))
        AP_placeholders.append(
            tf.placeholder(dtype=APs[i].dtype,
                           shape=APs[i].shape))

        f1_placeholders.append(
            tf.placeholder(dtype=f1s[i].dtype,
                           shape=f1s[i].shape))



        prmap_assign_ops.append( precisions[i].assign(precision_placeholders[i]))
        prmap_assign_ops.append(recalls[i].assign(recall_placeholders[i]))
        prmap_assign_ops.append(APs[i].assign(AP_placeholders[i]))
        prmap_assign_ops.append(f1s[i].assign(f1_placeholders[i]))



    #same for mean average precision

    mAP = tf.Variable(
        initial_value=0, trainable=False,
        name="mAP", dtype=tf.float32
    )

    mAP_placeholder = tf.placeholder(mAP.dtype, shape=())

    prmap_assign_ops.append(mAP.assign(mAP_placeholder))

    tf.summary.scalar("mAP", mAP)

    for i, name in enumerate(cfg.CLASS_NAMES):

        tf.summary.scalar("precision/" + name, precisions[i])
        tf.summary.scalar("recall/" + name, recalls[i])
        tf.summary.scalar("AP/" + name, APs[i])
        tf.summary.scalar("f1/" + name, f1s[i])

    merged = tf.summary.merge_all()

    if STARTWITH is None:
        #check for tensorboard dir and delete old stuff
        if tf.gfile.Exists(tensorboard_dir):
            tf.gfile.DeleteRecursively(tensorboard_dir)
        tf.gfile.MakeDirs(tensorboard_dir)

    writer = tf.summary.FileWriter(tensorboard_dir)


    #instantiate model
    squeeze = SqueezeDet(cfg)

    #dummy optimizer for compilation
    sgd = optimizers.SGD(lr=cfg.LEARNING_RATE, decay=0, momentum=cfg.MOMENTUM,
                         nesterov=False, clipnorm=cfg.MAX_GRAD_NORM)




    if GPUS > 1:

        #parallelize model
        model = multi_gpu_model(squeeze.model, gpus=GPUS)
        model.compile(optimizer=sgd,
                              loss=[squeeze.loss], metrics=[squeeze.bbox_loss, squeeze.class_loss,
                                                            squeeze.conf_loss, squeeze.loss_without_regularization])


    else:
    #compile model from squeeze object, loss is not a function of model directly
        squeeze.model.compile(optimizer=sgd,
                              loss=[squeeze.loss], metrics=[squeeze.bbox_loss, squeeze.class_loss,
                                                            squeeze.conf_loss, squeeze.loss_without_regularization])

        model = squeeze.model
    #models already evaluated
    evaluated_models = set()



    #get the best ckpts for test set

    best_val_loss_ckpt = None

    best_val_loss = np.inf

    best_mAP_ckpt = None

    best_mAP = -np.inf

    time_out_counter = 0


    #use this for saving metrics to a csv
    f = open( log_dir_name +  "/metrics.csv", "w")

    header = "epoch;regularized;loss;bbox;class;conf;"

    for i, name in enumerate(cfg.CLASS_NAMES):

        header += name +"_precision;" + name+"_recall;" + name + "_AP;" + name + "_f1;"

    header += "\n"

    f.write(header)

    #listening for new checkpoints
    while 1:

        current_model = None

        #go through checkpoint dir
        for ckpt in sorted(os.listdir(checkpoint_dir)):

            if STARTWITH is not None:
                if ckpt < STARTWITH:
                    evaluated_models.add(ckpt)



            #if model hasn't been evaluated
            if ckpt not in evaluated_models:

                #string for csv
                line = ""

                #add epoch to csv
                line += str(len(evaluated_models)+1) +";"

                print("Evaluating model {}".format(ckpt))

                #load this ckpt
                current_model= ckpt
                try:
                    squeeze.model.load_weights(checkpoint_dir + "/"+ ckpt)

                #sometimes model loading files, because the file is still locked, so wait a little bit
                except OSError as e:
                    print(e)
                    time.sleep(10)
                    squeeze.model.load_weights(checkpoint_dir + "/" + ckpt)

                # create 2 validation generators, one for metrics and one for object detection evaluation
                # we have to reset them each time to have the same data, otherwise we'd have to use batch size one.
                val_generator_1 = generator_from_data_path(img_names, gt_names, config=cfg)
                val_generator_2 = generator_from_data_path(img_names, gt_names, config=cfg)
                # create a generator for the visualization of bounding boxes
                vis_generator = visualization_generator_from_data_path(img_names, gt_names, config=cfg)

                print("  Evaluate losses...")
                #compute losses of whole val set
                losses = model.evaluate_generator(val_generator_1, steps=nbatches_valid, max_queue_size=10,
                                                         use_multiprocessing=False)


                #manually add losses to tensorboard
                sess.run(loss_assign_ops , {loss_placeholder: losses[0],
                                            loss_without_regularization_placeholder: losses[4],
                                            conf_loss_placeholder: losses[3],
                                            class_loss_placeholder: losses[2],
                                            bbox_loss_placeholder: losses[1]})

                #print losses
                print("  Losses:")
                print("  Loss with regularization: {}   val loss:{} \n     bbox_loss:{} \n     class_loss:{} \n     conf_loss:{}".
                      format(losses[0], losses[4], losses[1], losses[2], losses[3]) )

                line += "{};{};{};{};{};".format(losses[0] , losses[4], losses[1], losses[2], losses[3])

                #save model with smallest loss
                if losses[4] < best_val_loss:
                    best_val_loss = losses[4]
                    best_val_loss_ckpt = current_model



                #compute precision recall and mean average precision
                precision, recall, f1,  AP = evaluate(model=model, generator=val_generator_2, steps=nbatches_valid, config=cfg)

                #create feed dict for visualization
                prmap_feed_dict = {}
                for i, name in enumerate(cfg.CLASS_NAMES):

                    prmap_feed_dict[precision_placeholders[i]] = precision[i]
                    prmap_feed_dict[recall_placeholders[i]] = recall[i]
                    prmap_feed_dict[AP_placeholders[i]] = AP[i,1]
                    prmap_feed_dict[f1_placeholders[i]] = f1[i]

                    line += "{};{};{};{}".format( precision[i], recall[i],AP[i,1],f1[i])




                prmap_feed_dict[mAP_placeholder] = np.mean(AP[:,1], axis=0)

                #save model with biggest mean average precision
                if np.mean(AP[:,1], axis=0) > best_mAP:
                    best_mAP = np.mean(AP[:,1], axis=0)
                    best_mAP_ckpt = current_model


                #run loss assign ops for tensorboard
                sess.run(prmap_assign_ops, prmap_feed_dict)

                #create visualization
                imgs = visualize( model=model, generator=vis_generator, config=cfg)

                ##update op for images
                sess.run(update_images, {update_placeholder:imgs})

                #write everything to tensorboard
                writer.add_summary(merged.eval(session=sess), len(evaluated_models))

                writer.flush()

                f.write(line + "\n")

                f.flush()

                #mark as evaluated
                evaluated_models.add(ckpt)

                #reset timeout
                time_out_counter = 0

            #if all ckpts have been evaluated on val set end
            if len(evaluated_models) == EPOCHS:
                break


        #if all ckpts have been evaluated on val set end
        if len(evaluated_models) == EPOCHS:
            print("Evaluated all checkpoints")
            break

        #no new model found
        if current_model is None:

            #when timeout has been reached, abort
            if time_out_counter == TIMEOUT:
                print("timeout")
                break

            print("Waiting for new checkpoint....")
            time.sleep(60)
            time_out_counter += 1

    f.close()


    #evaluate best ckpts on test set

    #list of ckpts for test set evaluation
    ckpts = [best_val_loss_ckpt, best_mAP_ckpt]

    print("Lowest loss: {} at checkpoint {}".format(best_val_loss, best_val_loss_ckpt))
    print("Highest mAP: {} at checkpoint {}".format(best_mAP, best_mAP_ckpt))


    #evaluate on test set
    if TESTING:

        ckpts = set(ckpts)


        #get test images and gt
        with open(img_file_test) as imgs:

            img_names_test = imgs.read().splitlines()

        imgs.close()

        with open(gt_file_test) as gts:

            gt_names_test = gts.read().splitlines()

        gts.close()

        #compute number of batches per epoch
        nbatches_test, mod = divmod(len(gt_names_test), cfg.BATCH_SIZE)

        #if a number for steps was given
        if steps is not None:
            nbatches_test = steps



        #again create Variables to visualize losses for tensorboard, but this time for test set
        test_loss_var = tf.Variable(
            initial_value=0, trainable=False,
            name='test_loss', dtype=tf.float32
        )

        test_loss_without_regularization_var = tf.Variable(
            initial_value=0, trainable=False,
            name='test_loss_without_regularization', dtype=tf.float32
        )

        test_conf_loss_var = tf.Variable(
            initial_value=0, trainable=False,
            name='test_conf_loss', dtype=tf.float32
        )
        test_class_loss_var = tf.Variable(
            initial_value=0, trainable=False,
            name='test_class_loss', dtype=tf.float32
        )

        test_bbox_loss_var = tf.Variable(
            initial_value=0, trainable=False,
            name='test_bbox_loss', dtype=tf.float32
        )

        #we have to create the assign ops here and call the assign ops with a feed dictg, otherwise memory leak
        test_loss_placeholder = tf.placeholder(loss_var.dtype, shape=())
        test_loss_without_regularization_placeholder = tf.placeholder(loss_without_regularization_var.dtype, shape=())
        test_conf_loss_placeholder = tf.placeholder(conf_loss_var.dtype, shape=())
        test_class_loss_placeholder = tf.placeholder(class_loss_var.dtype, shape=())
        test_bbox_loss_placeholder = tf.placeholder(bbox_loss_var.dtype, shape=())

        test_loss_assign_ops = [ test_loss_var.assign(test_loss_placeholder),
                            test_loss_without_regularization_var.assign(test_loss_without_regularization_placeholder),
                            test_conf_loss_var.assign(test_conf_loss_placeholder),
                            test_class_loss_var.assign(test_class_loss_placeholder),
                            test_bbox_loss_var.assign(test_bbox_loss_placeholder) ]

        tf.summary.scalar("test/loss", loss_var, collections=["test"])
        tf.summary.scalar("test/loss_without_regularization", loss_without_regularization_var, collections=["test"])
        tf.summary.scalar("test/conf_loss", conf_loss_var, collections=["test"])
        tf.summary.scalar("test/class_loss", class_loss_var, collections=["test"])
        tf.summary.scalar("test/bbox_loss", bbox_loss_var, collections=["test"])


        #variables for precision recall and mean average precision
        precisions = []
        recalls = []
        APs = []
        f1s= []

        precision_placeholders = []
        recall_placeholders = []
        AP_placeholders = []
        f1_placeholders = []
        prmap_assign_ops = []

        for i, name in enumerate(cfg.CLASS_NAMES):

            precisions.append( tf.Variable(
                initial_value=0, trainable=False,
                name="test/precision/" +name , dtype=tf.float32
            ))
            recalls.append( tf.Variable(
                initial_value=0, trainable=False,
                name="test/recall/" +name , dtype=tf.float32
            ))

            f1s.append( tf.Variable(
                initial_value=0, trainable=False,
                name="test/f1/" +name , dtype=tf.float32
            ))

            APs.append( tf.Variable(
                initial_value=0, trainable=False,
                name="test/AP/" +name , dtype=tf.float32
            ))

            precision_placeholders.append(
                tf.placeholder(dtype=precisions[i].dtype,
                               shape=precisions[i].shape))

            recall_placeholders.append(
                tf.placeholder(dtype=recalls[i].dtype,
                               shape=recalls[i].shape))
            AP_placeholders.append(
                tf.placeholder(dtype=APs[i].dtype,
                               shape=APs[i].shape))

            f1_placeholders.append(
                tf.placeholder(dtype=f1s[i].dtype,
                               shape=f1s[i].shape))



            prmap_assign_ops.append(precisions[i].assign(precision_placeholders[i]))
            prmap_assign_ops.append(recalls[i].assign(recall_placeholders[i]))
            prmap_assign_ops.append(APs[i].assign(AP_placeholders[i]))
            prmap_assign_ops.append(f1s[i].assign(f1_placeholders[i]))


        test_mAP = tf.Variable(
            initial_value=0, trainable=False,
            name="mAP", dtype=tf.float32
        )

        mAP_placeholder = tf.placeholder(test_mAP.dtype, shape=())

        prmap_assign_ops.append(test_mAP.assign(mAP_placeholder))

        tf.summary.scalar("test/mAP", mAP, collections=["test"])
        tf.summary.scalar("test/precision/" + name, precisions[i], collections=["test"])
        tf.summary.scalar("test/recall/" + name, recalls[i], collections=["test"])
        tf.summary.scalar("test/AP/" + name, APs[i], collections=["test"])
        tf.summary.scalar("test/f1/" + name, f1s[i], collections=["test"])

        merged = tf.summary.merge_all(key="test")


        #check for tensorboard dir and delete old stuff
        if tf.gfile.Exists(tensorboard_dir_test):
            tf.gfile.DeleteRecursively(tensorboard_dir_test)
        tf.gfile.MakeDirs(tensorboard_dir_test)

        writer = tf.summary.FileWriter(tensorboard_dir_test)



        i=1

        #go through given checkpoints
        for ckpt in ckpts:


            print("Evaluating model {} on test data".format(ckpt) )


            #load this ckpt
            current_model = ckpt
            squeeze.model.load_weights(checkpoint_dir + "/"+ ckpt)

            # create 2 validation generators, one for metrics and one for object detection evaluation
            # we have to reset them each time to have the same data
            val_generator_1 = generator_from_data_path(img_names_test, gt_names_test, config=cfg)
            val_generator_2 = generator_from_data_path(img_names_test, gt_names_test, config=cfg)
            # create a generator for the visualization of bounding boxes

            print("  Evaluate losses...")
            #compute losses of whole val set
            losses = model.evaluate_generator(val_generator_1, steps=nbatches_test, max_queue_size=10,
                                                     use_multiprocessing=False)


            #manually add losses to tensorboard
            sess.run(loss_assign_ops , {loss_placeholder: losses[0],
                                        loss_without_regularization_placeholder: losses[4],
                                        conf_loss_placeholder: losses[3],
                                        class_loss_placeholder: losses[2],
                                        bbox_loss_placeholder: losses[1]})

            print("  Losses:")
            print("  Loss with regularization: {}   val loss:{} \n     bbox_loss:{} \n     class_loss:{} \n     conf_loss:{}".
                  format(losses[0], losses[4], losses[1], losses[2], losses[3]) )



            #compute precision recall and mean average precision
            precision, recall, f1,  AP = evaluate(model=model, generator=val_generator_2, steps=nbatches_test, config=cfg)

            #create feed dict for visualization
            prmap_feed_dict = {}
            for i, name in enumerate(cfg.CLASS_NAMES):
                prmap_feed_dict[precision_placeholders[i]] = precision[i]
                prmap_feed_dict[recall_placeholders[i]] = recall[i]
                prmap_feed_dict[AP_placeholders[i]] = AP[i,1]
                prmap_feed_dict[f1_placeholders[i]] = f1[i]


            prmap_feed_dict[mAP_placeholder] = np.mean(AP[:,1], axis=0)


            sess.run(prmap_assign_ops, prmap_feed_dict)


            #write everything to tensorboard
            writer.add_summary(merged.eval(session=sess), i)

            writer.flush()

            i+=1




if __name__ == "__main__":

    #argument parsing
    parser = argparse.ArgumentParser(description='Evaluate squeezeDet keras checkpoints after each epoch on validation set.')
    parser.add_argument("--logdir", help="dir with checkpoints and loggings. DEFAULT: ./log")
    parser.add_argument("--val_img", help="file of full path names for the validation images. DEFAULT: img_val.txt")
    parser.add_argument("--val_gt", help="file of full path names for the corresponding validation gts. DEFAULT: gt_val.txt")
    parser.add_argument("--test_img", help="file of full path names for the test images. DEFAULT: img_test.txt")
    parser.add_argument("--test_gt", help="file of full path names for the corresponding test gts. DEFAULT: gt_test.txt")
    parser.add_argument("--steps",  type=int, help="steps to evaluate. DEFAULT: length of imgs/ batch_size")
    parser.add_argument("--gpu",  help="gpu to use. DEFAULT: 1")
    parser.add_argument("--gpus",  type=int, help="gpus to use for multigpu usage. DEFAULT: 1")
    parser.add_argument("--epochs", type=int, help="number of epochs to evaluate before terminating. DEFAULT: 100")
    parser.add_argument("--timeout", type=int, help="number of minutes before the evaluation script stops after no new checkpoint has been detected. DEFAULT: 20")
    parser.add_argument("--init" , help="start evaluating at a later checkpoint")
    parser.add_argument("--config",   help="Dictionary of all the hyperparameters. DEFAULT: squeeze.config")
    parser.add_argument("--testing",   help="Run eval on test set. DEFAULT: False")

    args = parser.parse_args()

    #set global variables according to optional arguments
    if args.logdir is not None:
        log_dir_name = args.logdir
        checkpoint_dir = log_dir_name + '/checkpoints'
        tensorboard_dir = log_dir_name + '/tensorboard_val'

    if args.val_img is not None:
        img_file = args.val_img
    if args.val_gt is not None:
        gt_file = args.val_gt

    if args.test_img is not None:
        img_file_test = args.test_img
    if args.test_gt is not None:
        gt_file_test = args.test_gt

    if args.gpu is not None:
        CUDA_VISIBLE_DEVICES = args.gpu
    if args.epochs is not None:
        EPOCHS = args.epochs
    if args.timeout is not None:
        TIMEOUT = args.timeout

    if args.gpus is not None:
        GPUS = args.gpus


        #if there were no GPUS explicitly given, take the last ones
        #the assumption is, that we use as many gpus for evaluation as for training
        #so we have to hide the other gpus to not try to allocate memory there
        if args.gpu is None:
            CUDA_VISIBLE_DEVICES = ""
            for i in range(GPUS, 2*GPUS):
                CUDA_VISIBLE_DEVICES += str(i) + ","
            print(CUDA_VISIBLE_DEVICES)

    if args.init is not None:
        STARTWITH = args.init

    if args.steps is not None:
        steps = args.steps

    if args.config is not None:
        CONFIG = args.config

    if args.testing is not None:
        TESTING = args.testing

    eval()