# Project: squeezeDetOnKeras
# Filename: visualization
# Author: Christopher Ehmann
# Date: 12.12.17
# Organisation: searchInk
# Email: christopher@searchink.com

from main.model.evaluation import filter_batch
import cv2
import numpy as np


def visualize(model, generator, config):
    """Creates images with ground truth and from the model predicted boxes.
    
    Arguments:
        model {[type]} -- SqueezeDet Model
        generator {[type]} -- data generator yielding images and ground truth
        config {[type]} --  dict of various hyperparameters
    
    Returns:
        [type] -- numpy array of images with ground truth and prediction boxes added
    """


    #this is needed, if batch size is smaller than visualization batch size
    nbatches, mod = divmod( config.VISUALIZATION_BATCH_SIZE, config.BATCH_SIZE )



    print("  Creating Visualizations...")
    #iterate one batch

    count = 0


    all_boxes = []

    for images, y_true, images_only_resized in generator:



        #predict on batch
        y_pred = model.predict(images)


        #create visualizations
        images_with_boxes = visualize_dt_and_gt(images_only_resized, y_true, y_pred, config)

        #lazy hack if nothing was detected
        try:
            all_boxes.append(np.stack(images_with_boxes))
        except:
            pass




        count += 1

        if count >= nbatches:
            break
    try:
        return np.stack(all_boxes).reshape((-1, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3))
    except:
        return np.zeros( (nbatches*config.BATCH_SIZE, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3))



def visualize_dt_and_gt(images, y_true, y_pred, config):
    """Takes a batch of images and creates bounding box visualization on top
    
    Arguments:
        images {[type]} -- numpy tensor of images
        y_true {[type]} -- tensor of ground truth
        y_pred {[type]} -- tensor of predictions
        config {[type]} -- dict of various hyperparameters
    
    Returns:
        [type] -- dict of various hyperparameters
    """


    #normalize for printing
    #images = (images - np.min(images, axis=(1,2,3), keepdims=1)) / np.max(images, axis=(1,2,3), keepdims=1) - np.min(images, axis=(1,2,3), keepdims=1)

    img_with_boxes = []

    #filter batch with nms
    all_filtered_boxes, all_filtered_classes, all_filtered_scores = filter_batch(y_pred, config)


    #print(len(all_filtered_scores))
    #print(len(all_filtered_scores[0]))

    #get gt boxes
    box_input = y_true[:, :, 1:5]

    #and gt labels
    labels = y_true[:, :, 9:]

    font = cv2.FONT_HERSHEY_SIMPLEX

    #iterate images
    for i, img in enumerate(images):

        #get predicted boxes
        non_zero_boxes = box_input[i][box_input[i] > 0].reshape((-1,4))


        non_zero_labels = []

        #get the predicted labels
        for k, coords in enumerate(box_input[i,:]):
            if np.sum(coords) > 0:

                for j, l in enumerate(labels[i, k]):
                    if l == 1:
                        non_zero_labels.append(j)

        #iterate predicted boxes
        for j, det_box in enumerate(all_filtered_boxes[i]):

            #transform into xmin, ymin, xmax, ymax
            det_box = bbox_transform_single_box(det_box)

            #add rectangle and text
            cv2.rectangle(img, (det_box[0], det_box[1]), (det_box[2], det_box[3]), (0,0,255), 1)
            cv2.putText(img, config.CLASS_NAMES[all_filtered_classes[i][j]] + " " + str(all_filtered_scores[i][j]) , (det_box[0], det_box[1]), font, 0.5, (0,0,255), 1, cv2.LINE_AA)

        #iterate gt boxes
        for j, gt_box in enumerate(non_zero_boxes):

            #transform into xmin, ymin, xmax, ymax
            gt_box = bbox_transform_single_box(gt_box)

            #add rectangle and text
            cv2.rectangle(img, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), (0, 255, 0), 1)
            cv2.putText(img, config.CLASS_NAMES[int(non_zero_labels[j])], (gt_box[0], gt_box[1]), font, 0.5,
                        (0, 255, 0), 1, cv2.LINE_AA)

        #chagne to rgb
        img_with_boxes.append(img[:,:, [2,1,0]])

    return img_with_boxes








def bbox_transform_single_box(bbox):
    """convert a bbox of form [cx, cy, w, h] to [xmin, ymin, xmax, ymax]. Works
    for numpy array or list of tensors.
    """
    cx, cy, w, h = bbox
    out_box = [[]]*4
    out_box[0] = int(np.floor(cx-w/2))
    out_box[1] = int(np.floor(cy-h/2))
    out_box[2] = int(np.floor(cx+w/2))
    out_box[3] = int(np.floor(cy+h/2))

    return out_box

