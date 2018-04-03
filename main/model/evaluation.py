# Project: squeezeDetOnKeras
# Filename: evaluation
# Author: Christopher Ehmann
# Date: 08.12.17
# Organisation: searchInk
# Email: christopher@searchink.com


# Project: squeezeDetOnKeras
# Filename: visualization
# Author: Christopher Ehmann
# Date: 07.12.17
# Organisation: searchInk
# Email: christopher@searchink.com

import numpy as np
import main.utils.utils as utils


def evaluate(model, generator, steps, config):
    """evaluates a model on a generator
    
    Arguments:
        model  -- Keras model
        generator  -- The data generator to evaluate
        steps  -- Number of steps to evaluate
        config  -- a squeezedet config file
    
    Returns:
        [type] -- precision, recall, f1, APs for all classes
    """

    #count the  number of correctly processed batches
    batches_processed = 0
    
    all_boxes = []
    all_classes = []
    all_scores = []
    all_gts = []

    print("  Metric Evaluation:")
    print("    Predicting on batches...")

    #iterate all batches
    for images, y_true in generator:

        #predict on batch
        y_pred  = model.predict(images)

        #filter the batch
        boxes , classes, scores = filter_batch(y_pred, config)


        all_boxes.append(boxes)
        all_classes.append(classes)
        all_scores.append(scores)
        all_gts.append(y_true)
        batches_processed+=1

        if batches_processed == steps:
            break



    #compute evaluation statistics on whole evaluation set
    print("    Computing statistics...")
    precision, recall, f1,  APs = compute_statistics(all_boxes, all_classes, all_scores, all_gts,  config)

    return precision, recall, f1, APs


def filter_batch( y_pred,config):
    """filters boxes from predictions tensor
    
    Arguments:
        y_pred {[type]} -- tensor of predictions
        config {[type]} -- squeezedet config
    
    Returns:
        lists -- list of all boxes, list of the classes, list of the scores
    """




    #slice predictions vector
    pred_class_probs, pred_conf, pred_box_delta = utils.slice_predictions_np(y_pred, config)
    det_boxes = utils.boxes_from_deltas_np(pred_box_delta, config)

    #compute class probabilities
    probs = pred_class_probs * np.reshape(pred_conf, [config.BATCH_SIZE, config.ANCHORS, 1])
    det_probs = np.max(probs, 2)
    det_class = np.argmax(probs, 2)



    #count number of detections
    num_detections = 0


    all_filtered_boxes = []
    all_filtered_scores = []
    all_filtered_classes = [ ]

    #iterate batch
    for j in range(config.BATCH_SIZE):

        #filter predictions with non maximum suppression
        filtered_bbox, filtered_score, filtered_class = filter_prediction(det_boxes[j], det_probs[j],
                                                                          det_class[j], config)


        #you can use this to use as a final filter for the confidence score
        keep_idx = [idx for idx in range(len(filtered_score)) if filtered_score[idx] > float(config.FINAL_THRESHOLD)]

        final_boxes = [filtered_bbox[idx] for idx in keep_idx]

        final_probs = [filtered_score[idx] for idx in keep_idx]

        final_class = [filtered_class[idx] for idx in keep_idx]


        all_filtered_boxes.append(final_boxes)
        all_filtered_classes.append(final_class)
        all_filtered_scores.append(final_probs)


        num_detections += len(filtered_bbox)


    return all_filtered_boxes, all_filtered_classes, all_filtered_scores


def compute_statistics(all_boxes, all_classes, all_scores, all_gts, config):
    """Computes statistics of all predictions
    
    Arguments:
        all_boxes {[type]} -- list of predicted boxes
        all_classes {[type]} -- list of predicted classes
        all_scores {[type]} --list of predicted scores  
        all_gts {[type]} -- list of all y_trues
        config {[type]} -- squeezedet config
    
    Returns:
        [type] --  prec, rec, f1, APs for all classes
    """


    #compute the initial scores for precision recall thresholding
    boxes_per_img, boxes_per_gt, all_tps, all_fps, all_fns, is_gt, all_scores = \
    compute_statistics_for_thresholding(all_boxes, all_classes, all_scores, all_gts, config)


    #precision and recall for printing
    prec = precision(tp=np.sum(all_tps,axis=0), fp=np.sum(all_fps,axis=0))
    rec = recall(tp=np.sum(all_tps, axis=0), fn=np.sum(all_fns,axis=0))


    #compute f1 score
    f1 = 2 * prec * rec / (prec+rec+1e-20)

    #compute mean average precisions
    APs, precs, iprecs = AP(is_gt, all_scores)


    #print some info
    print("    Objects {} of {} detected with {} predictions made".format(np.sum(all_tps), np.sum(boxes_per_gt), np.sum(boxes_per_img)))
    for i, name in enumerate(config.CLASS_NAMES):
        print("    Class {}".format(name))
        print("      Precision: {}  Recall: {}".format(prec[i], rec[i]))
        print("      AP: {}".format(APs[i,1]))



    return prec, rec, f1, APs


def compute_statistics_for_thresholding(all_boxes, all_classes, all_scores, all_gts, config):
    """Compute tps, fps, fns, and other stuff for computing APs
    
    
    Arguments:
        all_boxes {[type]} -- list of predicted boxes
        all_classes {[type]} -- list of predicted classes
        all_scores {[type]} --list of predicted scores  
        all_gts {[type]} -- list of all y_trues
        config {[type]} -- squeezedet config
    
    Returns:
        [type] -- boxes_per_img , boxes_per_gt, np.stack(all_tps), np.stack(all_fps), np.stack(all_fns), is_gt, all_score_thresholds
    """



    boxes_per_img = []
    boxes_per_gt = []

    all_tps = []
    all_fps = []

    all_fns = []
    all_score_thresholds = [ [] for c in range(config.CLASSES) ]
    is_gt = [ [] for c in range(config.CLASSES) ]



    #print(all_score_thresholds)

    #here we compute the false positives, false negatives and true positives of the network predictions
    #we cannot do everything in a numpy array as each image has a different number of filtered detections

    #iterate all batches
    for i in range(len(all_boxes)):

        batch_gt = all_gts[i]

        batch_classes = all_classes[i]

        batch_scores = all_scores[i]

        #shape is batch_size * achors * x
        box_input = batch_gt[:, :, 1:5]
        labels = batch_gt[:, :, 9:]

        #print(labels.shape)


        #iterate images per batch for image level analysis
        for j in range(len(all_boxes[i])):

            # add number of detections
            boxes_per_img.append(len(all_boxes[i][j]))

            #get index of non zero boxes
            non_zero_idx = np.sum(box_input[j][:], axis=-1) > 0

            #get non zero gt boxes
            nonzero_gts = np.reshape(box_input[j][non_zero_idx], [-1,4])

            # add number of gt boxes
            boxes_per_gt.append(len(nonzero_gts))


            #get labels
            labels_per_image = labels[j]


            #get non zero labels
            nonzero_labels = [ tuple[0]  for labels in  labels_per_image[non_zero_idx,:].astype(int) for tuple in enumerate(labels) if tuple[1]==1  ]

            #for every class count the true positives, false positives and false negatives
            tp_per_image = np.zeros(config.CLASSES)
            fp_per_image = np.zeros(config.CLASSES)
            fn_per_image = np.zeros(config.CLASSES)


            #print(batch_classes[j])

            #use this to check if predicted box has already been assigned to a different gt
            assigned_idx = np.zeros_like(batch_classes[j])

            # for every gt per image compute overlaps with detections
            for k in range(len(nonzero_gts)):

                try:
                    #get overlap between gt box and all predictions
                    ious = utils.batch_iou(np.stack(all_boxes[i][j]), nonzero_gts[k])

                    #use this to check for biggest score
                    current_score = -1
                    #index of best detection
                    current_idx = -1

                    #iterate all the ious
                    for iou_index, iou in enumerate(ious):


                        # check if iou is above threshold, if classes match,
                        # if it has not been assigned before and if the score is bigger than the current best score
                        # if all conditions are satisfied this marked as the current biggest detection
                        if iou > config.IOU_THRESHOLD \
                        and batch_classes[j][iou_index] == nonzero_labels[k] \
                        and not assigned_idx[iou_index]\
                        and batch_scores[j][iou_index] > current_score:

                            #update current score
                            current_score  = batch_scores[j][iou_index]
                            #update idx of best
                            current_idx = iou_index

                    #if nothing was assigned to this box add a false negative
                    if current_score < 0:
                        fn_per_image[nonzero_labels[k]] += 1

                        #for mAP calc set this to a gt
                        is_gt[nonzero_labels[k]].append(1)
                        #append 0 as the score, as we did not detect it
                        all_score_thresholds[nonzero_labels[k]].append(0)
                    else:
                        #otherwise add a true positive for the corresponding class
                        tp_per_image[nonzero_labels[k]] += 1
                        # set to ignore assigned box
                        assigned_idx[current_idx] = 1
                        #append it as a gt
                        is_gt[nonzero_labels[k]].append(1)
                        #save threshold
                        all_score_thresholds[nonzero_labels[k]].append(current_score)
                   


                except:

                    fn_per_image[nonzero_labels[k]] = len(nonzero_gts[k])



            #calculate false positives, that is boxes that have not been assigned to a gt
            for index, ai in enumerate(assigned_idx):
                #if box has not been assigned

                if ai == 0:

                    #add a false positive to the corresponding class
                    fp_per_image[batch_classes[j][index]] +=1
                    #add this as a non gt
                    is_gt[batch_classes[j][index]].append(0)
                    #append the predicted score to the predicted class
                    all_score_thresholds[batch_classes[j][index]].append(batch_scores[j][index])



            all_tps.append(tp_per_image)
            all_fns.append(fn_per_image)
            all_fps.append(fp_per_image)


    return boxes_per_img , boxes_per_gt, np.stack(all_tps), np.stack(all_fps), np.stack(all_fns), is_gt, all_score_thresholds

def AP( predictions, scores):
    """
    Computes the  average precision per class, the average precision and the interpolated average precision at 11 points
    :param predictions: list of lists of every class with tp, fp and fn. fps are zeros, the others one, indicating this is a ground truth
    :param scores: confidences scores with the same lengths
    :return: mAPs a classes x 2 matrix, first entry is without interpolation.
    The average precision and the interpolated average precision at 11 points
    """

    #recall levels
    recalls = np.arange(0,1.1,0.1)

    #average precisions over all classes
    prec = np.zeros_like(recalls)

    #average interpolated precision over all classes
    iprec =  np.zeros_like(recalls)

    #average precision
    ap = np.zeros( ( len(predictions), 2))

    for i in range(len(predictions)):

        #if this is dummy class with no predictions and gts

        if len(predictions[i]) == 0:
            ap[i,0] = 0
            ap[i,1] = 0

        else:

            #sort zipped lists
            zipped = zip(predictions[i], scores[i])


            spreds_and_scores = sorted(zipped, key=lambda x: x[1], reverse=True)

            #unzip
            spreds, sscores = zip(*spreds_and_scores)

            #get the indices of gts
            npos = [ t[0] for t in enumerate(spreds) if t[1] > 0 ]


            #count gts
            N = len(npos)

            #compute the precisions at every gt
            nprec = np.arange(1,N+1) / (np.array(npos)+1)

            #store the mean
            ap[i, 0] = np.mean(nprec)

            #interpolated precisions
            inprec =  np.zeros_like(nprec)

            #maximum
            mx = nprec[-1]

            inprec[-1] = mx

            #go backwards through precisions and check if current precision is bigger than last max
            for j in range(len(npos)-2, -1, -1):

                if nprec[j] > mx:
                    mx = nprec[j]
                inprec[j] = mx

            #mean of interpolated precisions
            ap[i,1] = np.mean(inprec)


            #get 11 indices
            idx =  (np.concatenate( (np.zeros((1)), np.maximum(np.zeros(10), np.around((N-1)/(10) * np.arange(1,11))-1)))).astype(int)


            iprec += inprec[idx]
            prec += nprec[idx]


    return ap, prec / len(predictions), iprec / len(predictions)



def filter_prediction(boxes, probs, cls_idx, config):
    """Filter bounding box predictions with probability threshold and
    non-maximum supression.
    
    Args:
      boxes: array of [cx, cy, w, h].
      probs: array of probabilities
      cls_idx: array of class indices
    Returns:
      final_boxes: array of filtered bounding boxes.
      final_probs: array of filtered probabilities
      final_cls_idx: array of filtered class indices
    """

    #check for top n detection flags
    if config.TOP_N_DETECTION < len(probs) and config.TOP_N_DETECTION > 0:
      order = probs.argsort()[:-config.TOP_N_DETECTION-1:-1]
      probs = probs[order]
      boxes = boxes[order]
      cls_idx = cls_idx[order]
      
    else:

      filtered_idx = np.nonzero(probs>config.PROB_THRESH)[0]
      probs = probs[filtered_idx]
      boxes = boxes[filtered_idx]
      cls_idx = cls_idx[filtered_idx]
    
    final_boxes = []
    final_probs = []
    final_cls_idx = []

    #go trough classes
    for c in range(config.CLASSES):
      idx_per_class = [i for i in range(len(probs)) if cls_idx[i] == c]

      #do non maximum suppresion
      keep = utils.nms(boxes[idx_per_class], probs[idx_per_class], config.NMS_THRESH)
      for i in range(len(keep)):
        if keep[i]:
          final_boxes.append(boxes[idx_per_class[i]])
          final_probs.append(probs[idx_per_class[i]])
          final_cls_idx.append(c)

    return final_boxes, final_probs, final_cls_idx


def precision(tp,fp):
    """Computes precision for an array of true positives and false positives
    
    Arguments:
        tp {[type]} -- True positives
        fp {[type]} -- False positives
    
    Returns:
        [type] -- Precision
    """

    return tp / (tp+fp+1e-10)


def recall(tp,fn):
    """Computes recall  for an array of true positives and false negatives
    
    Arguments:
        tp {[type]} -- True positives
        fn {function} -- False negatives
    
    Returns:
        [type] -- Recalll
    """
    return tp / (tp+fn+1e-10)



