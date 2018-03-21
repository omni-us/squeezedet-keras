# Project: squeezeDetOnKeras
# Filename: predict.py
# Author: Christopher Ehmann
# Date: 30.01.18
# Organisation: searchInk
# Email: christopher@searchink.com


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import traceback
import cv2
import enum
import numpy as np
import tensorflow as tf
import pagexml
import keras.backend as K
from operator import itemgetter
from main.model.squeezeDet import SqueezeDet
from main.config.create_config import  load_dict

model_config = cfg['model']
model_threshold = cfg['detection']['threshold']


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
        'checkpoint', model_config['path'],
        """Path to the model parameter file.""")





#loads model, don't know if it needs a session

def load_model(config, weights):
    '''
    Loads the model from a give path
    :param model_path:
    :return: model
    '''
    try:

            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            K.set_session(sess)

            config.BATCH_SIZE = 1

            model = SqueezeDet(config)
            model.load_weights(weights)

            #possibly we have to compile the model?
            # model = model.compile()
    except Exception as e:
        raise e



    return model,config,sess






def predict(images, page_xml, model, mc, sess):
    '''
    :param image_list:
    :param page_xml:
    :return:
'''

    try:
        # output = ""
        pxml = pagexml.PageXML()
        pxml.loadXmlString(page_xml.read())
        # print (images)
        for page_number, img in enumerate(images):
            page_id = int(img.filename[:-4][6:])
            page = np.asarray(bytearray(img.read()), dtype=np.uint8)

            #load image and
            im = cv2.imdecode(page,1)
            im = im.astype(np.float32, copy=False)
            height, width = im.shape[:2]
            im = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))
            scale_x = height/mc.IMAGE_HEIGHT
            scale_y = width/mc.IMAGE_WIDTH

            input_image = im - mc.BGR_MEANS


            model = load_model(mc, weights)

            # Detect
            det_boxes, det_probs, det_class = sess.run(
                [model.det_boxes, model.det_probs, model.det_class],
                feed_dict={model.image_input: [input_image]})

            # Filter
            final_boxes, final_probs, final_class = model.filter_prediction(
                det_boxes[0], det_probs[0], det_class[0])

            keep_idx = [idx for idx in range(len(final_probs)) \
                        if final_probs[idx] > float(model_threshold)]

            final_boxes = [final_boxes[idx] for idx in keep_idx]
            final_probs = [final_probs[idx] for idx in keep_idx]
            final_class = [final_class[idx] for idx in keep_idx] # Class name not used for now for text regions
            transformed_box = []
            sorted_conf = []

            #transforming all bboxes from x,y,w,h to xmin,ymin,xmax,ymax and also streching all boxes to end of width of the pages.
            for it, boxes in enumerate(final_boxes):
                temp_box = util.bbox_transform([boxes[0], boxes[1], boxes[2], boxes[3]])
                temp_box[0] = float(0)
                temp_box[2] = float(mc.IMAGE_WIDTH)
                transformed_box.append(temp_box)
                sorted_conf.append(final_probs[it])

            #sort the list with y min value from transformed box
            sort_boxes = sorted(enumerate(transformed_box), key=itemgetter(1))

            # sorting confidence values from sort_boxes values
            for it, boxes in enumerate(sort_boxes):
                sorted_conf[it]=final_probs[boxes[0]]

            # write page xml
            for it, boxes in enumerate(sort_boxes):
                page = pxml.selectNth("//_:Page",page_id) # Page number here starts from 0 and not 1
                pos_id = "page"+str(page_id+1)+"_pos"+str(int(it)+1) # +1 for page id as it should start from 1 and not 0
                # print (pos_id)
                reg = pxml.addTextRegion(page,pos_id)
                conf = pagexml.ptr_double()
                conf.assign(float(sorted_conf[it]))
                # scale all the values from resized to original size as the image
                xmin = (float(boxes[1][0]))*scale_y
                ymin =(float(boxes[1][1]))*scale_x
                width =  (float(boxes[1][2])-float(boxes[1][0]))*scale_y
                height =(float(boxes[1][3])-float(boxes[1][1]))*scale_x

                pxml.setCoordsBBox(reg, xmin, ymin, width, height, conf)

        output = pxml.toString()
        return output
        # if output:
        #     return output
        # else:
        #     return
    except Exception as e:
        raise e
        #print(traceback.format_exc(e))

def image_demo():
  """Detect image."""


def main(argv=None):

  if not tf.gfile.Exists(FLAGS.out_dir):
    tf.gfile.MakeDirs(FLAGS.out_dir)
  image_demo()

if __name__ == '__main__':
    tf.app.run()