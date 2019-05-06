import numpy as np
import cv2
import argparse
from binarization_utils import binarize
import time

import threading

#import Object
#import StopLine

from model import NeuralNetwork


#################################################################

from ctypes import *
import math
import random

import time

import io
import socket
import struct
from PIL import Image

import cv2 as cv
import numpy as np




def sample(probs):
    s = sum(probs)
    probs = [a / s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs) - 1


def c_array(ctype, values):
    arr = (ctype * len(values))()
    arr[:] = values
    return arr


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


# lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("/home/pirl/darknet/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)


def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

# 0.5 0.5 0.45
def detect(net, meta, image, thresh=.4, hier_thresh=.4, nms=.35):
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res

####################################################################################



class CollectTrainingBinaryData(object):

    def __init__(self, client, steer):        
        import argparse

        self.client = client        
        self.steer = steer

        #self.args = Namespace(classes='yolov3.txt', config='yolov2-tiny.cfg', image='dog.jpg', weights='yolov2-tiny.weights')
        #self.stopline = StopLine.Stop()


        #self.dect = Object.Object_Detection(self.steer) # hcw

        # model create

        self.model = NeuralNetwork()
        # self.model.load_model(path = 'model_data/190504_v5.h5')
        self.model.load_model(path = 'model_data/posicar_binary_v11.h5')




    def collect(self):


        #################### detection code inserted ################################################



        # function to get the output layer names
        # in the architecture
        def get_output_layers(net):

            layer_names = net.getLayerNames()

            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

            return output_layers

        # function to draw bounding box on the detected object with class name

        def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

            label = str(classes[class_id])

            color = COLORS[class_id]

            cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

            cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        ###############################################################################################

        print("Start video stream")

        stream_bytes = b' '
        test = 0
        cnt = 0
        net = load_net(b"/home/pirl/Desktop/A4_noruway_code/computer/yolo/python/yolo-obj.cfg", b"/home/pirl/Desktop/A4_noruway_code/computer/yolo/python/yolo-obj_400.weights", 0)
        meta = load_meta(b"/home/pirl/Desktop/A4_noruway_code/computer/yolo/cfg/POSLA.data")

        while True :
	        #print("WHY")Image
            stream_bytes += self.client.recv(1024)
            first = stream_bytes.find(b'\xff\xd8')
            last = stream_bytes.find(b'\xff\xd9')

#########################################
                
##########################################






            if first != -1 and last != -1:
                test = test+1
                #print("IN TRY", test)
                jpg = stream_bytes[first:last + 2]
                stream_bytes = stream_bytes[last + 2:]

                gray = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                image = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            #
            #     ################# detection code inserted ##################

                cv.imwrite("../data/picamera/img.jpg", image)

                # start = time.time()

                results = detect(net, meta, b".yolo/data/picamera/img.jpg")
                print(results)
                # finish = time.time()
                #
                # print(finish - start)

                detect_list = []
                img = cv.imread('../data/picamera/img.jpg')

                for cat, score, bounds in results:
                    x, y, w, h = bounds
                    cv.rectangle(img,
                                 (int(x - w / 2), int(y - h / 2)),
                                 (int(x + w / 2), int(y + h / 2)),
                                 (255, 0, 0),
                                 thickness=2)
                    cv.putText(img,
                               str(cat.decode("utf-8")),
                               (int(x - w / 2), int(y + h / 4)),
                               cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255))
                    detect_list.append(cat.decode())

                cv.imshow('dect', img)
                cv.waitKey(1)

                ############################################################

                #print(image)
                #print('type : ',type(image))
                #print('shape : ', image.shape)

################# detection code inserted ##################

                # Width = image.shape[1]
                # Height = image.shape[0]
                # scale = 0.00392

                # read class names from text file
                #classes = None

                # with open('yolov3.txt', 'r') as f:
                #     classes = [line.strip() for line in f.readlines()]
                #
                # # generate different colors for different classes
                # COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
                #
                # # read pre-trained model and config file
                # net = cv2.dnn.readNet('yolov2-tiny.weights', 'yolov2-tiny.cfg')
                #
                # # create input blob
                # blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
                #
                # # set input blob for the network
                # net.setInput(blob)
                #
                # if cnt % 10 == 0:
                #
                #     # run inference through the network
                #     # and gather predictions from output layers
                #     outs = net.forward(get_output_layers(net))
                #
                #     # initialization
                #     class_ids = []
                #     confidences = []
                #     boxes = []
                #     conf_threshold = 0.5
                #     nms_threshold = 0.4
                #
                #     # for each detetion from each output layer
                #     # get the confidence, class id, bounding box params
                #     # and ignore weak detections (confidence < 0.5)
                #
                #     for out in outs:
                #         for detection in out:
                #             scores = detection[5:]
                #             class_id = np.argmax(scores)
                #             confidence = scores[class_id]
                #             if confidence > 0.5:
                #                 center_x = int(detection[0] * Width)
                #                 center_y = int(detection[1] * Height)
                #                 w = int(detection[2] * Width)
                #                 h = int(detection[3] * Height)
                #                 x = center_x - w / 2
                #                 y = center_y - h / 2
                #                 class_ids.append(class_id)
                #                 confidences.append(float(confidence))
                #                 boxes.append([x, y, w, h])
                #
                #     # apply non-max suppression
                #
                #     indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
                #
                #     # go through the detections remaining
                #     # after nms and draw boundng box
                #
                #     for i in indices:
                #         i = i[0]
                #         box = boxes[i]
                #         x = box[0]
                #         y = box[1]
                #         w = box[2]
                #         h = box[3]
                #         draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w),
                #                         round(y + h))

                ############################################################




                #cv2.imshow('image',image)










#################################
#		inserted code start (by hcw)
#################################


##################################
#		inserted code end
##################################


                    #print("RGB: ",rgb)

                    #cv2.imshow('Origin', rgb)
                    #cv2.waitKey(1)
                    #cv2.imshow('GRAY', image)
                    #cv2.imshow('roi', roi)
                    #print("hihi3")
                    # reshape the roi image into a vector
                # start = time.time()
                img = binarize(img=image, verbose=True)
                roi = img[120:240, :]
                
                image_array = np.reshape(roi, (-1, 120, 320, 1))
                    #print("hihi2")


                    # neural network makes prediction
                self.steer.Set_Line(self.model.predict(image_array))
                    #self.steer.Set_Stopline(self.stopline.GetStopLine(roi2))
                    #print(self.dect.Detection(rgb))
                    #print("hihi")

                    #self.dect.Detection(rgb)
                    #print("hi")

                self.steer.Control()
                cv2.imshow('roi',roi)
                # print("time::",time.time()-start)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()	 # inserted (by hcw)


























