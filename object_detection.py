import numpy
import cv2
from cv2 import *




cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
cap.set(1, 70)

classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    classIds, conf, bbox = net.detect(img, confThreshold=0.45)
    print(classIds, bbox)

    if len(classIds) != 0:
        for classId, confidence, bbox in zip(classIds.flatten(), conf.flatten(), bbox):
            cv2.rectangle(img, bbox, color=(0, 255, 0), thickness=2)
            cv2.putText(img, classNames[classId - 1].upper(), (bbox[0] + 10, bbox[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (0, 255, 0), 2)
            cv2.putText(img, str(round(confidence * 100, 2)), (bbox[0] + 200, bbox[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (0, 255, 0), 2)
    cv2.imshow("Output", img)
    cv2.waitKey(1)
    
    
cap.release()
cv2.destroyAllWindows()
