# Data Gathering Module
import cv2
from HTM import hand_detector
import numpy as np
import math

img_size = 300

detector = hand_detector(MIN_DETECTION_CONFIDENCE=0.8,MAX_NUM_HANDS=1)
cap = cv2.VideoCapture(0)

path = "Data/C"
counter = 0

while True:
    success, img = cap.read()
    img = detector.detect(img)
    positions, bounding_box = detector.get_absolute_position(img)
    try:
        if bounding_box:
            img_background = np.ones((img_size,img_size,3),np.uint8)*255
            img_croped = img[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2]]

            height, width, _ = img_croped.shape
            
             
            aspect_ratio = height/width
            if aspect_ratio >1:
                const = img_size/height
                new_width = math.ceil(const*width)
                img_resized = cv2.resize(img_croped, (new_width,img_size))

                height_resized, width_resized, _ = img_resized.shape
                boundry_width = math.ceil((img_size-new_width)/2)

                img_background[0:height_resized, boundry_width:width_resized+boundry_width] = img_resized
            else:
                const = img_size/width
                new_height = math.ceil(const*height)
                img_resized = cv2.resize(img_croped, (img_size, new_height))

                height_resized, width_resized, _ = img_resized.shape
                boundry_height = math.ceil((img_size-new_height) / 2)

                img_background[boundry_height:height_resized+boundry_height, 0:width_resized] = img_resized

            cv2.imshow("Image_Croped", img_background)
            cv2.waitKey(1)
    except:
        print("Out of Range")

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f"{path}/C_{counter:003}.jpg", img_background)
        print(counter)
    if key == ord("q"):
        exit()