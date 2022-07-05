# Hand Tracking Module

#import Packages
import cv2
import mediapipe as mp
import time

class hand_detector():
    def __init__(self, STATIC_IMAGE_MODE=False, MAX_NUM_HANDS=2, MODEL_COMPLEXITY=1, MIN_DETECTION_CONFIDENCE=0.5, MIN_TRACKING_CONFIDENCE=0.5):
        self.mode = STATIC_IMAGE_MODE
        self.max_hands = MAX_NUM_HANDS
        self.model_complexity = MODEL_COMPLEXITY
        self.detection_confidence = MIN_DETECTION_CONFIDENCE
        self.tracking_confidence = MIN_TRACKING_CONFIDENCE
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.max_hands, self.model_complexity, self.detection_confidence, self.tracking_confidence)
        self.mpDraw =mp.solutions.drawing_utils
    
    def detect(self, img, draw=True):
        imgRGB =cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:  
                    self.mpDraw.draw_landmarks(img, hand_landmarks,self.mpHands.HAND_CONNECTIONS)
        return img

    def get_absolute_position(self, img, hand_number=0, draw=True):
        landmark_list = []
        x_list = []
        y_list = []
        bounding_box = []
        boundry = 20
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_number]
            for id, lm in enumerate(hand.landmark):
                h, w, c = img.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                x_list.append(cx)
                y_list.append(cy)
                landmark_list.append([id,cx,cy])
                # if draw:
                #     cv2.circle(img, (cx, cy), 15, (255, 0, 0),cv2.FILLED)

            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)
            bounding_box = x_min - boundry, y_min - boundry, x_max + boundry, y_max + boundry

            if draw:
                cv2.rectangle(img, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), (255, 255, 255), 2)

        return landmark_list, bounding_box 
        
    
# def main():
#     cap = cv2.VideoCapture(0)
#     detector = hand_detector(MIN_DETECTION_CONFIDENCE=0.8)
    
#     while True:
#         success, img = cap.read()
#         img = detector.detect(img,)
#         positions, bbox = detector.get_absolute_position(img)
#         # if len(positions) != 0:
#         #     print(positions)
        
        
#         cv2.imshow("Capture",img)
#         cv2.waitKey(1)

    
# if __name__ == "__main__":
#     main()