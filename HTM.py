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

    def get_absolute_position (self, img, hand_number=0, draw=False):
        landmark_list = []
        
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_number]
            for id, lm in enumerate(hand.landmark):
                h, w, c = img.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                landmark_list.append([id, cx,cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 0),cv2.FILLED)
                
        return landmark_list
        
    
def main():
    cap = cv2.VideoCapture(0)
    detector = hand_detector()
    
    while True:
        success, img = cap.read()
        img = detector.detect(img,)
        positions = detector.get_absolute_position(img)
        if len(positions) != 0:
            print(positions[8])
        
        
        cv2.imshow("Capture",img)
        cv2.waitKey(1)

    
if __name__ == "__main__":
    main()