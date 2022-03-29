#import Packages
import cv2
import mediapipe as mp

#capture video on camera 0
cap = cv2.VideoCapture(0)

#mediapipe hand detection imports
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw =mp.solutions.drawing_utils

#defining variables for FPS Counter
pTime = 0
cTime = 0

#displaying video
while True:
    #read captue
    success, img = cap.read()
    
    #convert color mode and pass to mediapipe to process
    imgRGB =cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    #get results from mediapipe and drawing it on the image
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = img.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                
            mpDraw.draw_landmarks(img, hand_landmarks,mpHands.HAND_CONNECTIONS)
    
    #calculate FPS
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime = cTime
    
    #Drawing Fps results o the screen
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 3)
    
    #Dysplay the video
    cv2.imshow("Capture",img)
    cv2.waitKey(1)