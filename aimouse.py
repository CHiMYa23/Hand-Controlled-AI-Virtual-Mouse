import cv2
import numpy as np
import HandTrackingModule as htm
import  time
import autopy

wcam, hcam = 648, 480
frameR = 100
smoothening = 7
pTime = 0
plocX, plocY = 0 , 0
cloX, clocY = 0 , 0
cap = cv2.VideoCapture(0)
cap.set(3,wcam)
cap.set(4,hcam)

detector = htm.handDetector(maxHands=1)
wscr,hscr = autopy.screen.size()
print(wscr,hscr)

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    if len(lmList) != 0:
        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]
        print(x1,y1,x2,y2)
        fingers = detector.fingersUp()
        print(fingers)
        cv2.rectangle(img, (frameR, frameR), (wcam - frameR, hcam - frameR), (255, 0, 255), 2)
        if fingers[1]==1 and fingers[2]==0:

            x3 = np.interp(x1, (frameR,wcam-frameR),(0,wscr))
            y3 = np.interp(y1, (frameR, hcam-frameR), (0, hscr))
            cloX = plocX + (x3 - plocX) / smoothening
            cloY = plocY + (y3 - plocY) / smoothening
            autopy.mouse.move(wscr-cloX, cloY)
            cv2.circle(img,(x1,y1),15,(255,0,255),cv2.fILLED)
            plocX,plocY = cloX , cloY
            if fingers[1] == 1 and fingers[2] == 1:
                length,img, lineinfo = detector.findDistance(8,12,img)
                print(length)
                if length < 40:
                    cv2.circle(img,(lineinfo[4],lineinfo[5]),15,(0,255,0),cv2.fILLED)
                    autopy.mouse.click()

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)),(20,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
    cv2.imshow('Image',img)
    cv2.waitKey(1)