import cv2
import mediapipe as mp
import autopy
import numpy as np

cap = cv2.VideoCapture(0)
initHand = mp.solutions.hands
mainHand = initHand.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
draw = mp.solutions.drawing_utils
wScr, hScr = autopy.screen.size()
pX, pY = 0, 0
cX, cY = 0, 0

def handLandmarks(colorImg):
    landmarkList = []
    landmarkPositions = mainHand.process(colorImg)
    landmarkCheck = landmarkPositions.multi_hand_landmarks
    if landmarkCheck:
        for hand in landmarkCheck:
            for index, landmark in enumerate(hand.landmark):
                draw.draw_landmarks(colorImg, hand, initHand.HAND_CONNECTIONS)
                h, w, c = colorImg.shape
                centerX, centerY = int(landmark.x * w), int(landmark.y * h)
                landmarkList.append([index, centerX, centerY])
    return landmarkList

def fingers(landmarks):
    fingerTips = []
    tipIds = [4, 8, 12, 16, 20]
    for id in range(0, 5):
        if landmarks[tipIds[id]][2] < landmarks[tipIds[id] - 2][2]:
            fingerTips.append(1)
        else:
            fingerTips.append(0)
    return fingerTips

while True:
    check, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lmList = handLandmarks(imgRGB)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        finger = fingers(lmList)

        if finger[1] == 1 and finger[2] == 1:
            x3 = np.interp(x1, (75, 640 - 75), (0, wScr))
            y3 = np.interp(y1, (75, 480 - 75), (0, hScr))

            cX = pX + (x3 - pX) / 5
            cY = pY + (y3 - pY) / 5

            autopy.mouse.move(wScr - cX, cY)
            pX, pY = cX, cY

            autopy.mouse.click()

        elif finger[1] == 1 and finger[2] == 0:
            x3 = np.interp(x1, (75, 640 - 75), (0, wScr))
            y3 = np.interp(y1, (75, 480 - 75), (0, hScr))

            cX = pX + (x3 - pX) / 5
            cY = pY + (y3 - pY) / 5

            autopy.mouse.move(wScr - cX, cY)
            pX, pY = cX, cY

        else:
            autopy.mouse.toggle(down=False)

    cv2.imshow("webcam", imgRGB)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
