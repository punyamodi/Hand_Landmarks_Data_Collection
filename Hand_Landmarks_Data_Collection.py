import cv2
import mediapipe as mp
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
print("********************Welcome to Hand Landmarks Data Collection*************************")

print("press control c to exit program")
print("file will be saved at the location of program")

name=input("Enter Csv file name, to be saved:")

l=[]
l.append("type")
for i in range(21):
    for p in ["X","Y"]:
        str1=str(i)+p
        l.append(str1)
df=pd.DataFrame(columns=l)

def findHands(results, img, draw=True, flipType=True):
    allHands = []
    h, w, c = img.shape
    if results.multi_hand_landmarks:
        for handType, handLms in zip(results.multi_handedness, results.multi_hand_landmarks):
            myHand = {}
            ## lmList
            mylmList = []
            xList = []
            yList = []
            for id, lm in enumerate(handLms.landmark):
                px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * c)
                mylmList.append([px, py])

            myHand["lmList"] = mylmList

            if flipType:
                if handType.classification[0].label == "Right":
                    myHand["type"] = "Left"
                else:
                    myHand["type"] = "Right"
            else:
                myHand["type"] = handType.classification[0].label
            allHands.append(myHand)

            return allHands



def hand():
    cap = cv2.VideoCapture(0)

    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.7)
    mpDraw = mp.solutions.drawing_utils

    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img2 = cv2.flip(img, 1)
        results = hands.process(img2)

        try:
            f=int(input)
            if f==1:
                break
        except:
            pass

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                hand_data = findHands(results, img2)  # with draw
                # hands = findHands(results, img2, draw=False)  # without draw
                if hand_data:
                    # Hand 1
                    hand = hand_data[0]
                    side = hand["type"]
                    print(side)
                    row=[side]
                    lmList = hand["lmList"]  # List of 21 Landmark points
                    for lm in lmList:
                        x, y= lm
                        row.append(x)
                        row.append(y)
                        print(f"X: {x}, Y: {y}")
                    print("**************************************************")
    
                    df.loc[len(df)]=row

                mp.solutions.drawing_utils.draw_landmarks(img2, handLms, mp.solutions.hands.HAND_CONNECTIONS)

        cv2.imshow("Image", img2)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

try:
    hand()
except KeyboardInterrupt:
    df.to_csv(name+".csv")
    exit()


## makes program go faster
with ThreadPoolExecutor() as executor:
    executor.submit(hand)



