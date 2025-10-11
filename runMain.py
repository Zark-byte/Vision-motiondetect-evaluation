import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import matplotlib.pyplot as plt
# import json as js

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

ExampleList=['example1(3).mp4','example2(3).mp4','example3(3).mp4','example4(1).mp4','example5(1).mp4','example6(1).mp4','example7(2).mp4','example9(2).mp4','example10(2).mp4']

def calculate_angle(p1, p2, p3):
    vector1 = (p2[0] - p1[0], p2[1] - p1[1])
    vector2 = (p3[0] - p2[0], p3[1] - p2[1])

    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]

    magnitude1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
    magnitude2 = math.sqrt(vector2[0]**2 + vector2[1]**2)

    cos_angle = dot_product / (magnitude1 * magnitude2)

    angle = math.acos(cos_angle)

    angle_degrees = 180-math.degrees(angle)

    return angle_degrees

for j in range(9):
    lmlist_24=[]
    lmlist_12=[]
    lmlist_14=[]
    lmlist_16=[]
    angle1=[]
    angle2=[]
    time_second=[]
    cap = cv2.VideoCapture(ExampleList[j])
    while True:
        success, img = cap.read()
        if not success:
            break
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if results.pose_landmarks:
            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            for id, lm in enumerate(results.pose_landmarks.landmark):
                current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                seconds = current_frame / fps
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id ==24:
                    lmlist_24.append([seconds,id,cx,cy])
                if id == 12:
                    lmlist_12.append([seconds,id,cx,cy])
                if id == 14:
                    lmlist_14.append([seconds,id,cx,cy])
               
                if id == 16:
                    lmlist_16.append([seconds,id,cx,cy])
    
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        cv2.imshow("Image", img)
        cv2.waitKey(10)
    for i in range(len(lmlist_12)):
        angle1.append(
        calculate_angle((lmlist_24[i][2],lmlist_24[i][3]),(lmlist_12[i][2],lmlist_12[i][3]),(lmlist_14[i][2],lmlist_14[i][3]))
        )
        angle2.append(calculate_angle((lmlist_12[i][2],lmlist_12[i][3]),(lmlist_14[i][2],lmlist_14[i][3]),(lmlist_16[i][2],lmlist_16[i][3])))
        time_second.append(lmlist_12[i][0])

    plt.scatter(time_second,angle1)
    plt.scatter(time_second,angle2)
    plt.plot(time_second,angle1)
    plt.plot(time_second,angle2)
    plt.xlabel("time(s)",fontsize=20)
    plt.ylabel("angle(degree)",fontsize=20)
    plt.show()
    plt.plot()

# with open('Label.json', 'w', encoding='utf-8') as f:
#     js.dump(X_Lable, f, ensure_ascii=False, indent=4)
