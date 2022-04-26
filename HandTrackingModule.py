import math
import time

import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

import kalman


class HandDetector:
    def __init__(self, mode=False, max_hands=1, detection_conf=0.7, track_conf=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_conf = detection_conf
        self.track_conf = track_conf
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.detection_conf, self.track_conf)
        self.mp_draw = mp.solutions.drawing_utils

        self.tip_ids = [4, 8, 12, 16, 20]
        self.name_mappings = {0: "thumb", 1: "index", 2: "middle", 3: "ring", 4: "little"}
        self.lm_list = []
        self.first_det = True

        self.one_euro_filter_x = []
        self.one_euro_filter_y = []
        # self.one_euro_filter_z = []
        self.fps = 0

    def filter(self, landmarks):
        filtered_landmarks = []
        for i, landmark in enumerate(landmarks.landmark):
            # x = self.one_euro_filter_x[i](fps, landmark.x)
            #
            # y = self.one_euro_filter_x[i](fps,landmark.y)
            # #print(x)
            # z = self.one_euro_filter_x[i](fps, landmark.z)
            self.one_euro_filter_x[i].update([landmark.x])
            x = self.one_euro_filter_x[i].get_results()[0]
            self.one_euro_filter_y[i].update([landmark.y])
            y = self.one_euro_filter_y[i].get_results()[0]
            # print(x)
            # self.one_euro_filter_z[i].update([landmark.z])
            # z = self.one_euro_filter_z[i].get_results()[0]
            # flm = landmark_pb2.NormalizedLandmark(x=x, y=y, z=z)
            filtered_landmarks.append(landmark_pb2.NormalizedLandmark(x=x, y=y))
        filtered_landmark_list = landmark_pb2.NormalizedLandmarkList(landmark=filtered_landmarks)
        return filtered_landmark_list

    def find_hands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    # print(f'landmark:{handLms.landmark[0].x}')

                    if self.first_det:
                        self.first_det = False
                        min_cutoff = 0.1
                        beta = 0.9
                        for i in range(21):
                            # self.one_euro_filter_x.append(one_euro_filter.OneEuroFilter(0, handLms.landmark[i].x, min_cutoff=min_cutoff, beta=beta))
                            # self.one_euro_filter_y.append(
                            #     one_euro_filter.OneEuroFilter(0 , handLms.landmark[i].y, min_cutoff=min_cutoff, beta=beta))
                            # self.one_euro_filter_z.append(
                            #     one_euro_filter.OneEuroFilter(0, handLms.landmark[i].z, min_cutoff=min_cutoff, beta=beta))
                            # self.one_euro_filter_x.append(
                            #     one_euro_filter.MeanFilter(0, handLms.landmark[i].x))
                            # self.one_euro_filter_y.append(
                            #     one_euro_filter.MeanFilter(0, handLms.landmark[i].y))
                            # self.one_euro_filter_z.append(
                            #     one_euro_filter.MeanFilter(0, handLms.landmark[i].z))
                            self.one_euro_filter_x.append(kalman.Stabilizer([handLms.landmark[i].x], 1))
                            self.one_euro_filter_y.append(kalman.Stabilizer([handLms.landmark[i].y], 1))
                            # self.one_euro_filter_z.append(kalman.Stabilizer([handLms.landmark[i].z], 1))


                    else:
                        self.fps = self.fps + 1
                        # # print(f'fps: {self.fps}')
                        # print(handLms.landmark[0].x)
                        handLms = self.filter(handLms)
                        # print(f'after:{handLms.landmark[0].x}')
                        # print(f'after 1:{handLms.landmark[1].x}')

                    self.mp_draw.draw_landmarks(img, handLms, self.mp_hands.HAND_CONNECTIONS)

        return img

    def find_position(self, img, handNo=0, draw=True):

        lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for tid, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print("Z axis = ", lm.z, int(lm.z * w))
                lmlist.append([tid, cx, cy])
                if draw:
                    cv2.putText(img, str(tid), (cx, cy), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 255, 0), 2)
                    cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)
        return lmlist

    def find_raw_position(self, img, hand_no=0, draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[hand_no]
            for tid, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print("Z axis = ", lm.z, int(lm.z * w))
                lmlist.append([tid, lm.x, lm.y, lm.z])
                if draw:
                    cv2.putText(img, str(tid), (cx, cy), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 255, 0), 2)
                    cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)
        return lmlist

    def fingers_status(self, lm_list):
        self.lm_list = lm_list

        fingers = []
        if self.results.multi_hand_landmarks:
            myHandType = self.hand_type()
            fingers.append(0)

            # self.mpDraw.plot_landmarks(
            #     lm_list, self.mpHands.HAND_CONNECTIONS, azimuth=5)

            # # Thumb
            # if myHandType == "Right":
            #     if self.lm_list[self.tip_ids[0]][0] > self.lm_list[self.tip_ids[0] - 1][0]:
            #         fingers.append(1)
            #     else:
            #         fingers.append(0)
            # else:
            #     if self.lm_list[self.tip_ids[0]][0] < self.lm_list[self.tip_ids[0] - 1][0]:
            #         fingers.append(1)
            #     else:
            #         fingers.append(0)

            print("index_points = ", [self.lm_list[8][1:], self.lm_list[7][1:], self.lm_list[6][1:], self.lm_list[5][1:]])
            print("middle_points = ", [self.lm_list[12][1:], self.lm_list[11][1:], self.lm_list[10][1:], self.lm_list[9][1:]])
            print("ring_points = ", [self.lm_list[16][1:], self.lm_list[15][1:], self.lm_list[14][1:], self.lm_list[13][1:]])
            print("little_points = ", [self.lm_list[20][1:], self.lm_list[19][1:], self.lm_list[18][1:], self.lm_list[17][1:]])
            print("lowest point = ", [self.lm_list[0][1:]])

            # thumb and 4 Fingers
            for tip_id in range(0, 5):
                # if self.tip_ids[tip_id] == 8:
                #     if self.lm_list[8][1] <= self.lm_list[7][1]:
                #         print("Second index point is higher!")

                # # using palm base point and tip and base points of the finger
                # bent_percent = self.__radian_slope2percent(self.lm_list[self.tip_ids[tip_id]][1:],
                #                                            self.lm_list[self.tip_ids[tip_id - 3]][1:],
                #                                            self.lm_list[0][1:])

                # # using 1st, 3rd and 4th points of the finger
                # bent_percent = self.__radian_slope2percent(self.lm_list[self.tip_ids[tip_id]][1:],
                #                                            self.lm_list[self.tip_ids[tip_id - 2]][1:],
                #                                            self.lm_list[self.tip_ids[tip_id - 3]][1:])

                # # using sequence finger segments calc
                # bent_percent = self.__finger_bent_percent([self.lm_list[tip_id - i][1:] for i in range(0, 4)])
                bent_percent = self.__finger_bent_percent([self.lm_list[self.tip_ids[tip_id]][1:],
                                                           self.lm_list[self.tip_ids[tip_id - 1]][1:],
                                                           self.lm_list[self.tip_ids[tip_id - 2]][1:],
                                                           self.lm_list[self.tip_ids[tip_id - 3]][1:]])
                print(f"{self.name_mappings.get(tip_id)} finger bent percent = {round(bent_percent, 2)} %")

                # if self.lm_list[self.tip_ids[tip_id]][1] < self.lm_list[self.tip_ids[tip_id] - 2][1]:
                #     fingers.append(1)
                # else:
                #     fingers.append(0)
        return fingers

    def __finger_bent_percent(self, finger_points):
        v1, v2, v3, v4 = finger_points
        m1, m2, m3 = self.__slope(v1, v2), self.__slope(v2, v3), self.__slope(v3, v4)
        first_angle, second_angle = self.__slope_angle(m1, m2), self.__slope_angle(m2, m3)
        print(first_angle, second_angle)
        angle_sum = first_angle + second_angle
        angle_sum = 90 if angle_sum > 90 else angle_sum
        return (1 - math.sin((angle_sum / 180) * math.pi)) * 100

    def __radian_slope2percent(self, v1, v2, v3):
        m1 = self.__slope(v1, v2)
        m2 = self.__slope(v2, v3)
        if m1 == 0 or m2 == 0:
            print("slope is zero")
            return 0
        else:
            radian = self.__slope_radian(m1, m2)
            # print("not zero", (radian * 180) / math.pi,  math.sin(radian), (1 - math.sin(radian)) * 100, "%")
            # return (1 - math.sin(radian)) * 100
            # TODO try add for some fingers logic with 1 - x and select math function
            print(math.cos(radian), math.sin(radian), math.tan(radian), math.tanh(radian))
            return math.cos(radian) * 100

    @staticmethod
    def __slope(v1, v2):
        try:
            return (v2[1] - v1[1]) / (v2[0] - v1[0])
        except ZeroDivisionError:
            print("slope error")
            return 0

    @staticmethod
    def __slope_angle(m1, m2):
        # Store the tan value  of the angle
        angle = abs((m2 - m1) / (1 + m1 * m2))

        # Calculate tan inverse of the angle
        rad = math.atan(angle)

        # Convert the angle from
        # radian to degree
        val = (rad * 180) / math.pi



        # Print the result
        return val

    @staticmethod
    def __slope_radian(m1, m2):
        # Store the tan value  of the angle
        angle = abs((m2 - m1) / (1 + m1 * m2))

        # Calculate tan inverse of the angle
        rad = math.atan(angle)

        # Convert the angle from
        # radian to degree
        # val = (rad * 180) / math.pi

        # Print the result
        return rad

    def hand_type(self):
        """
        Check whether the incoming hand is left or right
        : return: "Right" or "Left"
        """
        if self.results.multi_hand_landmarks:
            if self.lm_list[17][0] < self.lm_list[5][0]:
                return "Right"
            else:
                return "Left"


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        lmlist = detector.find_position(img)
        if len(lmlist) != 0:
            detector.fingers_status(lmlist)
            # print(fingers_status)
            # print(lmlist[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


# if __name__ == "__main__":
#     main()
