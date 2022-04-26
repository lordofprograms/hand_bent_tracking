import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math


class HandBentStatus:
    def __init__(self, camera_num=0, detection_conf=0.7, tracking_conf=0.7):
        self.cap = cv2.VideoCapture(camera_num)
        self.detector = htm.HandDetector(detection_conf=detection_conf, track_conf=tracking_conf)

        self.minVal = 0
        self.maxVal = 100
        # if needed can change middle to little finger base, 9 to 17
        self.thumb_finish_fb = 9
        self.min_tip_dict = {4: 0.1, 8: 1.7, 12: 1.5, 16: 1.7, 20: 2}
        self.max_tip_dict = {4: 4}

        # TODO move to class for normal usage of heuristic for checking diff and make less updates
        vals_amount = 5
        self.prev_len_arr = [0.0] * vals_amount
        self.prev_percent_arr = [0.0] * vals_amount
        self.prev_ref_point = 0.0
        self.name_mappings = {0: "thumb", 1: "index", 2: "middle", 3: "ring", 4: "little"}
        self.idx = 0

    def fingers_bent_img(self, draw_landmarks: bool = True, return_step=15, frame_width=None, frame_height=None):
        """
        Calculate bent percent by hand landmarks position
        :param draw_landmarks: show green lines that tracks and highlight hand
        :param return_step: return all finger bent percents or skip some frames
        :param frame_width: screen width
        :param frame_height: screen height
        :return: cv2 img, list of bent percents
        """

        if frame_width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        if frame_height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

        success, cv_img = self.cap.read()
        cv_img = self.detector.find_hands(cv_img, draw=draw_landmarks)
        lm_list = self.detector.find_position(cv_img, draw=False)

        self.idx += 1
        if len(lm_list) != 0 and (return_step == 0 or self.idx % return_step == 0):
            fingers_stats = self.fingers_bent_percent(lm_list)
            # cv2.putText(img, f"{fingers_stats}", (640, 640), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)
            self.idx = 1
            fingers_stats.reverse()
            return cv_img, fingers_stats
        else:
            return cv_img, None

    def selected_tip_demo(self, lm_list, tip_id: int = 8):
        # tip_id = 8
        x1, y1 = lm_list[tip_id][1], lm_list[tip_id][2]
        if tip_id != 4:
            x2, y2 = lm_list[tip_id - 3][1], lm_list[1][2]
        else:
            x2, y2 = lm_list[self.thumb_finish_fb][1], lm_list[self.thumb_finish_fb][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        # print(f"length = {length}, prev length was ", prev_len)
        # prev_len = length

        # length_arr.append(length)
        # print(f"min: {np.amin(length_arr)}, max: {np.amax(length_arr)} and arr size = {len(length_arr)}")

        # print(length)

        # Hand range 50 - 300
        # Volume Range -65 - 0

        # rel_hand_max_range = (abs(lmList[tip_id][2] - lmList[tip_id - 2][2]) +
        #                       abs(lmList[1][2] - lmList[tip_id - 3][2])) * 1.3
        # rel_hand_min_range = abs(lmList[tip_id - 1][2] - lmList[tip_id - 2][2]) * 2

        # print("range min/max", rel_hand_min_range, rel_hand_max_range)

        ref_point = abs(lm_list[2][2] - lm_list[1][2])
        # print(f"ref point = {ref_point}, prev ref point was {prev_ref_point}")
        # prev_ref_point = ref_point

        rel_hand_max_range = ref_point * self.max_tip_dict.get(tip_id, 5.5)
        rel_hand_min_range = ref_point * self.min_tip_dict.get(tip_id)

        # # config for small finger
        # rel_hand_max_range = ref_point * 5.5
        # rel_hand_min_range = ref_point * 2

        vol = np.interp(length, [rel_hand_min_range, rel_hand_max_range], [self.minVal, self.maxVal])
        volBar = np.interp(length, [rel_hand_min_range, rel_hand_max_range], [400, 150])
        # volPer = np.interp(length, [rel_hand_min_range, rel_hand_max_range], [0, 100])
        volPer = vol
        print("experiment: ", int(length), vol, lm_list[tip_id][3])

        if length < 50:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

        cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
        cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                    1, (255, 0, 0), 3)

    def fingers_bent_percent(self, lm_list) -> list:
        # TODO add thumb drive
        ref_point = abs(lm_list[2][2] - lm_list[1][2])
        # print(f"test point finger base = {abs(lmList[5][1] - lmList[17][1])} and palm base test = "
        #       f"{abs(lmList[17][2] - lmList[0][2])}")
        # TODO maybe do something when we show front side of the palm
        new_prev_diff = abs(ref_point - self.prev_ref_point)
        if new_prev_diff > 30:
            print(f"ref point = {ref_point}, prev ref point was {self.prev_ref_point}, |diff| = {new_prev_diff}")
            # TODO test it more to make it more stable
            self.prev_ref_point = ref_point
            # print("ref point size = ", ref_point)
        else:
            ref_point = self.prev_ref_point

        fingers_percent = [0]
        tip_id_list = [4, 8, 12, 16, 20]
        for i, tip_id in enumerate(tip_id_list):
            x1, y1 = lm_list[tip_id][1], lm_list[tip_id][2]

            if tip_id != 4:
                x2, y2 = lm_list[tip_id - 3][1], lm_list[1][2]
            else:
                x2, y2 = lm_list[self.thumb_finish_fb][1], lm_list[self.thumb_finish_fb][2]

            length = math.hypot(x2 - x1, y2 - y1)
            # print(f"length = {length}, prev len was {self.prev_len_arr[i]}, diff = {length - self.prev_len_arr[i]}")
            # self.prev_len_arr[i] = length

            # TODO maybe use another ref points or another solution for calculating relation values
            rel_hand_max_range = ref_point * self.max_tip_dict.get(tip_id, 5.5)
            rel_hand_min_range = ref_point * self.min_tip_dict.get(tip_id)

            open_percent = np.interp(length, [rel_hand_min_range, rel_hand_max_range], [self.minVal, self.maxVal])
            bent_percent = round(100 - open_percent)

            # TODO test it
            if abs(bent_percent - self.prev_percent_arr[i]) > 5:
                print(f"{self.name_mappings.get(i)}({i}) bent % ={bent_percent}, prev % was {self.prev_percent_arr[i]},"
                      f"diff = {abs(bent_percent - self.prev_percent_arr[i])}")
                self.prev_percent_arr[i] = bent_percent
            else:
                bent_percent = self.prev_percent_arr[i]

            fingers_percent.append(bent_percent)
        return fingers_percent


if __name__ == '__main__':
    finger_regressor = HandBentStatus()
    pTime = 0
    while True:
        img, bent_stats = finger_regressor.fingers_bent_img(return_step=0)
        if bent_stats:
            print(bent_stats)

        # redundant part for integration
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                    1, (255, 0, 0), 3)

        cv2.imshow("Img", img)
        cv2.waitKey(1)
