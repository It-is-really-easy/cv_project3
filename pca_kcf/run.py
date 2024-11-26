import numpy as np
import cv2
import sys
from time import time

import kcftracker

selectingObject = False
initTracking = False
onTracking = False
ix, iy, cx, cy = -1, -1, -1, -1
w, h = 0, 0

inteval = 1
duration = 0.01
first_frame = None

# mouse callback function
def draw_boundingbox(event, x, y, flags, param):
    global selectingObject, initTracking, onTracking, ix, iy, cx, cy, w, h

    if event == cv2.EVENT_LBUTTONDOWN:
        selectingObject = True
        onTracking = False
        ix, iy = x, y
        cx, cy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if selectingObject:
            cx, cy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        selectingObject = False
        if abs(x - ix) > 10 and abs(y - iy) > 10:
            w, h = abs(x - ix), abs(y - iy)
            ix, iy = min(x, ix), min(y, iy)
            initTracking = True
        else:
            onTracking = False

    elif event == cv2.EVENT_RBUTTONDOWN:
        onTracking = False
        if w > 0:
            ix, iy = x - w // 2, y - h // 2
            initTracking = True

if __name__ == '__main__':
    # 检查命令行参数
    if len(sys.argv) == 2:
        cap = cv2.VideoCapture(sys.argv[1])
        inteval = 30
        ret, first_frame = cap.read()
        if not ret:
            print("无法读取视频文件")
            sys.exit(1)

        cv2.imshow('tracking', first_frame)
        cv2.waitKey(0)  # 等待用户完成选择
        first_frame = None  # 选择完成后，清空 first_frame
    else:
        cap = cv2.VideoCapture(0)

    tracker = kcftracker.KCFTracker(True, True, True)  # hog, fixed_window, multiscale
    # 如果使用 hog 特征，您在绘制第一个边界框后会有一个短暂的暂停，这是由于使用了 Numba。

    cv2.namedWindow('tracking')
    cv2.setMouseCallback('tracking', draw_boundingbox)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if first_frame is not None:
            frame = first_frame

        if selectingObject:
            cv2.rectangle(frame, (ix, iy), (cx, cy), (0, 255, 255), 1)
        elif initTracking:
            cv2.rectangle(frame, (ix, iy), (ix + w, iy + h), (0, 255, 255), 2)
            print([ix, iy, w, h])
            tracker.init([ix, iy, w, h], frame)

            initTracking = False
            onTracking = True
        elif onTracking:
            t0 = time()
            boundingbox = tracker.update(frame)
            t1 = time()

            boundingbox = list(map(int, boundingbox))
            print(boundingbox)
            cv2.rectangle(frame, (boundingbox[0], boundingbox[1]), (boundingbox[0] + boundingbox[2], boundingbox[1] + boundingbox[3]), (0, 255, 255), 1)

            duration = 0.8 * duration + 0.2 * (t1 - t0)
            # cv2.putText(frame, 'FPS: ' + str(1 / duration)[:4].strip('.'), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow('tracking', frame)
        c = cv2.waitKey(inteval) & 0xFF
        if c == 27 or c == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()