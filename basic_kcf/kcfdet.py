import argparse
import cv2
import numpy as np
from numpy.fft import fft2, ifft2, fftshift
from HOG import HOG

class Tracker:
    def __init__(self):
        # 超参数设置
        self.max_patch_size = 256
        self.padding = 2.5
        self.sigma = 0.6
        self.lambdar = 0.0001
        self.update_rate = 0.012
        self.gray_feature = False
        self.debug = False

        # 算法变量定义
        self.scale_h = 0.
        self.scale_w = 0.

        self.ph = 0
        self.pw = 0
        self.hog = HOG((self.pw, self.ph))
        self.alphaf = None
        self.x = None
        self.roi = None
        self.first_frame_flag = True  # 新增的标志变量，用于判断是否是第一次读取帧

    def first_frame(self, image, roi):
        """
        对视频的第一帧进行标记，更新tracer的参数
        :param image: 第一帧图像
        :param roi: 第一帧图像的初始ROI元组
        :return: None
        """
        x1, y1, w, h = roi
        cx = x1 + w // 2
        cy = y1 + h // 2
        roi = (cx, cy, w, h)

        # 确定Patch的大小，并在此Patch中提取HOG特征描述子
        scale = self.max_patch_size / float(max(w, h))
        self.ph = int(h * scale) // 4 * 4 + 4
        self.pw = int(w * scale) // 4 * 4 + 4
        self.hog = HOG((self.pw, self.ph))

        # 在矩形框的中心采样、提取特征
        x = self.get_feature(image, roi)
        y = self.gaussian_peak(x.shape[2], x.shape[1])

        self.alphaf = self.train(x, y, self.sigma, self.lambdar)
        self.x = x
        self.roi = roi
        self.first_frame_flag = False  # 更新标志，表示第一个帧已经处理过

    def update(self, image, face_cascade=None):
        """
        对给定的图像，重新计算其目标的位置
        :param image: 当前帧图像
        :param face_cascade: 人脸检测器（仅在使用摄像头时传入）
        :return: 如果找到目标，返回ROI四元组；否则返回None
        """
        if self.roi is None:
            print("Error: ROI is None. Cannot update tracking.")
            return None

        # 包含矩形框信息的四元组(min_x, min_y, w, h)
        cx, cy, w, h = self.roi
        max_response = -1   # 最大响应值

        for scale in [0.85, 0.9, 1.0, 1.05, 1.1]:
            # 将ROI值处理为整数
            roi = (cx, cy, w * scale, h * scale)
            roi = tuple(map(int, roi))

            z = self.get_feature(image, roi)    # tuple(36, h, w)
            if z is None:
                continue
            # 计算响应
            responses = self.detect(self.x, z, self.sigma)
            responses = cv2.GaussianBlur(responses, (5, 5), 0)  # 高斯平滑滤波
            height, width = responses.shape
            if self.debug:
                cv2.imshow("res", responses)
                cv2.waitKey(0)
            idx = np.argmax(responses)
            res = np.max(responses)
            if res > max_response:
                max_response = res
                dx = int((idx % width - width / 2) / self.scale_w)
                dy = int((idx // width - height / 2) / self.scale_h)
                best_w = int(w * scale)
                best_h = int(h * scale)
                best_z = z
        # print(max_response)
        if max_response < 0.005:
            print("Warning: Low confidence in target. Consider re-detection.")
            if face_cascade:
                roi = self.redetect_target(image, face_cascade)
                if roi:
                    self.first_frame(image, roi)
                    return roi[0] - roi[2] // 2, roi[1] - roi[3] // 2, roi[2], roi[3]
                else:
                    print("Warning: No target detected. Keeping the original ROI.")
                    return cx - w // 2, cy - h // 2, w, h

        # 更新矩形框的相关参数
        self.roi = (cx + dx, cy + dy, best_w, best_h)

        # 更新模板
        motion_estimate = np.linalg.norm(np.array([dx, dy]))  # 计算目标运动的大小
        dynamic_update_rate = self.update_rate * (1 + motion_estimate / 5)
        self.x = self.x * (1 - dynamic_update_rate) + best_z * dynamic_update_rate

        y = self.gaussian_peak(best_z.shape[2], best_z.shape[1])
        new_alphaf = self.train(best_z, y, self.sigma, self.lambdar)
        self.alphaf = self.alphaf * (1 - dynamic_update_rate) + new_alphaf * dynamic_update_rate

        cx, cy, w, h = self.roi
        return cx - w // 2, cy - h // 2, w, h

    def get_feature(self, image, roi):
        # 对矩形框做2.5倍的Padding处理
        cx, cy, w, h = roi
        wp = int(w * self.padding) // 2 * 2
        hp = int(h * self.padding) // 2 * 2
        x = int(cx - wp // 2)
        y = int(cy - hp // 2)

        # 检查边界
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x + wp > image.shape[1]:
            wp = image.shape[1] - x
        if y + hp > image.shape[0]:
            hp = image.shape[0] - y

        # 检查sub_img的大小是否为空
        if wp <= 0 or hp <= 0:
            print(f"Warning: ROI is too small or out of bounds! ROI: {roi}, Image size: {image.shape}")
            return None

        sub_img = image[y:y + hp, x:x + wp, :]
        resized_img = cv2.resize(src=sub_img, dsize=(self.pw, self.ph))

        if self.gray_feature:
            feature = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
            feature = feature.reshape(1, self.ph, self.pw) / 255.0 - 0.5
        else:
            feature = self.hog.get_feature(resized_img)
            if self.debug:
                self.hog.show_hog(feature)

        # Hog特征的通道数、高估、宽度
        fc, fh, fw = feature.shape
        self.scale_h = float(fh) / h
        self.scale_w = float(fw) / w

        # 两个二维数组，前者(fh，1)，后者(1，fw)
        hann2t, hann1t = np.ogrid[0:fh, 0:fw]

        hann1t = 0.5 * (1 - np.cos(2 * np.pi * hann1t / (fw - 1)))
        hann2t = 0.5 * (1 - np.cos(2 * np.pi * hann2t / (fh - 1)))

        # 一个fh x fw的矩阵
        hann2d = hann2t * hann1t

        feature = feature * hann2d
        return feature

    def gaussian_peak(self, w, h):
        """
        生成一个w*h的高斯矩阵
        :param w:
        :param h:
        :return:
        """
        output_sigma = 0.125
        sigma = np.sqrt(w * h) / self.padding * output_sigma
        syh, sxh = h // 2, w // 2

        y, x = np.mgrid[-syh:-syh + h, -sxh:-sxh + w]
        x = x + (1 - w % 2) / 2.
        y = y + (1 - h % 2) / 2.
        g = 1. / (2. * np.pi * sigma ** 2) * np.exp(-((x ** 2 + y ** 2) / (2. * sigma ** 2)))
        return g

    def kernel_correlation(self, x1, x2, sigma):
        """
        核化的相关滤波操作
        :param x1:
        :param x2:
        :param sigma:   高斯参数sigma
        :return:
        """
        # 转换到傅里叶空间
        fx1 = fft2(x1)
        fx2 = fft2(x2)
        # \hat{x^*} \otimes \hat{x}'，x*的共轭转置与x'的乘积
        tmp = np.conj(fx1) * fx2
        # 离散傅里叶逆变换转换回真实空间
        idft_rbf = ifft2(np.sum(tmp, axis=0))
        # 将零频率分量移到频谱中心。
        idft_rbf = fftshift(idft_rbf)

        # 高斯核的径向基函数
        d = np.sum(x1 ** 2) + np.sum(x2 ** 2) - 2.0 * idft_rbf
        k = np.exp(-1 / sigma ** 2 * np.abs(d) / d.size)
        return k

    def train(self, x, y, sigma, lambdar):
        """
        原文所给参考train函数
        :param x:
        :param y:
        :param sigma:
        :param lambdar:
        :return:
        """
        k = self.kernel_correlation(x, x, sigma)
        return fft2(y) / (fft2(k) + lambdar)

    def detect(self, x, z, sigma):
        k = self.kernel_correlation(x, z, sigma)
        response = np.real(ifft2(self.alphaf * fft2(k)))
        return response

    def redetect_target(self, image, face_cascade):
        """
        在当前帧中重新检测目标（人脸）
        :param image: 当前帧图像
        :param face_cascade: 人脸检测器
        :return: 如果找到目标，返回ROI四元组；否则返回None
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            # 找到最大的脸
            max_face = max(faces, key=lambda face: face[2] * face[3])
            return max_face
        else:
            print("Warning: No face detected in the current frame.")
            return None

def draw_rectangle(frame, x, y, w, h, color=(0, 255, 255), thickness=2):
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

def select_roi(frame):
    return cv2.selectROI("tracking", frame, fromCenter=False, showCrosshair=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="Path to the video file. If not provided, the webcam will be used.", type=str, default=None)
    args = parser.parse_args()

    # 加载人脸检测器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 判断是否提供了视频路径
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"Error: Cannot open video file {args.video}. Exiting.")
            exit()
        # 使用标志设置是否使用人脸检测
        use_face_detection = False
    else:
        print("No video file provided. Using webcam.")
        cap = cv2.VideoCapture(0)  # 打开摄像头
        if not cap.isOpened():
            print("Error: Cannot open webcam. Exiting.")
            exit()
        # 使用人脸检测来确定初始ROI
        use_face_detection = True

    tracker = Tracker()

    # 读取第一帧
    ok, frame = cap.read()
    if not ok:
        print("Error: Cannot read video frame.")
        cap.release()
        cv2.destroyAllWindows()
        exit()

    if use_face_detection:
        # 使用人脸检测来确定初始ROI
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            # 找到最大的脸
            roi = max(faces, key=lambda face: face[2] * face[3])
        else:
            print("Warning: No face detected. Selecting ROI manually.")
            roi = select_roi(frame)
            if roi == (0, 0, 0, 0):
                print("No ROI selected. Exiting.")
                cap.release()
                cv2.destroyAllWindows()
                exit()
    else:
        # 直接手动选择ROI
        roi = select_roi(frame)
        if roi == (0, 0, 0, 0):
            print("No ROI selected. Exiting.")
            cap.release()
            cv2.destroyAllWindows()
            exit()

    tracker.first_frame(frame, roi)
    draw_rectangle(frame, *roi)  # 在首次检测到的人脸或手动选择的ROI周围画框

    # 显示第一帧，并绘制初始ROI
    cv2.imshow('tracking', frame)
    cv2.waitKey(0)  # 等待用户按键，以便用户看到初始ROI

    # 主循环
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        if use_face_detection:
            # 在每一帧中使用人脸检测来更新ROI
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(faces) > 0:
                # 找到最大的脸
                roi = max(faces, key=lambda face: face[2] * face[3])
                tracker.first_frame(frame, roi)
                draw_rectangle(frame, *roi)  # 更新画框位置
            else:
                print("Warning: No face detected in the current frame.")
        else:
            # 更新跟踪器
            _x, _y, _w, _h = tracker.update(frame)
            if _x is not None:
                draw_rectangle(frame, _x, _y, _w, _h)

        cv2.imshow('tracking', frame)

        # 按下 'ESC' 或 'q' 键退出
        c = cv2.waitKey(1) & 0xFF
        if c == 27 or c == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()