import cv2

# 提示信息
print("Select a ROI and then press SPACE or ENTER button!")
print("Cancel the selection process by pressing c button!")

# 打开视频文件
video_path = './Ke.mp4'  # 替换为你的视频文件路径
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Unable to open the video file.")
    exit()

# 读取第一帧
ret, frame = cap.read()
if not ret:
    print("Error: Unable to read from the video file.")
    cap.release()
    exit()

# 选择感兴趣区域
bbox = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
if bbox == (0, 0, 0, 0):
    print("No ROI selected. Exiting.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# 初始化 KCF 跟踪器
tracker = cv2.TrackerKCF_create()

# 初始化跟踪器
tracker.init(frame, bbox)

# 循环处理视频帧
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video file.")
        break

    # 更新跟踪器
    success, bbox = tracker.update(frame)

    if success:
        # 目标被成功跟踪
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        # 目标丢失
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # 显示当前帧
    cv2.imshow("Tracking", frame)

    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
# import cv2

# # 提示信息
# print("Select a ROI and then press SPACE or ENTER button!")
# print("Cancel the selection process by pressing c button!")

# # 打开摄像头
# cap = cv2.VideoCapture(0)  # 参数 0 代表默认摄像头
# if not cap.isOpened():
#     print("Error: Unable to access the webcam.")
#     exit()

# # 读取第一帧
# ret, frame = cap.read()
# if not ret:
#     print("Error: Unable to read from the webcam.")
#     cap.release()
#     exit()

# # 选择感兴趣区域
# bbox = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
# if bbox == (0, 0, 0, 0):
#     print("No ROI selected. Exiting.")
#     cap.release()
#     cv2.destroyAllWindows()
#     exit()

# # 初始化 KCF 跟踪器
# tracker = cv2.TrackerKCF_create()

# # 初始化跟踪器
# tracker.init(frame, bbox)

# # 循环处理摄像头帧
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Unable to fetch frame from the webcam.")
#         break

#     # 更新跟踪器
#     success, bbox = tracker.update(frame)

#     if success:
#         # 目标被成功跟踪
#         x, y, w, h = [int(v) for v in bbox]
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#     else:
#         # 目标丢失
#         cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

#     # 显示当前帧
#     cv2.imshow("Tracking", frame)

#     # 按下 'q' 键退出
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # 释放资源
# cap.release()
# cv2.destroyAllWindows()
