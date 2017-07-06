import cv2
from video_face_detector import VideoFaceDetector

camera = cv2.VideoCapture(0)

vfd = VideoFaceDetector('haarcascade_frontalface_default.xml', camera)

smooth_fps = 0
while True:
    start_time = cv2.getTickCount()

    frame = vfd.getFrameAndDetect()

    if vfd.isFaceFound:
        x, y, w, h = [int(i) for i in vfd.face()]
        p1 = (x, y)
        p2 = (x+w, y+h)
        cv2.rectangle(frame, p1, p2, (255, 0, 0))

    end_time = cv2.getTickCount()

    fps = cv2.getTickFrequency() / (end_time - start_time)
    smooth_fps = 0.9 * smooth_fps + 0.1 * fps

    print("FPS:", smooth_fps)

    cv2.imshow('Camera', frame)

    if cv2.waitKey(30) & 0xff == ord('q'):
        exit(0)
