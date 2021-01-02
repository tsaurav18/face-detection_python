import cv2
from random import randrange

frameWidth = 700
frameHeight = 550
# cv2.namedWindow("output", cv2.WINDOW_NORMAL)
# load some pre-trained data on face frontal from opency
trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# choose an image to detect face in
# img = cv2.imread("IMG_1334.jpg")
webcam = cv2.VideoCapture(0)
webcam.set(3, frameWidth)
webcam.set(4, frameHeight)
webcam.set(10, 150)
# iterate forever over frames
while True:
    # read the current frame
    successful_frame_read, frame = webcam.read()

    # must convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 256, 0),
            10,
        )
    cv2.imshow("face detector", frame)

    key = cv2.waitKey(1)
    ## stop Q the video
    if key == 81 or key == 113:
        break
webcam.release()


# must convert to grayscale
# grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect faces
# face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
# draw a rectangle around face
# for (x, y, w, h) in face_coordinates:
#     cv2.rectangle(
#         img,
#         (x, y),
#         (x + w, y + h),
#         (randrange(256), randrange(256), randrange(256)),
#         10,
#     )
# # print(face_coordinates)

# cv2.imshow("output", img)

# cv2.waitKey()
print("code completed")

# source work/bin/activate