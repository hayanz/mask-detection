from collections import OrderedDict
import cv2
import dlib

# marks for detecting the face with the predictor
FACEMARKS = OrderedDict([("mouth", (48, 68)), ("r_eyebrow", (17, 22)), ("l_eyebrow", (22, 27)),
                         ("r_eye", (36, 42)), ("l_eye", (42, 48)), ("nose", (27, 35)),
                         ("jaw", (0, 17))])

detector = dlib.get_frontal_face_detector()
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# load an image to detect
img = cv2.imread("test_images/mask_not_cropped/15.jpg")
# convert the image into grayscale
grey = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
faces = detector(grey)

for face in faces:
    x1, x2 = face.left(), face.right()
    y1, y2 = face.top(), face.bottom()
    # print((x1, x2), (y1, y2))  # for debugging

    facemarks = predictor(image=grey, box=face)

    for i in range(FACEMARKS["nose"][0], FACEMARKS["nose"][1]):
        x = facemarks.part(i).x
        y = facemarks.part(i).y

        # draw a circle
        cv2.circle(img=img, center=(x, y), radius=3, color=(0, 255, 0), thickness=1)

    # draw a border of the face
    cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=4)

cv2.imshow("example", mat=img[y1:y2, x1:x2])
cv2.waitKey(3000)  # wait for 3 seconds
cv2.destroyAllWindows()

# crop the face part and save as a different file
# cv2.imwrite("test_images/mask_cropped/11.jpg", img[y1:y2, x1:x2])
