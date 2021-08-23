import cv2
import dlib


class Face(object):
    def __init__(self, filename):
        self.filename = filename
        self.img = cv2.imread(filename)
        self.grey = self.make_grayscale()
        self.detector = dlib.get_frontal_face_detector()

    def make_grayscale(self):
        temp = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
        return cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)

    # detect the face
    def find_face(self):
        img, grey = self.img, self.grey
        faces = self.detector(self.img)

        for face in faces:
            x1, x2 = face.left(), face.right()
            y1, y2 = face.top(), face.bottom()
            # draw a rectangle to show a border of the face
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # cv2.imshow(winname="found faces", mat=img)  # for debugging
        # cv2.waitKey(3000)  # wait for 3 seconds
        # cv2.destroyAllWindows()  # close all windows

        return [x1, x2, y1, y2]

    # crop the face image
    def crop(self):
        try:
            # find coordinates of the border of the face
            faces = self.find_face()
            if faces is None or len(faces) == 0:
                raise UnboundLocalError
        except UnboundLocalError:
            print("Cannot crop the face")  # error message for debugging
            return  # the function ends and return nothing

        # crop the image
        x1, x2, y1, y2 = faces
        cropped = self.img[y1:y2, x1:x2]

        cv2.imshow(winname="cropped faces", mat=cropped)  # for debugging
        cv2.waitKey(3000)  # wait for 3 seconds
        cv2.destroyAllWindows()  # close all windows

        return cropped


# for testing the code with some images
if __name__ == "__main__":
    for i in range(5):
        target = "faces_dataset/sample_" + str(i) + ".jpg"
        tester = Face(target)
        tester.crop()
