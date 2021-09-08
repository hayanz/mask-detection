import os
import cv2
import numpy as np

datapath = "yolo/yolo_dataset/"
np.random.seed(42)


def load_path(datapath, filename):
    assert type(filename) == str
    return os.path.sep.join([datapath, filename])


class Yolo:
    def __init__(self, model, load, datapath):
        self.net = cv2.dnn.readNet(load_path(datapath, model), load_path(datapath, load))
        self.classes = self.load_classes()
        self.layer_names = self.net.getLayerNames()
        self.out_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        self.color = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.threshold = 1.0

    def load_classes(self):
        with open(load_path(datapath, "obj.names"), "r") as f:
            classes = [line.strip() for line in f.readlines()]
        return classes

    def detect_image(self, img, path=True):
        net, classes, out_layers = self.net, self.classes, self.out_layers
        if path:
            img = cv2.imread(img)
        img = cv2.resize(img, None, fx=0.4, fy=0.4)
        width, height, channels = img.shape

        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(out_layers)

        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                id = np.argmax(detection)
                confidence = detection[id]
                if confidence > self.threshold:  # Object detected
                    center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                    w, h = int(detection[2] * width), int(detection[3] * height)
                    # coordinates
                    x, y = int(center_x - w / 2), int(center_y - h / 2)
                    if x < 0 or y < 0 or w < 0 or h < 0:
                        continue
                    boxes.append([x, y, w, h, confidence])
                    confidences.append(float(confidence))

        # filter out of duplicate data.
        coordinates = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        return coordinates, boxes, img

    def max_conifidence(self, info):
        confidences = [i[4] for i in info]
        max_idx = np.argmax(confidences)
        return info[max_idx]

    def crop_detected(self, img, coordinates):
        detected = []

        x, y, w, h, _ = coordinates
        x1, y1 = x + w, y + h
        cropped = img[y:y1, x:x1]
        detected.append(cropped)

        # show a cropped image
        cv2.imshow(winname="cropped", mat=cropped)  # for debugging
        cv2.waitKey(1000)  # for debugging
        cv2.destroyAllWindows()  # for debugging

        return detected

    def show_image(self, img, path=True):
        indexes, boxes, img = self.detect_image(img, path=path)
        max_confidence = self.max_conifidence(boxes)
        cropped = self.crop_detected(img, max_confidence)
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h, conf = boxes[i]
                x1, y1 = x + w, y + h
                cv2.rectangle(img, (x, y), (x1, y1), self.color[0], thickness=2)

        # show the result
        # cv2.imshow("Image", img)  # for debugging
        # cv2.waitKey(1000)  # for debugging
        # cv2.destroyAllWindows()  # for debugging

        return img

    def show_video_cv(self, loaded_video):
        video = cv2.VideoCapture(loaded_video)

        while True:  # read frames
            _, img = video.read()  # predict yolo
            img = self.show_image(img, path=False)
            cv2.imshow("", img)  # press q or ESC to quit
            if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1) == 27):
                break  # close camera

        video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    yolo = Yolo(model="obj.cfg", load="obj_10000.weights", datapath=datapath)
    # test yolo with images
    """
    for i in range(4):
        yolo.show_image("test_images/sample_%d.jpg" % i)
    for i in range(11, 15):
        yolo.show_image("test_images/mask_not_cropped/%d.jpg" % i)
    """
    yolo.show_image("test_images/for_yolo_test.png")

    # test yolo with video
    # yolo.show_video_cv("test_images/video_for_yolo.mp4")
