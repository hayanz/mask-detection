import os
import cv2
import numpy as np

datapath = "yolo/yolo_dataset/"
np.random.seed(42)


def load_path(filename):
    assert type(filename) == str
    return os.path.sep.join([datapath, filename])


class Yolo:
    def __init__(self):
        self.net = cv2.dnn.readNet(load_path("obj_10000.weights"), load_path("yolov3.cfg"))
        self.classes = self.load_classes()
        self.layer_names = self.net.getLayerNames()
        self.out_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        self.color = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.threshold = 1.0

    def load_classes(self):
        with open(load_path("obj.names"), "r") as f:
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
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))

        # filter out of duplicate data.
        coordinates = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        return coordinates, boxes, img

    def show_image(self, img, path=True):
        indexes, boxes, img = self.detect_image(img, path=path)
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                if x < 0 or y < 0 or w < 0 or h < 0:
                    continue
                x1, y1 = x + w, y + h
                cv2.rectangle(img, (x, y), (x1, y1), self.color[0], thickness=2)
        cv2.imshow("Image", img)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

        return img

    def show_video_cv(self):
        cam = cv2.VideoCapture(0)  # 0=front-cam, 1=back-cam
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1300)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1500)

        while True:  # read frames
            _, img = cam.read()  # predict yolo
            img = self.show_image(img, path=False)
            cv2.imshow("", img)  # press q or ESC to quit
            if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1) == 27):
                break  # close camera
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    yolo = Yolo()
    # yolo.show_image("test_images/for_yolo_test.png")
    yolo.show_image("test_images/sample_3.jpg")
    yolo.show_video_cv()
