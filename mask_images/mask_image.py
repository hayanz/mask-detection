from PIL import Image
import numpy as np
import cv2


# find the pixels around the target pixel
def find_neighbors(img, col, row):
    left, right = col - 1, col + 1
    up, down = row - 1, row + 1
    neighbors = [img.getpixel((col, up)), img.getpixel((col, down)), \
                 img.getpixel((left, row)), img.getpixel((right, row))]
    return neighbors


# class of the mask image
class Mask(object):
    def __init__(self, filename):
        self.filename = filename
        self.img = Image.open(filename).convert("RGBA")
        self.width, self.height = self.img.size

    # resize the image
    def resize(self, width, height, returned=False):
        # initial size of the mask is 1200 * 1200
        resized = self.img.resize((width, height))
        if returned:  # return as the new object
            return resized
        self.img = resized  # save the resized image to the current file
        self.width, self.height = self.img.size  # redefine the width and the height

    # find the coordinate of pixels at the border of the image
    def find_borders(self):
        borders = []  # an empty list to save the coordinate of pixels
        img = self.img
        for col in range(img.size[0]):  # img.size[0] = width
            for row in range(img.size[1]):  # img.size[1] = height
                # (the value of column, the value of row) = the coordinate of the pixel
                pixel = img.getpixel((col, row))
                if pixel[3] > 0:
                    neighbors = find_neighbors(img, col, row)
                    # check if the pixel is at the border of the image
                    for n in neighbors:
                        if n[3] == 0:
                            borders.append((col, row))
        return borders

    # find the coordinate of the mask that would be the
    def find_topbottom(self):
        points = []  # an empty list to save the result
        borders = self.find_borders()
        for pixel in borders:
            if pixel[0] == self.width / 2:
                points.append(list(pixel))
        return points

    # find the coordinate of the center of the mask image
    def find_center(self):
        top, bottom = self.find_topbottom()
        center = [np.mean(n) for n in zip(top, bottom)]
        return center

    # get the ratio between the entire length of the image and the mask
    def get_ratio(self):
        top, bottom = self.find_topbottom()
        # calculate the length of the mask using coordinates
        mask_length = float([abs(p - q) for p, q in zip(top, bottom)][1])
        return self.height / mask_length

    # for debugging
    def show_points(self):
        img = cv2.imread(self.filename)
        points = self.find_topbottom()
        # points = self.find_borders()
        for point in points:
            # draw a circle
            cv2.circle(img=img, center=point, radius=1, color=(0, 255, 0), thickness=1)
        cv2.imshow("test", mat=img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # save the image
    def save(self):
        self.img.save(self.filename)

    # save the image as the new one
    def save_as(self, filename, filetype, filepath=None):
        valid_type = {"PNG": ".png", "JPEG": ".jpeg", "JPG": ".jpg"}
        try:
            if filetype not in valid_type:
                raise TypeError
        except TypeError:
            print("Invalid type.")  # print the error message
            return  # make the program ends
        # save the image
        filename = filename + valid_type[filetype]
        if filepath is not None:
            filename = filepath + filename
        self.img.save(filename)

    # close the image
    def close(self):
        self.img.close()


# for testing the code with some images
if __name__ == "__main__":
    test = Mask("mask_images/black_grey.png")
    # resize the image (1200 = original size)
    test.resize(1200, 1200, returned=True).save("mask_images/test.png")
    test2 = Mask("mask_images/test.png")
    test2.find_center()
    print(test2.get_ratio())  # for testing the method
    # test2.show_points()
    test.close()
