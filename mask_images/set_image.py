from collections import OrderedDict
from PIL import Image
import shutil

COLORS = OrderedDict([("black", (0, 0, 0, 255)), ("black_grey", (64, 64, 64, 255)),
                      ("grey", (128, 128, 128, 255)), ("white_grey", (191, 191, 191, 255))])
original = "white.png"  # the default color is white

for color in COLORS.keys():
    newfile = color + ".png"
    shutil.copyfile(original, newfile)
    img = Image.open(newfile)
    img = img.convert("RGBA")  # 4x8-bit pixels, true color with transparency mask
    img_data = img.getdata()

    new_data = []
    for item in img_data:
        # check the transparency of of the pixel
        if item[3] > 0:
            # change the color only if the pixel is not transparent
            color_item = COLORS[color]
        else:

            color_item = (255, 255, 255, 0)
        new_data.append(color_item)
    img.putdata(new_data)

    img.save(color + ".png")
    img.show()  # for debugging
