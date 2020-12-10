import numpy as np
from matplotlib import image as im
import matplotlib.pyplot as plt


def shear_image(img, l):
    """shears(skews) given image according to lambda(l)

    Args:
        img ([3darr]): source image data
        l ([float]): lambda or shear coefficient(sign will be considered)

    Returns:
        [3darr]: sheared image data
    """
    # building a suitable array for sheared image
    img_h = img.shape[0]  # source image data height
    img_w = img.shape[1]  # source image data width
    # new image data width(maximum row offset)
    new_img_w = int(img_w+(abs(l)*img_h))
    # new image data shape
    new_img = np.full((img_h, new_img_w, 3), 255, dtype='uint8')    

    # pasting each row of source image with corresponding offset into new data
    for i in range(img_h):
        # calculated offset for each row(lambda sign also considered!)
        row_offset = (int((img_h - i)*l), int(i*(-l)))[l < 0]
        new_img[i, row_offset:row_offset+img_w, :] = img[i, :, :]
    return new_img


def paint_shadow(img, sheared_img, l):
    """[summary]

    Args:
        img ([3darr]): [description]
        sheared_img ([3darr]): [description]
        l ([float]): lambda or shear coefficient(sign will be considered)

    """
    img_h = img.shape[0]  # source image data height
    img_w = img.shape[1]  # source image data width
    # sh_img_w = sheared_img.shape[1]  # sheared image data width
    new_img = np.full((img_h, img_w, 3), 255, dtype='uint8')

    for i in range(img_h):
        row_offset = int((img_h - i)*l)  # negative l?
        for j in range(img_w):
            # if in source img data is not white
            if(not np.all(img[i, j] >= (245, 245, 245))):
                new_img[i, j] = img[i, j]
            # else if in sheared img we have non white
            elif(j > row_offset and not np.all(sheared_img[i, j] >= (245, 245, 245))):
                new_img[i, j] = (100, 100, 100)
            else:
                new_img[i, j] = img[i, j]

    return new_img


if __name__ == "__main__":

    #inp = input("Enter the Image Direcotry:(default is Default.jpg)\n")
    #img_file_address = (inp, "Default.jpg")[inp == ""]
    img_file_address = "Default.jpg"
    #inp = input("Enter the desired value of shear:(default is 0.1)\n")
    #img_file_address = (float(inp), float(0.1))[inp == ""]
    l = float(0.1)
    # try read img!!
    img = im.imread(img_file_address)
    new_img = shear_image(img, l)
    shadow_img = paint_shadow(img, new_img, l)
    plt.imshow(shadow_img)
    plt.show()
