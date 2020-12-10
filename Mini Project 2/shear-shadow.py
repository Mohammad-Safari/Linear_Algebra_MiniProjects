import numpy as np
from matplotlib import image as im
import matplotlib.pyplot as plt


def default_coord(height, width, h_offset=0, w_offset=0):
    """fills an array with default coordination(initial map construction)
    e.g [
        [[0,0],[0,1]],
        [[1,0],[1,1]]
        ]
    Args:
        height (int): given initial height
        width (int): given initial width
        h_offset (int, optional): new data extra height. Defaults to 0.
        w_offset (int, optional): new data extra height. Defaults to 0.

    Returns:
        3darr: matrice of unchanged(not trasnformed) coordination
    """
    coord_mat = np.zeros((height+h_offset, width+w_offset, 2), dtype='int')
    for i in range(height):
        for j in range(width):
            coord_mat[i, j] = (i, j)
    return coord_mat


def transform_calcualtion(height, width, t_mat, h_offset, w_offset, nh_flag, nw_flag):
    """a suitable matrix multiplication(customized for 2d transfroms)
        actually use one square transform matrix and one horizontal 
        matrix(but used in multiplication as its vertical!)

    Args:
        height (int): given initial height
        width (int): given initial width
        t_mat (2darr(2,2)): transform matrix with (2, 2) shape
        h_offset (int, optional): new data extra height. Defaults to 0.
        w_offset (int, optional): new data extra height. Defaults to 0.
        nh_flag (boolean, optioanl): checking negative height existance
        nw_flag (boolean, optioanl): checking negative width existance

    Returns:
        1darr: flatened result of multiplication
    """
    # default coordination/location of transformed matrix according to source data(data map)
    coord_map = default_coord(height, width, h_offset, w_offset)

    for i in range(height):
        for j in range(width):
            # base calculations
            result = [(t_mat[0][0])*(coord_map[i, j, 0])+int((t_mat[0][1])*(coord_map[i, j, 1])),
                      (t_mat[1][0])*(coord_map[i, j, 0])+(t_mat[1][1])*(coord_map[i, j, 1])]
            # since all coordinations must not be negative
            # if happened also apply a translation by offset
            coord_map[i, j, :] = [(result[0], result[0]+h_offset)[nh_flag],
                                  (result[1], result[1]+w_offset)[nw_flag]]
    return coord_map


def affine_2Dtransform(img, t_mat, height, width, h_offset=0, w_offset=0, nh_flag=False, nw_flag=False):
    """applies a 2D transform(coordination-based) to the given data according to a given transform matrix(2,2)

    Args:
        img (3darr): given image data
        t_mat (2darr(2,2)): transform matrix with (2, 2) shape
        height (int): source data given height
        width (int): source data given height
        h_offset (int, optional): new data extra height for possible data increase size(with default whilte space). Defaults to 0.
        w_offset (int, optional): new data extra width for possible data increase size(with default whilte space). Defaults to 0.
        nh_flag (boolean, optioanl): checking negative height existance
        nw_flag (boolean, optioanl): checking negative width existance

    Returns:
        3darr: transformed image data
    """
    # transform matrix must be validated
    if(np.shape(t_mat) != (2, 2)):
        return img

    # implementing matrix multiplication to a default map of source data in order to apply transform
    # and to achieve coordination/location of transformed matrix according to source data(data map)
    coord_map = transform_calcualtion(
        height, width, t_mat, h_offset, w_offset, nh_flag, nw_flag)

    # transformed image data construction
    t_img = np.full((height+h_offset, width+w_offset, 3), 255, dtype='uint8')

    # applying new map to image inorder to complete the transform
    try:
        for i in range(height):
            for j in range(width):
                [i_new_coord, j_new_coord] = coord_map[i, j, :]
                # unhandled bound-jumpout
                t_img[i_new_coord, j_new_coord, :] = img[i, j, :]
    except:
        print("not enough offset/negative coordination pushed")
        return img
    return t_img


def shear_image(img, l):
    """shears(skews) given image according to lambda(l)

    Args:
        img (3darr): source image data
        l (float): lambda or shear coefficient(sign will be considered)

    Returns:
        3darr: sheared image data
    """
    # building a suitable array for sheared image
    img_h = img.shape[0]  # source image data height
    img_w = img.shape[1]  # source image data width
    # new image data width(maximum row offset)
    new_img_off = int((abs(l)*img_h))
    # transform matrix
    transform = [[1, 0], [l, 1]]
    # new image data
    nw_flag = (l < 0)
    new_img = affine_2Dtransform(
        img, transform, img_h, img_w, 0, new_img_off, False, nw_flag)

    return new_img


def paint_shadow(img, sheared_img, l):
    """painting the source image(img) shadow according to transform(shear)

    Args:
        img (3darr): source image data
        sheared_img (3darr): shadow image data
        l (float): lambda or shear coefficient(sign have been considered in shear)

    Returns:
        new_img (3darr): source image with shadow
    """
    img_h = img.shape[0]  # source image data height
    img_w = img.shape[1]  # source image data width

    # white data
    new_img = np.full((img_h, img_w, 3), 255, dtype='uint8')

    for i in range(img_h):
        row_offset = int((img_h - i)*l) if l>=0 else int(i*l)
        for j in range(img_w):
            # if in source img data is not white paint source image
            if(not np.all(img[i, j] >= (240, 240, 240))):
                new_img[i, j] = img[i, j]
            # else if in sheared img we have non white
            elif(j > row_offset and not np.all(sheared_img[i, j] >= (230, 230, 230))):
                new_img[i, j] = (100, 100, 100)
            else:
                new_img[i, j] = img[i, j]
    return new_img

# driver code
if __name__ == "__main__":

    inp = input("Enter the Image Direcotry:(default is Default.jpg)\n")
    img_file_address = "Default.jpg" if(inp == "") else (inp)
    # img_file_address = "Default.jpg"
    inp = input("Enter the desired value of shear - negative also possible:(default is 0.1)\n")
    l = float(0.1) if(inp == "") else float(inp)
    # l = float(-0.1)
    try:
        img = im.imread(img_file_address)
        new_img = shear_image(img, l)
        # plt.imshow(new_img)
        # plt.show()
        shadow_img = paint_shadow(img, new_img, l)
        plt.imshow(shadow_img)
        plt.show()
    except:
        print("could not read from the given address/given shear parameter")
