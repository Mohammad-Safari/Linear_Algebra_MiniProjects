import numpy    as np
from   numpy    import linalg as la
from   numpy    import matrix as mt
from matplotlib import pyplot as plt
from matplotlib.pyplot import title

def singularValueCutOff(Sig, img_shape, cutoff, diff):
    """cut of singular values according to a given cutoff and regenerates the new singualr value matrix

    Args:
        Sig ([type]): Singular value matrix
        img ([type]): image shape
        cutoff ([type]): cutoff value for skipping noise generator values in sigma
        diff ([type]): dimension difference

    Returns:
        [type]: new(revised) Singular value matrix
    """
    R = np.zeros(img_shape[1])
    for i in range(len(Sig)):
        R[i] = Sig[i] if(Sig[i] > cutoff) else 0
    R = np.reshape(np.append(np.diag(R), np.zeros((diff, img_shape[1]))),(img_shape[0], img_shape[1]))
    return R

# driver code
if __name__ == "__main__":

    inp = input("Enter the img Direcotry:(default is noisy.jpg)\n")
    data_file_address = "noisy.jpg" if(inp == "") else (inp)

    img = plt.imread(data_file_address) # read picture from path
    print("..close the noisy image to continue..")
    plt.imshow(img)
    plt.title("noisy-image")
    plt.show()

    ########################################
    # ### calculating matrix decomposition #
    ########################################

    dcm = {
        'U':[0, 0, 0],
        'S':[0, 0, 0],
        'V':[0, 0, 0],
        'R':[0, 0, 0]
    }
    for i in range(3):
        dcm['U'][i], dcm['S'][i], dcm['V'][i] = la.svd(img[:, :, i])

    ########################################################
    # #### choosing a cuttoff value according to histogram #
    ########################################################

    plt.hist(dcm['S'],range(0,10000,100))
    plt.title("singular values histohram(RGB)")
    print("..close the histogram to continue..")
    plt.show()


    plt.scatter(range(142), dcm['S'][0], color='red'  , s=1, label="R singualar values scatter")
    plt.scatter(range(142), dcm['S'][2], color='green', s=1, label="G singualar values scatter")
    plt.scatter(range(142), dcm['S'][1], color='blue' , s=1, label="B singualar values scatter")
    print("..close the scatter to continue..")
    plt.legend()
    plt.show()

    #####################################################################
    # ### Denoising Picture and Recostructing Image by New Sigma matrix #
    #####################################################################
    cutoff = [1300,1300,1300]
    
    ##
    diff = img.shape[0]-img.shape[1] ## dimension difference must be filled with zeros
    diff = 0 if(diff < 0) else diff 
    ##
    for k in range(3):
        dcm['R'][k] = singularValueCutOff(dcm['S'][k], img.shape, cutoff[k], diff)

    ############################
    # ### Image Reconstruction #
    ############################
    new_img = np.full((356,142,3),255);
    for k in range(3):
        new_img[:,:,k] = np.matrix(dcm['U'][k])*np.matrix(dcm['R'][k])*np.matrix(dcm['V'][k])


    print("..close denoised image to finish..")
    plt.imshow(new_img)
    plt.title("Denoised - cutoff:" + str(cutoff[0]) + ", " + str(cutoff[1]) + ", "  + str(cutoff[2]))
    plt.show()
