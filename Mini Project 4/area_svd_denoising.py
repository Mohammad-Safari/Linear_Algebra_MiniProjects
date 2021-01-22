import numpy    as np
from   numpy    import linalg as la
from   numpy    import matrix as mt
from matplotlib import pyplot as plt
from matplotlib.pyplot import title
import utility  as ut

# ### Surface Denoise
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

    print("..close main area to continue..")
    ut.show_main()
    print("..close noisy area to continue..")
    ut.show_noisy()

    ########################
    # ### SVD Decompostion #
    ########################
    records = ut.get_function()
    records.shape
    fdc = {
        'U':0,
        'S':0,
        'V':0,
        'R':0
    }
    fdc['U'], fdc['S'], fdc['V'] = la.svd(records)

    ######################################################
    # #### choosing a cuttoff value according to scatter #
    ######################################################
    plt.scatter(range(30), fdc['S'], color='red', s=1)
    plt.scatter(range(30), fdc['S'], color='blue', s=1)
    plt.title("singular value scatter")
    print("..close scatter to continue..")
    plt.show()

    ########################################################################
    # ### Denoising Coordination and Reconstructing Function Coordinations #
    ########################################################################
    fcutoff = 0.5
    ##
    diff = records.shape[0] - records.shape[1]
    diff = 0 if(diff < 0) else diff
    ## cutting off
    fdc['R'] = singularValueCutOff(fdc['S'], records.shape, 0.5, diff)

    ######################
    # ### reconstruction #
    ######################
    new_func = np.matrix(fdc['U'])*np.matrix(fdc['R'])*np.matrix(fdc['V'])

    #########################
    # ### comparing results #
    #########################
    print('Noisy function:\n')
    ut.show_noisy()
    print('Denoised function: with cut off value ' + str(fcutoff) + '\n')
    ut.show_my_matrix(new_func)



