
import pandas   as pd
import numpy    as np
from   numpy    import matrix as mt
from   numpy    import linalg as la
import math     as ma
from matplotlib import pyplot as plt

def uinput():
    """gets user input

    Returns:
        string , string: file address ,  file data column
    """
    inp = input("Enter the data Direcotry: (default is GOOGL.csv)\n")
    data_file_address = "GOOGL.csv" if(inp == "") else (inp)
    inp = input("Enter the desired Column: (default is Open)\n")
    data_file_column  = "Open"      if(inp == "") else (inp)
    return (data_file_address, data_file_column)

def csvload(data_file_address, data_file_column):
    """load csv file data and returns total data, training(head) data number and content

    Args:
        data_file_address (string): address
        data_file_column (string): data tag

    Returns:
        md array, int, nd array: total data, training(head) data number and content
    """
    data = pd.read_csv(data_file_address, index_col=0, header=0, parse_dates=True)
    train = len(data) - 10
    head = data.head(train)[data_file_column].to_numpy()
    return [data, train, head]
    
def regression_weights(x_t, head, exponential=False):
    """does needed matrix calculations for regression weights

    Args:
        x_t (nd arr): X transpose matrix
        head (md arr): Y(training data) matrix
        exponential (bool, optional): if a simple exponential regression is about to do. Defaults to False.

    Returns:
        pd arr: calculated weights
    """
    product = np.copy(x_t)
    product = np.matmul(product, mt.transpose(product))
    product = la.inv(product)
    product = np.matmul(product, x_t)
    product = np.matmul(product,(np.log(mt.transpose(head)) if (exponential)  else mt.transpose(head))) # exponential must fit the logarithm of records
    return product

# driver code
if __name__ == "__main__":

    #############
    ## user input
    #############
    data_file_address, data_file_column = uinput()
    
    ###################################
    ## loading data frame from csv file
    ###################################
    # try:
    [data, train, head] = csvload(data_file_address, data_file_column)
    # except:
    #     print("could not read from the given address")

    ############################
    ## plotting training records 
    ############################
    plt.plot(head, color="blue", label="training records")
    plt.legend()
    print("..close training data plot to continue..")
    plt.show()

    ############################################################################
    # #### calculating coeffitients of 3 regression among head(training) records 
    ############################################################################
    # #### decription:
    # #### regression 1 linear model: b0 + b1x
    # #### regression 2 quadratic model: b0 + b1x + b2x^2
    # #### regression 3 linear-sinusoidial model: b0 + b1x + b2 sin(wx)
    # #### regression 4 exponential model: a0(a1^x) or exp(b0 + b1x) - b0=log(a0), b1=log(a1)
    reg_detail = {
        0:"linear(simple) regression",
        1:"quadratic regression",
        2:"linear-sinusoidal regression",
        3:"exponential regression"
    }
    ## initial matrices for calculations
    x_t = np.arange(1, train + 1, 1)
    x_t = [x_t, np.copy(x_t), np.copy(x_t), np.copy(x_t)]
    ## 
    x_t[0] = np.array([np.ones(train), x_t[0]])
    x_t[1] = np.array([np.ones(train), x_t[1], x_t[1]**2])
    x_t[2] = np.array([np.ones(train), x_t[2], np.sin(2 * ma.pi / 500 * x_t[2])])
    x_t[3] = np.array([np.ones(train), x_t[3]])
    ## matrix calculation on data
    weights = [np.ones(train), np.ones(train), np.ones(train), np.ones(train)]
    for i in range(4):
        weights[i] = regression_weights(x_t[i], head, True if(i == 3) else False)
        


    ## result wieghts
    print("calculated weights:");
    print(weights);
    print("\n");

    ########################################################################
    # ### plotting whole records and regression points through whole domain
    ########################################################################

    domain = np.arange(1., train + 11, 1)
    regression = np.array([np.copy(domain), np.copy(domain), np.copy(domain), np.copy(domain)])
    regression[0] = weights[0][0] + weights[0][1] * (regression[0])
    regression[1] = weights[1][0] + weights[1][1] * (regression[1]) + weights[1][2] * (regression[1]**2) 
    regression[2] = weights[2][0] + weights[2][1] * (regression[2]) + weights[2][2] * np.sin(2 * ma.pi / 500 * (regression[2]))
    regression[3] = np.exp(weights[3][0] + weights[3][1] * (regression[3]))



    plt.plot(regression[0], color='red', linewidth=3, label=reg_detail[0])
    plt.plot(regression[1], color='orange', linewidth=3, label=reg_detail[1])
    plt.plot(regression[2], color='black', label=reg_detail[2], linestyle='-.')
    plt.plot(regression[3], color='blue', label=reg_detail[3], linestyle='--')
    plt.scatter(domain, data.head(train + 10)['Open'], color='green', s=1, label="total records")
    plt.legend()
    print("..close reressions plot to continue..")
    plt.show()

    ##########################################
    # ### calculating error of 10 tail records
    ##########################################

    reg_flag = -1   ## best regression
    min_er = 10**10 ## minimal   error
    ## logging error calculations and choosing the best
    error = np.array([0, 0, 0, 0])
    tail = data.tail(10)[data_file_column].to_list()
    for k in range(4):
        print("############ " + reg_detail[k] + " ############")
        for i in range(10):
            predicted = regression[k][train + i]
            e = predicted - tail[i]
            print("actual:" + str(tail[i]) + ", predicted:" + str(predicted) + ", error:" + str(e) + "\n")
            error[k] += e
            ##
            if abs(min_er) >= abs(error[k]):
                min_er = error[k]
                reg_flag = k;
        print("total predication error:" + str(error[k]) + "\n" + "mean error:" + str(error[k]/10) + "\n")
    
    ## plotting the final result
    print("best prediction by " +  reg_detail[reg_flag] + "with mean error: " + str(error[reg_flag]/10)) ## min_er
    plt.plot(regression[reg_flag], color='red', linestyle="--"  , label=reg_detail[reg_flag])
    plt.scatter(domain, data.head(train + 10)[data_file_column], color='green', s=1, label="total records")
    plt.legend()
    print("..close plot to finish..")
    plt.show()

