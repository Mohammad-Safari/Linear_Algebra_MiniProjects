import numpy as np


def rowSwap(matrix, i, _i):
    """intended to swap(interchange) to rows of a given matrix

    Args:
        matrix (numpy array): the origin matrix which is going to be row swapped
        i (int): row number which is going to be swapped
        _i (int): row number which is going to be swapped
    """
    # interchanging two row subset matrices
    matrix[[i, _i], :] = matrix[[_i, i], :]
    return


def rowReplacement(matrix, dstRow, srcRow, coef=1):
    """intended to do a matrice row replacement calculation

    Args:
        matrix (numpy array): the origin matrix which is going to have row replacement
        dstRow (int): the row number whose entries are going to be used for replacement
        srcRow (int): the row number whose entries are going to be affected
        coef (int, optional): the coefitient which is going to multiply with srcRow entries before replacement Defaults to 1.
    """
    matrix[[dstRow], :] += matrix[[srcRow], :]*coef
    return


def echelonReducedRowAnalyze(row, i, j):
    """analyze reduced echelon matrix in given pivot position

    Args:
        matrix (numpy array): calculated echelon matrix
        i (int): row number of passed pivot
        j (int): column number of passed pivot

    Returns:
        string: analyzed result in string
    """
    index = i + 1  # starting index from next entry of pivot
    # inserting the constant at the first of result expression
    result = " ({})-".format(row[-1])
    for num in row[:-1]:
        index += 1  # increasing the variable index
        if num != 0 and index != i+1:  # if coefficient was not zero
            # inserting variable with coefficient
            result += "({}x_{})-".format(num, index)
    return result[:-1]


###------------------------------------------------------------------------------------------------------------------------###
# GETTING INPUTS FROM USER
print("Coefficient martix:")
# m rows number & n cols number
# m, n = [3, 4]
m, n = list(
    map(int, input("Enter number of rows and columns respectively:\n").split()))
# mat a ones matrix
mat = np.ones((m, n))
# mat = np.array([[0.0, 0, 1, -2, -3], [1, -7, 0, 6, 5], [-1, 7, -4, 2, 7]])

for i in range(m):  # getting user input as a row
    mat[i] = np.array(
        list(map(float, input("Enter row {}:\n".format(i+1)).split())))

# getting user input as a row, then mapping it to a column of 2d array so matches the coefficient matrix
cArr = np.array(
    list(map(lambda cons: [cons], map(int, input("Enter constant vlaues:\n").split()))))

mat = np.array(np.hstack((mat, cArr)))  # appending two matrices horizontally

print("Given matrix:")
print(mat)  # printing result
n++  # a constant value column has been added
###------------------------------------------------------------------------------------------------------------------------###
# ALGORITHM STARTS
pivot = [-1, -1]  # last pivot position
cpivot = [-1, -1]  # current pivot position
# going through the columns
# skipping the (last)pivot column and before('cause pivots have been found)
for j in range(0, n):
    hasPivot = False
    # skipping the (last)pivot row('cause precedents must have turned to zero)
    for i in range(pivot[0]+1, m):
        if mat[i][j] != 0:  # first non-zero entry
            hasPivot = True  # checking pivot flag
            # current pivot position (definite column found still row is not definite)
            cpivot = [i, j]
            # getting the non-zero pivot postion row to the top row
            for _i in range(pivot[0]+1, cpivot[0]):
                if mat[_i][j] == 0:
                    rowSwap(mat, cpivot[0], _i)
                    cpivot[0] = _i
                    break
            # turn all below entries to zero by replacement calculations
            for _i in range(cpivot[0]+1, m):
                if mat[_i][j] != 0:
                    coef = mat[_i][j]/mat[cpivot[0], cpivot[1]]
                    coef *= -1
                    rowReplacement(mat, _i, cpivot[0], coef)
                    continue
            break
    if hasPivot:
        pivot = cpivot[:]

# reduceing echelon matrix if neccesary
for i in range(m):
    for j in range(n):
        if mat[i][j] != 0:
            if mat[i][j] != 1:
                mat[i] = mat[i]/mat[i][j]
            break;

# printing calculated final matrix
print("Calculated (reduced)echelon matrix:")
print(mat)
###------------------------------------------------------------------------------------------------------------------------###
# ANALYSING THE RESULT MATRIX AND OUTPUT
--n  # we do not mention constant column any more
# initializing output
results = []
for j in range(n):
    results.append("x_{} is free".format(j+1))
# filling the results
for i in range(m):
    isAllZero = True; # checking if coefficients be zero
    for j in range(n):
        if mat[i][j] != 0:
            isAllZero = False # check existance of not zero coefficient
            # sending matrix and pivots for analysis(string builder!:))
            results[j] = ("x_{} = ".format(j+1) +
                          echelonReducedRowAnalyze(mat[i][j+1:], i, j))
            break
    if isAllZero and mat[i][n]!=0:
        results.clear()
        results.append("Row {}th not possible!\nSystem is inconsistent!".format(i+1))
        break;
# AND at last printing the final results
for statement in results:
    print(statement)
