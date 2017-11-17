import numpy as np
import TPB01_methodes
import trieddeep
import triedtools

import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors

import math
from prettytable import PrettyTable

# initialise seed
np.random.seed(0)    # (Re)positionnement du random (ou pas)

# generate 1000 data points for train & validation
X, Y = TPB01_methodes.schioler(1000, sigma=0.5)

# generate 500 data points for test set
X_test, Y_test = TPB01_methodes.schioler(500, sigma=0.5)

# plot test on its own
fig = plt.figure()
plt.title('Test dataset')
plt.xlabel('X value')
plt.ylabel('Y value')
plt.xlim(-2.2, 2.2)
plt.ylim(-1.65, 1.6)

plt.scatter(X_test, Y_test, color='r', label='Test', edgecolors='w', s=40)

plt.legend()
plt.tight_layout()
plt.show()
fig.savefig('noisy_test_dataset.png')


steps = [2, 3, 4, 5, 10, 30, 100]

for j in steps:

    # split 1000 into train and validation by taking every other point
    X_train = X[0::j]
    Y_train = Y[0::j]

    X_validation = X[1::j]
    Y_validation = Y[1::j]

    # plot train and validation together
    fig = plt.figure()
    plt.title('Train and validation datasets, step = ' + str(j) + ', noise sigma = 0.5')
    plt.xlabel('X value')
    plt.ylabel('Y value')
    plt.xlim(-2.2, 2.2)
    plt.ylim(-1.65, 1.6)

    plt.scatter(X_train, Y_train, color='b', label='Train', edgecolors='w', s=40)
    plt.scatter(X_validation, Y_validation, color='g', label='Validation', edgecolors='w', s=40)

    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig('noisy_train_valid_n_' + str(j) + '.png')


    # ******************
    # train NN with increasing m (number of elements in hidden layer)
    # visualise regression curve for each value of m

    # Transfer functions
    F1 = "tah"
    F2 = "lin"
    transfer_functions = [F1, F2]

    # Learning parameters
    nbitemax = 2000
    dfreq = 200
    dcurves = 2

    # number of perceptrons in hidden layers
    m_list = [1, 3, 5, 7, 9]
    Y_train_regression_list = []
    Y_validation_regression_list = []
    Y_test_regression_list = []

    # calculate Y values for training set
    for m in m_list:
        # initialise the NN
        weights = trieddeep.pmcinit(X_train, Y_train, m, k=0.1)
        # Train the NN
        weights = trieddeep.pmctrain(X_train, Y_train, weights, transfer_functions,
                                     nbitemax=nbitemax, dfreq=dfreq, dcurves=dcurves)
        # Get regression
        Y_train_regression = trieddeep.pmcout(X_train, weights, transfer_functions)
        # add results to the list
        Y_train_regression_list.append(Y_train_regression)

        # Get regression
        Y_validation_regression = trieddeep.pmcout(X_validation, weights, transfer_functions)
        # add results to the list
        Y_validation_regression_list.append(Y_validation_regression)

        # Get regression
        Y_test_regression = trieddeep.pmcout(X_test, weights, transfer_functions)
        # add results to the list
        Y_test_regression_list.append(Y_test_regression)


    # plot scatter of 5 regression curves, plus actual function
    # set up plot vars
    values = range(5)
    jet = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=0, vmax=values[-1])
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

    # **********
    # plot training set regression curves
    fig = plt.figure()
    plt.title('Training dataset & NN regression of training data for varying m')
    plt.xlabel('X value')
    plt.ylabel('Y value')
    plt.xlim(-2.2, 2.2)
    plt.ylim(-1.65, 1.6)
    # plot scatter of training data
    plt.scatter(X_train, Y_train, color='tab:gray', label='Train', s=30)
    # plot actual sin function
    sin_vals = []
    for i in X_test:
        if i < -1:
            sin_vals.append(0)
        elif i > 1:
            sin_vals.append(0)
        else:
            sin_vals.append(math.sin(i * math.pi))

    # plot sin function
    plt.plot(X_test, sin_vals, color='k', label='sin function')

    # plot regression curves
    for i in values:
        colorVal = scalarMap.to_rgba(values[i])
        plt.plot(X_train, Y_train_regression_list[i], color=colorVal, label='m = ' + str(m_list[i]))

    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig('noisy_regression_n_' + str(j) + '.png')

    # **********
    # plot test set regression curves
    fig = plt.figure()
    plt.title('Training dataset & NN regression of test data for varying m')
    plt.xlabel('X value')
    plt.ylabel('Y value')
    plt.xlim(-2.2, 2.2)
    plt.ylim(-1.65, 1.6)
    # plot scatter of training data
    plt.scatter(X_train, Y_train, color='tab:gray', label='Train', s=30)

    # plot sin function
    plt.plot(X_test, sin_vals, color='k', label='sin function')

    # plot regression curves
    for i in values:
        colorVal = scalarMap.to_rgba(values[i])
        plt.plot(X_test, Y_test_regression_list[i], color=colorVal, label='m = ' + str(m_list[i]))

    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig('noisy_regression_test_n_' + str(j) + '.png')

    # ******************
    # calculate the RMS errors for all dataset/m combos

    error_train = []
    error_validation = []
    error_test = []
    for i in range(len(m_list)):
        temp = triedtools.errors(Y_train, Y_train_regression_list[i])
        error_train.append(temp[0])

        temp = triedtools.errors(Y_validation, Y_validation_regression_list[i])
        error_validation.append(temp[0])

        temp = triedtools.errors(Y_test, Y_test_regression_list[i])
        error_test.append(temp[0])


    fig = plt.figure()
    plt.title('NN test data regression error as a function of m')
    plt.xlabel('M perceptrons in hidden layer')
    plt.ylabel('Error')

    # plot sin function
    plt.plot(m_list, error_train, color='b', label='Train')
    plt.plot(m_list, error_validation, color='g', label='Validation')
    plt.plot(m_list, error_test, color='r', label='Test')

    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig('noisy_error_n_' + str(j) + '.png')

    # ************************
    # pretty table of error values

    ptable = PrettyTable()
    ptable.add_column('M', m_list)
    ptable.add_column('Train', np.round(error_train, 5))
    ptable.add_column('Validation', np.round(error_validation, 5))
    ptable.add_column('Test', np.round(error_test, 5))
    ptable.align = 'r'

    with open('noisy_out.txt', 'a') as f:
        f.write('n = ' + str(j) + '\n')
        f.write(str(ptable))
        f.write('\n\n')

    print(ptable)

# ************************
# intervals


