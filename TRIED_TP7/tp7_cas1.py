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
X, Y = TPB01_methodes.schioler(1000, sigma=0.2)

# generate 500 data points for test set
X_test, Y_test = TPB01_methodes.schioler(500, sigma=0.2)

# plot test on its own
fig = plt.figure()
plt.title('Test dataset')
plt.xlabel('X value')
plt.ylabel('Y value')

plt.scatter(X_test, Y_test, color='r', label='Test', edgecolors='w', s=40)

plt.legend()
plt.tight_layout()
plt.show()
fig.savefig('test_dataset.png')


# split 1000 into train and validation by taking every other point
X_train = X[0::2]
Y_train = Y[0::2]

X_validation = X[1::2]
Y_validation = Y[1::2]

# intervals
# - 1er cas : [-0.75 -0.25] U [1 1.25 ]
a = -0.75
b = -0.25
c = 1
d = 1.25

# - 2èmecas : [-0.75 -0.25] U [1 2.00 ]
# a = -0.75
# b = -0.25
# c = 1
# d = 2.00

X_train_masked = []
Y_train_masked = []
X_validation_masked = []
Y_validation_masked = []

# get indices of intervals
acount = 0
bcount = 0
ccount = 0
dcount = 0
ecount = 0
for i, j in zip(X_train, Y_train):
    if i < a:
        acount = acount + 1
    elif i < b:
        bcount = bcount + 1
    elif i < c:
        ccount = ccount + 1
    elif i < d:
        dcount = dcount + 1
    else:
        ecount = ecount + 1

print('acount = ' + str(acount))
print('bcount = ' + str(bcount))
print('ccount = ' + str(ccount))
print('dcount = ' + str(dcount))
print('ecount = ' + str(ecount))

# subtract values that lie in the intervals
for i, j in zip(X_train, Y_train):
    if a <= i <= b or c <= i <= d:
        continue
    else:
        X_train_masked.append(i)
        Y_train_masked.append(j)

for i, j in zip(X_validation, Y_validation):
    if a <= i <= b or c <= i <= d:
        continue
    else:
        X_validation_masked.append(i)
        Y_validation_masked.append(j)

X_train_masked = np.array(X_train_masked)
Y_train_masked = np.array(Y_train_masked)
X_validation_masked = np.array(X_validation_masked)
Y_validation_masked = np.array(Y_validation_masked)

# plot train and validation together
fig = plt.figure()
plt.title('Train and validation datasets, step = 2')
plt.xlabel('X value')
plt.ylabel('Y value')
plt.xlim(-2.2, 2.2)
plt.ylim(-1.65, 1.6)

plt.scatter(X_train_masked, Y_train_masked, color='b', label='Train', edgecolors='w', s=40)
plt.scatter(X_validation_masked, Y_validation_masked, color='g', label='Validation', edgecolors='w', s=40)

plt.legend()
plt.tight_layout()
plt.show()
fig.savefig('train_valid_cas1.png')


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

# init and train NN with masked data
# then calculate regression Y values for training/validation/test sets
for m in m_list:
    # initialise the NN
    weights = trieddeep.pmcinit(X_train_masked, Y_train_masked, m, k=0.1)
    # Train the NN
    weights = trieddeep.pmctrain(X_train_masked, Y_train_masked, weights, transfer_functions,
                                 nbitemax=nbitemax, dfreq=dfreq, dcurves=dcurves)

    # Get regression values for training set
    Y_train_regression = trieddeep.pmcout(X_train_masked, weights, transfer_functions)
    # add results to the list
    Y_train_regression_list.append(Y_train_regression)

    # Get regression values for validation set
    Y_validation_regression = trieddeep.pmcout(X_validation_masked, weights, transfer_functions)
    # add results to the list
    Y_validation_regression_list.append(Y_validation_regression)

    # Get regression values for test set
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
plt.scatter(X_train_masked, Y_train_masked, color='tab:gray', label='Train', s=30)
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

# plot regression curves for training data
for i in values:
    colorVal = scalarMap.to_rgba(values[i])
    plt.plot(X_train_masked, Y_train_regression_list[i], color=colorVal, label='m = ' + str(m_list[i]))

plt.legend()
plt.tight_layout()
plt.show()
fig.savefig('regression_cas1.png')

# **********
# plot test set regression curves
fig = plt.figure()
plt.title('Training dataset & NN regression of test data for varying m')
plt.xlabel('X value')
plt.ylabel('Y value')
plt.xlim(-2.2, 2.2)
plt.ylim(-1.65, 1.6)
# plot scatter of training data
plt.scatter(X_train_masked, Y_train_masked, color='tab:gray', label='Train', s=30)

# plot sin function
plt.plot(X_test, sin_vals, color='k', label='sin function')

# plot regression curves for training data
for i in values:
    colorVal = scalarMap.to_rgba(values[i])
    plt.plot(X_test, Y_test_regression_list[i], color=colorVal, label='m = ' + str(m_list[i]))

plt.legend()
plt.tight_layout()
plt.show()
fig.savefig('regression_test_cas1.png')

# ******************
# calculate the RMS errors for all dataset/m combos

error_train = []
error_validation = []
error_test = []
error_test_int1 = []
error_test_int2 = []
error_test_int3 = []
error_test_int4 = []
error_test_int5 = []
for i in range(len(m_list)):
    temp = triedtools.errors(Y_train_masked, Y_train_regression_list[i])
    error_train.append(temp[0])

    temp = triedtools.errors(Y_validation_masked, Y_validation_regression_list[i])
    error_validation.append(temp[0])

    temp = triedtools.errors(Y_test, Y_test_regression_list[i])
    error_test.append(temp[0])

    # interval 1
    temp = triedtools.errors(Y_test[0:160, :], Y_test_regression_list[i][0:160, :])
    error_test_int1.append(temp[0])

    # interval 2
    temp = triedtools.errors(Y_test[160:230, :], Y_test_regression_list[i][160:230, :])
    error_test_int2.append(temp[0])

    # interval 3
    temp = triedtools.errors(Y_test[230:381, :], Y_test_regression_list[i][230:381, :])
    error_test_int3.append(temp[0])

    # interval 4
    temp = triedtools.errors(Y_test[381:405, :], Y_test_regression_list[i][381:405, :])
    error_test_int4.append(temp[0])
    
    # interval 5
    temp = triedtools.errors(Y_test[405:, :], Y_test_regression_list[i][405:, :])
    error_test_int5.append(temp[0])

# plot error for whole dataset
fig = plt.figure()
plt.title('NN test data regression error as a function of m')
plt.xlabel('M perceptrons in hidden layer')
plt.ylabel('Error')

# plot sin function
plt.plot(m_list, error_train, 'bo-', label='Train')
plt.plot(m_list, error_validation, 'go-', label='Validation')
plt.plot(m_list, error_test, 'ro-', label='Test')

plt.legend()
plt.tight_layout()
plt.show()
fig.savefig('error_cas1.png')


# plot intervals in same way
fig = plt.figure()
plt.title('NN test data interval regression error as a function of m')
plt.xlabel('M perceptrons in hidden layer')
plt.ylabel('Error')

# plot intervals by m
# - 1er cas : [-0.75 -0.25] U [1 1.25 ]
plt.plot(m_list, error_test_int1, 'bo-', label='Int1: -2.00 < x < -0.75 (data)')
plt.plot(m_list, error_test_int2, 'go-', label='Int2: -0.75 < x < -0.25 (gap)')
plt.plot(m_list, error_test_int3, 'ro-', label='Int3: -0.25 < x <  1.00 (data)')
plt.plot(m_list, error_test_int4, 'yo-', label='Int4:  1.00 < x <  1.25 (gap)')
plt.plot(m_list, error_test_int5, 'mo-', label='Int5:  1.25 < x <  2.00 (data)')

plt.legend()
plt.tight_layout()
plt.show()
fig.savefig('error_cas1_int.png')

# ************************
# pretty table of error values

ptable = PrettyTable()
ptable.add_column('M', m_list)
ptable.add_column('Train', np.round(error_train, 5))
ptable.add_column('Validation', np.round(error_validation, 5))
ptable.add_column('Test', np.round(error_test, 5))
ptable.add_column('Test_int1', np.round(error_test_int1, 5))
ptable.add_column('Test_int2', np.round(error_test_int2, 5))
ptable.add_column('Test_int3', np.round(error_test_int3, 5))
ptable.add_column('Test_int4', np.round(error_test_int4, 5))
ptable.add_column('Test_int5', np.round(error_test_int5, 5))
ptable.align = 'r'

with open('out_cas1.txt', 'w') as f:
    f.write(str(ptable))
    f.write('\n\n')

print(ptable)



