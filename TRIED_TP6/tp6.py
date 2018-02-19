import numpy as np
import itertools

import triedtools
import triedrdf

import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from prettytable import PrettyTable

# ************************
# load the data
iris_cls = np.loadtxt('iris_cls.mat', dtype=int)
iris_don = np.loadtxt('iris_don.mat', dtype=int)
varnames = ['sepal height', 'sepal width', 'petal height', 'petal width']
leg = ['setosa', 'versicolor', 'virginica']

triedtools.plotby2(iris_don,Xclas=iris_cls,style=None,mks=10,cmap=cmx.jet,varnames=varnames,subp=1,Cvs=None,leg=None)

# ************************
# run k-nearest-neighbours

train_sizes = [0.33333333, 0.5, 0.666666666]
k_sizes = [2, 3, 4, 5, 6, 7, 8, 9, 10]
accuracy_list = []

for t in train_sizes:
    acc_temp = []
    iris_don_train, iris_don_test, iris_cls_train, iris_cls_test = \
        train_test_split(iris_don, iris_cls, test_size=1-t, random_state=42)

    for k in k_sizes:
        iris_cls_test_predicted = triedrdf.kppv(iris_don_test, iris_don_train, iris_cls_train, k).astype(int)

        acc_temp.append(accuracy_score(iris_cls_test, iris_cls_test_predicted))
        # accuracy_abs_num = accuracy_score(iris_cls_test, iris_cls_test_predicted, normalize=False)

    accuracy_list.append(acc_temp)

# ************************
# pretty table of accuracy values

# acc_ary = np.array(accuracy_list)
# k_sizes_str = ['2', '3', '4', '5', '6', '7', '10']
# train_sizes_str = ['0.333', '0.5', '0.666']
#
# ptable = PrettyTable()
# ptable.add_column('K', k_sizes_str)
# for header, col in zip(train_sizes_str, acc_ary):
#     ptable.add_column(header, col)
# ptable.align = 'r'
# print(ptable)

# ************************
# plot accuracy values

# set up values for graphs
values = range(3)

jet = plt.get_cmap('jet')
cNorm = colors.Normalize(vmin=0, vmax=values[-1])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

# create new figure
fig = plt.figure()
plt.title('Prediction accuracy for increasing K values & training set proportions')
plt.ylabel('Accuracy of prediction')
plt.xticks(k_sizes)
plt.xlabel('K nearest neighbours')

for i in values:
    colorVal = scalarMap.to_rgba(values[i])
    plt.plot(k_sizes, accuracy_list[i], color=colorVal, label=str(np.round(train_sizes[i]*100, 1)) + '%')

plt.legend()
plt.show()

# ************************
# confusion matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# make confusion matrix for best k for each of 3 train sizes:

k_optimal = [5, 3, 7]

for t, k in zip(train_sizes, k_optimal):
    iris_don_train, iris_don_test, iris_cls_train, iris_cls_test = \
        train_test_split(iris_don, iris_cls, test_size=1 - t, random_state=42)

    iris_cls_test_predicted = triedrdf.kppv(iris_don_test, iris_don_train, iris_cls_train, k).astype(int)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(iris_cls_test, iris_cls_test_predicted)
    np.set_printoptions(precision=3)

    # Plot non-normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=leg,
    #                       title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=leg, normalize=True,
                          title='Normalized confusion matrix')

    triedtools.plotby2(iris_don_test, Xclas=iris_cls_test, style=None, mks=10, cmap=cmx.jet, varnames=varnames, subp=1,
                       Cvs=iris_cls_test_predicted, leg=None)

plt.show()


# ************************
# k-means

# # todo test code for 1 run - delete
# iris_don_train, iris_don_test, iris_cls_train, iris_cls_test = \
#     train_test_split(iris_don, iris_cls, test_size=0.333, random_state=42)
#
# prototypes, iris_cls_train_predicted = triedrdf.kmoys(iris_don_train, 3)
#
# # classify prototypes with triedtools.kvotemaj
# prototypes_classes = triedtools.kvotemaj(iris_cls_train, iris_cls_train_predicted.astype(int))
#
# # classify test examples using triedrdf.kclassif
# iris_don_test_predicted = triedrdf.kclassif(iris_don_test, prototypes, clasprot=prototypes_classes)


# todo real code - complete
train_sizes = [0.33333333, 0.5, 0.666666666]
k_sizes = [2, 3, 4, 5, 6, 7, 8, 9, 10]
accuracy_list = []

for t in train_sizes:
    acc_temp = []
    iris_don_train, iris_don_test, iris_cls_train, iris_cls_test = \
        train_test_split(iris_don, iris_cls, test_size=1-t, random_state=42)

    for k in k_sizes:
        # run k-means training
        prototypes, iris_cls_train_predicted = triedrdf.kmoys(iris_don_train, k)
        plt.close('all')

        # classify prototypes with triedtools.kvotemaj
        prototypes_classes = triedtools.kvotemaj(iris_cls_train, iris_cls_train_predicted.astype(int))

        # classify test examples using triedrdf.kclassif
        iris_cls_test_predicted = triedrdf.kclassif(iris_don_test, prototypes, clasprot=prototypes_classes)

        # calculate prediction accuracy
        acc_temp.append(accuracy_score(iris_cls_test, iris_cls_test_predicted))

    accuracy_list.append(acc_temp)



# ************************
# pretty table of accuracy values

acc_ary = np.array(accuracy_list)
k_sizes_str = ['2', '3', '4', '5', '6', '7', '8', '9', '10']
train_sizes_str = ['0.333', '0.5', '0.666']

ptable = PrettyTable()
ptable.add_column('K', k_sizes_str)
for header, col in zip(train_sizes_str, acc_ary):
    ptable.add_column(header, col)
ptable.align = 'r'
print(ptable)


# ************************
# plot accuracy values

# set up values for graphs
values = range(3)

jet = plt.get_cmap('jet')
cNorm = colors.Normalize(vmin=0, vmax=values[-1])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

# create new figure
fig = plt.figure()
plt.title('Prediction accuracy for increasing K values & training set proportions')
plt.ylabel('Accuracy of prediction')
plt.xticks(k_sizes)
plt.xlabel('K nearest neighbours')

for i in values:
    colorVal = scalarMap.to_rgba(values[i])
    plt.plot(k_sizes, accuracy_list[i], color=colorVal, label=str(np.round(train_sizes[i]*100, 1)) + '%')

plt.legend()
plt.show()



# make confusion matrix for best k for each of 3 train sizes:

train_sizes = [0.33333333, 0.5, 0.666666666]
k_optimal = [8, 10, 7]

for t, k in zip(train_sizes, k_optimal):
    iris_don_train, iris_don_test, iris_cls_train, iris_cls_test = train_test_split(iris_don, iris_cls, test_size=1 - t,
                                                                                    random_state=42)

    prototypes, iris_cls_train_predicted = triedrdf.kmoys(iris_don_train, k)

    # classify prototypes with triedtools.kvotemaj
    prototypes_classes = triedtools.kvotemaj(iris_cls_train, iris_cls_train_predicted.astype(int))

    # classify test examples using triedrdf.kclassif
    iris_cls_test_predicted = triedrdf.kclassif(iris_don_test, prototypes, clasprot=prototypes_classes)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(iris_cls_test, iris_cls_test_predicted)
    np.set_printoptions(precision=3)

    # Plot non-normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=leg,
    #                       title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=leg, normalize=True, title='Normalized confusion matrix')

    triedtools.plotby2(iris_don_test, Xclas=iris_cls_test, style=None, mks=10, cmap=cmx.jet, varnames=varnames, subp=1,
                       Cvs=iris_cls_test_predicted, leg=None)

plt.show()


# *****

train_sizes = [0.33333333, 0.5, 0.666666666]
k_optimal = [8, 10, 7]
pred_list = []

iris_don_train, iris_don_test, iris_cls_train, iris_cls_test = train_test_split(iris_don, iris_cls, test_size=1 - 0.5,
                                                                                random_state=42)
for i in range(25):
    prototypes, iris_cls_train_predicted = triedrdf.kmoys(iris_don_train, 10)

    # classify prototypes with triedtools.kvotemaj
    prototypes_classes = triedtools.kvotemaj(iris_cls_train, iris_cls_train_predicted.astype(int))

    # classify test examples using triedrdf.kclassif
    iris_cls_test_predicted = triedrdf.kclassif(iris_don_test, prototypes, clasprot=prototypes_classes)

    # add iris_cls_test_predicted to list
    pred_list.append(iris_cls_test_predicted.tolist())

pred_ary = np.array(pred_list)

class_all_ary = np.ones(np.size(iris_cls_test_predicted)).astype(int)
class_diff = []
for i in range(np.size(iris_cls_test_predicted)):
    if len(np.unique(pred_ary[:, i])) == 1:
        # print('yes')
        class_diff.append(1)
    else:
        # print('no')
        class_diff.append(2)
class_diff_ary = np.array(class_diff).astype(int)


triedtools.plotby2(iris_don_test, Xclas=class_all_ary, style=['.y'], mks=10, cmap=cmx.jet, varnames=varnames, subp=1,
                   Cvs=class_diff_ary, leg=None)

# Compute confusion matrix
# cnf_matrix = confusion_matrix(iris_cls_test, iris_cls_test_predicted)
# np.set_printoptions(precision=3)

# print(accuracy_score(iris_cls_test, iris_cls_test_predicted))

# Plot non-normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=leg,
#                       title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=leg, normalize=True, title='Normalized confusion matrix')


# plt.show()

