import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.io import loadmat
import matplotlib.colors as colors
import matplotlib.cm as cmx
from scipy.interpolate import spline

import triedacp

def centre(X, axis):
    '''
    Centres an array by subtracting mean of column from each value
    :param X: X
    :type X: nparray
    :return: centre
    :rtype: nparray
    '''
    mean = np.mean(X, axis)
    centred = X - mean
    return centred


# ************************
# load the data

X = np.loadtxt('notes.dat', dtype=float)
# print(np.shape(myarray))
# for i in myarray:
#     print(i)

# calc means for each student (rows)
mean = np.mean(X, 1)


# ************************
# calc correlation values

students = ['Alain', 'Benoit', 'Cyril', 'Daisy', 'Emilie', 'Fanny', 'Gaétan', 'Hélène', 'Inès']
subjects = ['Maths', 'Phys', 'Fran', 'Latin', 'Dessin']
coef_2d_list = []
# print(np.shape(coef_table))

# for each subject
for index1 in np.arange(len(subjects)):
    # create an empty list to add corr values into
    inner_list = []
    # get col of subject scores
    subj1 = X[:, index1]
    # for each subject
    for index2 in np.arange(len(subjects)):
        # get column of temp data
        subj2 = X[:, index2]
        # get correlation coefficient matrix between the two towns
        corr = np.corrcoef(subj1, subj2)
        # append correlation value to the inner list
        inner_list.append(corr[0, 1])
        # print(corr[0,1])
    # append the inner list to the main list
    coef_2d_list.append(inner_list)
    coef_2d_array = np.array(coef_2d_list)

# ************************
# draw corrcoeff figure

fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111)
ax.set_title('Coefficients de corrélation linéaires des sujets', y=1.08)
plt.imshow(coef_2d_array)
ax.set_aspect('equal')
stud_ax = list(subjects)
stud_ax.insert(0,'')

# print(stud_ax)
ax.set_xticklabels(stud_ax)
ax.xaxis.set_tick_params(labeltop='on')
ax.set_yticklabels(stud_ax)

cax = fig.add_axes([0.2, 0.1, 0.78, 0.8])
cax.get_xaxis().set_visible(False)
cax.get_yaxis().set_visible(False)
cax.patch.set_alpha(0)
cax.set_frame_on(False)
plt.colorbar(orientation='vertical')
plt.show()

fig.savefig('corrcoef.png')
print(coef_2d_array)

# ************************
# draw line plot for each student across all subjects

fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111)
ax.set_title('notes des étudiants sur 5 sujets', y=1.08)
plt.xticks(range(len(subjects)), subjects)

values = np.arange(len(students)).tolist()
jet = plt.get_cmap('jet')
cNorm = colors.Normalize(vmin=0, vmax=values[-1])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

# for each subject
for i in np.arange(len(students)):
    # plot a line across all subjects
    colorVal = scalarMap.to_rgba(values[i])
    plt.plot(np.arange(np.size(X, 1)), X[i, :], color=colorVal, marker='o', label=students[i])

# for j in np.arange(len(subjects)):
plt.legend()
plt.show()

fig.savefig('lineplot_studentgrades.png')

# ************************
# draw line plot for each student across all students

fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111)
ax.set_title('notes des sujets sur 9 etudiant', y=1.08)
plt.xticks(range(len(students)), students)

values = np.arange(len(subjects)).tolist()
jet = plt.get_cmap('jet')
cNorm = colors.Normalize(vmin=0, vmax=values[-1])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

# for each subject
for i in np.arange(len(subjects)):
    # plot a line across all subjects
    colorVal = scalarMap.to_rgba(values[i])
    plt.plot(np.arange(np.size(X, 0)), X[:, i], color=colorVal, marker='o', label=subjects[i])

# for j in np.arange(len(subjects)):
plt.legend()
plt.show()

fig.savefig('lineplot_subjectgrades.png')


# ************************
# centre the variables and get eigenvalues & eigenvectors

# centre the variables by subtracting mean of each feature
X_centered = centre(X, 0)

eigval, eigvec, XU = triedacp.acp(X_centered)

# ************************
# plot eigenvalues as bar chart, and cumulative inertia as line plot

inertia, cumulative_inertia = triedacp.phinertie(eigval)
fig.savefig('bar_eigen_line_cumul_inertia.png')

# ************************
#  qual : Les qualités de réprésentation des individus par les axes
#  contrib: Les contributions des individus à la formation des axes

qual, contrib = triedacp.qltctr2(XU, eigval)

# verify that the sum of representations of each individual equals one
qual_sum = np.sum(qual, 1)
# output is:
# [ 1.  1.  1.  1.  1.  1.  1.  1.  1.]

# verify that the sum of contributions to each axis equals one
contrib_sum = np.sum(contrib, 0)
# output is:
# [ 1.  1.  1.  1.  1.]

# ************************
# calc XtV coords in the basis of principal components = principal axes * 1/root(lambda)
XtV = eigvec * np.sqrt(eigval)

# ************************
# plot cloud of principal components on axes 1-2, 2-3, 1-3


for inc in [0, 1, 2]:
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111)
    ax.set_title('PCA Projection of principal components Xu onto principal axes ' + str((inc % 3) + 1) + ' and ' + str(((inc + 1) % 3) + 1))
    plt.xlabel('Principal Axis ' + str((inc % 3) + 1))
    plt.ylabel('Principal Axis ' + str(((inc + 1) % 3) + 1))
    # plt.ylim(-15, 35)
    students = ['Alain', 'Benoit', 'Cyril', 'Daisy', 'Emilie', 'Fanny', 'Gaétan', 'Hélène', 'Inès']
    values = np.arange(len(students)).tolist()
    jet = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=0, vmax=values[-1])
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    plt.axhline(0, color='k')
    plt.axvline(0, color='k')

    for i in range(len(students)):
        colorVal = scalarMap.to_rgba(values[i])
        # takes columns 0-1, 1-2, 2-0
        plt.scatter(XU[i, inc % 3], XU[i, (inc + 1) % 3], color=colorVal, label=students[i], marker='o')

    plt.legend()
    plt.show()
    fig.savefig('pca-tp3b-axes-' + str((inc % 3) + 1) + '-' + str(((inc + 1) % 3) + 1) + '.png')



# plt.close('all')