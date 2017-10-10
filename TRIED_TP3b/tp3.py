import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.io import loadmat
import matplotlib.colors as colors
import matplotlib.cm as cmx
from scipy.interpolate import spline

import triedacp

def centre(X):
    '''
    Centres an array by subtracting mean of column from each value
    :param X: X
    :type X: nparray
    :return: centre
    :rtype: nparray
    '''
    mean = np.mean(X, 0)
    centred = X - mean
    return centred

# ************************

# load the data
X = np.loadtxt('notes.dat', dtype=float)
# print(np.shape(myarray))
# for i in myarray:
#     print(i)

X_centered = centre(X)

# Calcul les valeurs propres (eigval) et des vecteurs propres U (eigvec) de la
# matrice X'X qui sont retournées dans l'ordre de la plus grande à la plus
# petite valeur propre. Est également retourné XU qui sont les nouvelles
# coordonnées des individus (XU=X*U)
eigval, eigvec, XU = triedacp.acp(X_centered)

# ************************

fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111)
ax.set_title('PCA Projection of principal components Xu onto principal axes 1 and 2')
plt.xlabel('Principal Axis 1')
plt.ylabel('Principal Axis 2')
# plt.ylim(-15, 35)
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
values = np.arange(9).tolist()
jet = plt.get_cmap('jet')
cNorm = colors.Normalize(vmin=0, vmax=values[-1])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

plt.axhline(0, color='k')
plt.axvline(0, color='k')

for i in range(9):
    colorVal = scalarMap.to_rgba(values[i])
    plt.scatter(XU[i, 0], XU[i, 1], color=colorVal, label=labels[i], marker='o')

plt.legend()
plt.show()
fig.savefig('pca1-2.png')

# ************************

fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111)
ax.set_title('PCA Projection of principal components Xu onto principal axes 1 and 3')
plt.xlabel('Principal Axis 1')
plt.ylabel('Principal Axis 3')
# plt.ylim(-15, 35)
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
values = np.arange(9).tolist()
jet = plt.get_cmap('jet')
cNorm = colors.Normalize(vmin=0, vmax=values[-1])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

plt.axhline(0, color='k')
plt.axvline(0, color='k')

for i in range(9):
    colorVal = scalarMap.to_rgba(values[i])
    plt.scatter(XU[i, 0], XU[i, 2], color=colorVal, label=labels[i], marker='o')

plt.legend()
plt.show()
fig.savefig('pca1-3.png')

# ************************

eigval_t, eigvec_t, XU_t = triedacp.acp(X_centered.T)


# ************************

fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111)
ax.set_title('PCA Projection of principal components Xtv onto principal axes 1 and 2')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
# plt.ylim(-15, 35)
labels = ['1', '2', '3', '4', '5']
values = np.arange(5).tolist()
jet = plt.get_cmap('jet')
cNorm = colors.Normalize(vmin=0, vmax=values[-1])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

plt.axhline(0, color='k')
plt.axvline(0, color='k')

for i in range(5):
    colorVal = scalarMap.to_rgba(values[i])
    plt.scatter(XU_t[i, 0], XU_t[i, 1], color=colorVal, label=labels[i], marker='o')

plt.legend()
plt.show()
fig.savefig('pca1-2-XtV.png')

# ************************

fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111)
ax.set_title('PCA Projection of principal components Xtv onto principal axes 1 and 3')
plt.xlabel('Principal Axis 1')
plt.ylabel('Principal Axis 3')
# plt.ylim(-15, 35)
labels = ['1', '2', '3', '4', '5']
values = np.arange(5).tolist()
jet = plt.get_cmap('jet')
cNorm = colors.Normalize(vmin=0, vmax=values[-1])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

plt.axhline(0, color='k')
plt.axvline(0, color='k')

for i in range(5):
    colorVal = scalarMap.to_rgba(values[i])
    plt.scatter(XU_t[i, 0], XU_t[i, 2], color=colorVal, label=labels[i], marker='o')

plt.legend()
plt.show()
fig.savefig('pca1-3-XtV.png')

# ************************

# XtV coords in the basis of principal components = principal axes * 1/root(lambda)
XU_dual = eigvec * (1 / np.sqrt(eigval))

# ************************

fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111)
ax.set_title('PCA Projection of principal components Xtv recalculated using duality relation, onto principal axes 1 and 2')
plt.xlabel('Principal Axis 1')
plt.ylabel('Principal Axis 2')
# plt.ylim(-15, 35)
labels = ['1', '2', '3', '4', '5']
values = np.arange(5).tolist()
jet = plt.get_cmap('jet')
cNorm = colors.Normalize(vmin=0, vmax=values[-1])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

plt.axhline(0, color='k')
plt.axvline(0, color='k')

for i in range(5):
    colorVal = scalarMap.to_rgba(values[i])
    plt.scatter(XU_dual[i, 0], XU_dual[i, 1], color=colorVal, label=labels[i], marker='o')

plt.legend()
plt.show()
fig.savefig('pca1-2-XtV-dual.png')

# ************************

fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111)
ax.set_title('PCA Projection of principal components Xtv recalculated using duality relation, onto principal axes 1 and 3')
plt.xlabel('Principal Axis 1')
plt.ylabel('Principal Axis 3')
# plt.ylim(-15, 35)
labels = ['1', '2', '3', '4', '5']
values = np.arange(5).tolist()
jet = plt.get_cmap('jet')
cNorm = colors.Normalize(vmin=0, vmax=values[-1])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

plt.axhline(0, color='k')
plt.axvline(0, color='k')

for i in range(5):
    colorVal = scalarMap.to_rgba(values[i])
    plt.scatter(XU_dual[i, 0], XU_dual[i, 2], color=colorVal, label=labels[i], marker='o')

plt.legend()
plt.show()
fig.savefig('pca1-3-XtV-dual.png')

# ************************
