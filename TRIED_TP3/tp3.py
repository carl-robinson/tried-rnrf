import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.io import loadmat
import matplotlib.colors as colors
import matplotlib.cm as cmx
from scipy.interpolate import spline
from prettytable import PrettyTable


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
# projecting X onto eigenvectors of XXt matrix

eigval_t, eigvec_t, XU_t = triedacp.acp(X_centered.T)


# ************************
# projecting X onto eigenvectors of XXt matrix

fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111)
ax.set_title('PCA Projection of principal components Xtv onto principal axes 1 and 2')
plt.xlabel('Principal Axis 1')
plt.ylabel('Principal Axis 2')
# plt.ylim(-15, 35)
labels = ['Maths', 'Phys', 'Fran', 'Latin', 'Dessin']
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
# projecting X onto eigenvectors of XXt matrix

fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111)
ax.set_title('PCA Projection of principal components Xtv onto principal axes 1 and 3')
plt.xlabel('Principal Axis 1')
plt.ylabel('Principal Axis 3')
# plt.ylim(-15, 35)
labels = ['Maths', 'Phys', 'Fran', 'Latin', 'Dessin']
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
# Calculating XXt (using the duality principal) as Ui * root(lambda)
# XtV coords in the basis of principal components = principal axes of U * root of lambdas of U
# makes 5x5 array
XtV_dual = eigvec * np.sqrt(eigval)

# ************************

fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111)
ax.set_title('Xtv coords using duality relation - axes 1 and 2')
plt.xlabel('Principal Axis 1')
plt.ylabel('Principal Axis 2')
# plt.ylim(-15, 35)
labels = ['Maths', 'Phys', 'Fran', 'Latin', 'Dessin']
values = np.arange(5).tolist()
jet = plt.get_cmap('jet')
cNorm = colors.Normalize(vmin=0, vmax=values[-1])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

plt.axhline(0, color='k')
plt.axvline(0, color='k')

for i in range(5):
    colorVal = scalarMap.to_rgba(values[i])
    plt.scatter(XtV_dual[i, 0], XtV_dual[i, 1], color=colorVal, label=labels[i], marker='o')


plt.legend()
plt.show()
fig.savefig('pca1-2-XtV_dual.png')

# ************************

fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111)
ax.set_title('Xtv coords using duality relation - axes 1 and 3')
plt.xlabel('Principal Axis 1')
plt.ylabel('Principal Axis 3')
# plt.ylim(-15, 35)
labels = ['Maths', 'Phys', 'Fran', 'Latin', 'Dessin']
values = np.arange(5).tolist()
jet = plt.get_cmap('jet')
cNorm = colors.Normalize(vmin=0, vmax=values[-1])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

plt.axhline(0, color='k')
plt.axvline(0, color='k')

for i in range(5):
    colorVal = scalarMap.to_rgba(values[i])
    plt.scatter(XtV_dual[i, 0], XtV_dual[i, 2], color=colorVal, label=labels[i], marker='o')

plt.legend()
plt.show()
fig.savefig('pca1-3-XtV_dual.png')

# ************************
# 9x5 / 5
# XUi / root(Li)
# need to divide each element in column i by the corresponding lambda row i

eigvec_v = XU / np.sqrt(eigval)
XtV_dual_v = np.dot(X.T, eigvec_v)

# ************************
# print out pretty table of X

headers = ['Maths', 'Phys', 'Fran', 'Latin', 'Dessin']
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']

ptable = PrettyTable()
ptable.add_column('Student', labels)
for header, col in zip(headers, np.round(X.T, 3)):
    ptable.add_column(header, col)
ptable.align = 'r'
print(ptable)

# print out pretty table of X_centered
ptable = PrettyTable()
ptable.add_column('Student', labels)
for header, col in zip(headers, np.round(X_centered.T, 10)):
    ptable.add_column(header, col)
ptable.align = 'r'
print(ptable)

# ************************
# print out pretty table of eigval and eigvec

headers = ['Dim 1', 'Dim 2', 'Dim 3', 'Dim 4', 'Dim 5']
labels = ['1', '2', '3', '4', '5']

ptable = PrettyTable()
ptable.add_column('Eigenvalue ID', labels)
ptable.add_column('Value', np.round(eigval, 3))
ptable.align = 'r'
print(ptable)

# print out pretty table of X_centered
ptable = PrettyTable()
ptable.add_column('Eigenvector ID', labels)
for header, col in zip(headers, np.round(eigvec.T, 3)):
    ptable.add_column(header, col)
ptable.align = 'r'
print(ptable)

# ************************
# print out pretty table of XU
# rows = 9 x students
# cols = student id + 5 x coord on axes 1-5

headers = ['Axis 1', 'Axis 2', 'Axis 3', 'Axis 4', 'Axis 5']
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']

# print out pretty table of X_centered
ptable = PrettyTable()
ptable.add_column('Student', labels)
for header, col in zip(headers, np.round(XU.T, 3)):
    ptable.add_column(header, col)
ptable.align = 'r'
print(ptable)

# ************************
# print out pretty table of eigval_t and eigvec_t

headers = ['Dim 1', 'Dim 2', 'Dim 3', 'Dim 4', 'Dim 5', 'Dim 6', 'Dim 7', 'Dim 8', 'Dim 9']
labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

ptable = PrettyTable()
ptable.add_column('Eigenvalue ID', labels)
ptable.add_column('Value', np.round(eigval_t, 3))
ptable.align = 'r'
print(ptable)

# print out pretty table of X_centered
ptable = PrettyTable()
ptable.add_column('Eigenvector ID', labels)
for header, col in zip(headers, np.round(eigvec_t.T, 3)):
    ptable.add_column(header, col)
ptable.align = 'r'
print(ptable)

# ************************
# print out pretty table of XU_t
# rows = 9 x students
# cols = student id + 5 x coord on axes 1-5

headers = ['Axis 1', 'Axis 2', 'Axis 3', 'Axis 4', 'Axis 5']
labels = ['Maths', 'Phys', 'Fran', 'Latin', 'Dessin']

# print out pretty table of X_centered
ptable = PrettyTable()
ptable.add_column('Subject', labels)
for header, col in zip(headers, np.round(XU_t.T, 3)):
    ptable.add_column(header, col)
ptable.align = 'r'
print(ptable)

# ************************
# print out pretty table of XtV_dual

headers = ['Axis 1', 'Axis 2', 'Axis 3', 'Axis 4', 'Axis 5']
labels = ['Maths', 'Phys', 'Fran', 'Latin', 'Dessin']

# print out pretty table of X_centered
ptable = PrettyTable()
ptable.add_column('Subject', labels)
for header, col in zip(headers, np.round(XtV_dual.T, 3)):
    ptable.add_column(header, col)
ptable.align = 'r'
print(ptable)

# ************************
# print out pretty table of XtV_dual_v

headers = ['Axis 1', 'Axis 2', 'Axis 3', 'Axis 4', 'Axis 5']
labels = ['Maths', 'Phys', 'Fran', 'Latin', 'Dessin']

# print out pretty table of X_centered
ptable = PrettyTable()
ptable.add_column('Subject', labels)
for header, col in zip(headers, np.round(XtV_dual_v.T, 3)):
    ptable.add_column(header, col)
ptable.align = 'r'
print(ptable)

# ************************
# remake X using U, V and lambda
# that's eigvec 5x5, eigvec_v 9x5 and eigval 5
# should get 9x5 out (same as X) - which is 9x? * ?x5

# A diagonalizable matrix is diagonalized by a matrix of its eigenvectors.
# So A=PΛP−1A=PΛP−1, where PP is the matrix whose columns are the eigenvectors of
# AA and ΛΛ is a diagonal matrix whose diagonal entries are the eigenvalues of AA (listed
# in the same order as their corresponding eigenvectors listed in PP).
# So just multiply PΛP−1PΛP−1 to reconstruct AA.

# X_dual_1_dot = np.dot(eigvec_v, eigvec)
# X_dual_1 = X_dual_1_dot * np.sqrt(eigval)

# eigvec_v_t = eigvec_v[:, 0]
# eigvec_v_t = np.array(eigvec_v_t)
# eigvec_v_t = eigvec_v_t.transpose()

x1_m = eigvec_v[:, 0] * np.sqrt(eigval[0])
x1 = np.asmatrix(x1_m).reshape(9, 1) * np.asmatrix(eigvec[:, 0].T)

x2_m = eigvec_v[:, 1] * np.sqrt(eigval[1])
x2 = np.asmatrix(x2_m).reshape(9, 1) * np.asmatrix(eigvec[:, 1].T)

x3_m = eigvec_v[:, 2] * np.sqrt(eigval[2])
x3 = np.asmatrix(x3_m).reshape(9, 1) * np.asmatrix(eigvec[:, 2].T)

x4_m = eigvec_v[:, 3] * np.sqrt(eigval[3])
x4 = np.asmatrix(x4_m).reshape(9, 1) * np.asmatrix(eigvec[:, 3].T)

x5_m = eigvec_v[:, 4] * np.sqrt(eigval[4])
x5 = np.asmatrix(x5_m).reshape(9, 1) * np.asmatrix(eigvec[:, 4].T)

X_reconstructed = x1 + x2 + x3 + x4 + x5
X_reconstructed = np.array(X_reconstructed)

X_reconstructed_3 = x1 + x2 + x3
X_reconstructed_3 = np.array(X_reconstructed_3)

# ************************
# print out pretty table of X_dual_1
# rows = 9 x students
# cols = student id + 5 x coord on axes 1-5

headers = ['Maths', 'Phys', 'Fran', 'Latin', 'Dessin']
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']

# print out pretty table of X_centered
ptable = PrettyTable()
ptable.add_column('Student', labels)
for header, col in zip(headers, np.round(X_reconstructed_3.T, 10)):
    ptable.add_column(header, col)
ptable.align = 'r'
print(ptable)

# ************************
# Calculate difference between X_reconstructed_3 and X_centered

X_diff = X_centered - X_reconstructed_3

headers = ['Maths', 'Phys', 'Fran', 'Latin', 'Dessin']
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']

# print out pretty table of X_centered
ptable = PrettyTable()
ptable.add_column('Student', labels)
for header, col in zip(headers, np.round(X_diff.T, 3)):
    ptable.add_column(header, col)
ptable.align = 'r'
print(ptable)

# ************************
# Calculate correlation matrix

# Calculate centered-reduced data
X_std = np.std(X, 0)
X_centered_reduced = X_centered[:, None] / X_std

# this doesn't work...
# data - vector[:,None]
# X_correlation_matrix = X_centered_reduced / np.sqrt(np.size(X) - 1)
# X_centered_reduced_cov = np.cov(X_centered_reduced)

# this does...
X_correlation_matrix_calc = np.corrcoef(X.T)

# print out pretty table of X_correlation_matrix_calc
headers = ['Maths', 'Phys', 'Fran', 'Latin', 'Dessin']
labels = ['Maths', 'Phys', 'Fran', 'Latin', 'Dessin']

# print out pretty table of X_centered
ptable = PrettyTable()
ptable.add_column('Variable', labels)
for header, col in zip(headers, np.round(X_correlation_matrix_calc.T, 3)):
    ptable.add_column(header, col)
ptable.align = 'r'
print(ptable)

# ************************
# Calculate eigenvalues and eigenvectors for centered-reduced data
eigval_cr, eigvec_cr, XU_cr = triedacp.acp(X_centered_reduced[:, 0, :])

# ************************

fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111)
ax.set_title('Projection of centered-reduced X onto principal axes 1 and 2 in Rp')
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
    plt.scatter(XU_cr[i, 0], XU_cr[i, 1], color=colorVal, label=labels[i], marker='o')

plt.legend()
plt.show()
fig.savefig('pca1-2-cr.png')

# ************************

fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111)
ax.set_title('Projection of centered-reduced X onto principal axes 1 and 3 in Rp')
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
    plt.scatter(XU_cr[i, 0], XU_cr[i, 2], color=colorVal, label=labels[i], marker='o')

plt.legend()
plt.show()
fig.savefig('pca1-3-cr.png')

# ************************
# Calculating XXt (using the duality principal) as Ui * root(lambda)
# XtV coords in the basis of principal components = principal axes of U * root of lambdas of U
# makes 5x5 array
XtV_dual_cr = eigvec_cr * np.sqrt(eigval_cr)

# ************************

fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111)
ax.set_title('Centered-reduced Xtv coords using duality - axes 1 and 2')
plt.xlabel('Principal Axis 1')
plt.ylabel('Principal Axis 2')
# plt.ylim(-15, 35)
labels = ['Maths', 'Phys', 'Fran', 'Latin', 'Dessin']
values = np.arange(5).tolist()
jet = plt.get_cmap('jet')
cNorm = colors.Normalize(vmin=0, vmax=values[-1])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

plt.axhline(0, color='k')
plt.axvline(0, color='k')

for i in range(5):
    colorVal = scalarMap.to_rgba(values[i])
    plt.scatter(XtV_dual_cr[i, 0], XtV_dual_cr[i, 1], color=colorVal, label=labels[i], marker='o')

plt.legend()
plt.show()
fig.savefig('pca1-2-XtV_dual-cr.png')

# ************************

fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111)
ax.set_title('Centered-reduced Xtv coords using duality - axes 1 and 3')
plt.xlabel('Principal Axis 1')
plt.ylabel('Principal Axis 3')
# plt.ylim(-15, 35)
labels = ['Maths', 'Phys', 'Fran', 'Latin', 'Dessin']
values = np.arange(5).tolist()
jet = plt.get_cmap('jet')
cNorm = colors.Normalize(vmin=0, vmax=values[-1])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

plt.axhline(0, color='k')
plt.axvline(0, color='k')

for i in range(5):
    colorVal = scalarMap.to_rgba(values[i])
    plt.scatter(XtV_dual_cr[i, 0], XtV_dual_cr[i, 2], color=colorVal, label=labels[i], marker='o')

plt.legend()
plt.show()
fig.savefig('pca1-3-XtV_dual-cr.png')

# ************************
# plot correlation circle

import pandas as pd
import prince

# df = pd.DataFrame(data=XtV_dual_cr)
df = pd.DataFrame(data=X)

pca = prince.PCA(df, n_components=5)

fig1, ax1 = pca.plot_correlation_circle(axes=(0, 1), show_labels=True)
fig1.show()

fig2, ax2 = pca.plot_correlation_circle(axes=(0, 2), show_labels=True)
fig2.show()

# fig1.savefig('pca_cumulative_inertia.png', bbox_inches='tight', pad_inches=0.5)
# fig2.savefig('pca_row_principal_coordinates.png', bbox_inches='tight', pad_inches=0.5)
# fig3.savefig('pca_correlation_circle.png', bbox_inches='tight', pad_inches=0.5)