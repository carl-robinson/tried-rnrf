import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.io import loadmat
import matplotlib.colors as colors
import matplotlib.cm as cmx
import pandas as pd
import prince
import triedacp
import triedtools


def centre_and_reduce(X):
    '''
    Centres an array by subtracting mean of column from each value
    :param X: X
    :type X: nparray
    :return: centre
    :rtype: nparray
    '''
    mean = np.mean(X, 0)
    centered = X - mean

    stddev = np.std(centered, 0)
    # centered_reduced = centered[:, None] / stddev
    centered_reduced = centered[:, :] / stddev
    return centered_reduced


def tt_cerclecor():
    ''' Trace un cercle (de rayon 1 et de centre 0) pour le cercle des corrélations
    '''
    # plt.figure();
    # Construire et tracer un cercle de rayon 1 et de centre 0
    t = np.linspace(-np.pi, np.pi, 50);
    x = np.cos(t);
    y = np.sin(t);
    plt.plot(x, y, '-r');  # trace le cercle
    plt.axis("equal");
    # Tracer les axes
    xlim = plt.xlim();
    plt.xlim(xlim);
    plt.plot(xlim, np.zeros(2));
    ylim = plt.ylim();
    plt.ylim(ylim);
    plt.plot(np.zeros(2), ylim);


def tt_corcer(X, XU, pa, po, varnames, shape='o', coul='b', markersize=8, fontsize=11, start=1, step=1):
    '''corcer (X,XU,pa,po,varnames,shape,coul,markersize,fontsize)
    | Dessine le cercle des corrélations (CC)
    | X        : Les données de départ (qui peuvent avoir été transformées ou pas)
    | XU       : Les nouvelles coordonnées des indivisus d'une acp
    | pa-po    : Le plan d'axe pa-po d'une acp pour lequelle on veut le CC.
    | varnames : Les noms des variables.
    | shape    : La forme des points du nuage
    | coul     : Couleur des points du nuage (à choisir parmi les caractère permi
    |            de la fonction plot de matplotlib.
    | markersize : Taille des points
    | fontsize   : Taille du texte
    '''
    pa = pa - 1;
    po = po - 1;
    p = np.size(XU, 1);
    tt_cerclecor();
    # Déterminer les corrélations et les ploter
    XUab = XU[:, [pa, po]];
    W = np.concatenate((X, XUab), axis=1);
    R = np.corrcoef(W.T);
    a = R[0:p, p];
    b = R[0:p, p + 1];
    #
    plt.plot(a[start::step], b[start::step], shape, color=coul, markersize=markersize);
    # for i in range(p):
    #     plt.text(a[i], b[i], varnames[i], fontsize=fontsize);
    #
    # plt.xlabel("axe %d" % (pa + 1), fontsize=fontsize);
    # plt.ylabel("axe %d" % (po + 1), fontsize=fontsize);
    plt.suptitle("ACP : Cercle des corrélations plan %d-%d" % (pa + 1, po + 1), fontsize=fontsize);
    return a[start::step], b[start::step]


# ************************
# load the data

mat_dict = loadmat('/Users/carl/Dropbox/Docs/Python/PyCharm/TRIED_RNRF_GIT/TRIED_TP4/clim_t2C_J1982D2010.mat')
clim_t2 = mat_dict['clim_t2']

# find out what type is in the dict
if isinstance(clim_t2, list):
    print('list')
elif isinstance(clim_t2, np.ndarray):
    print('ndarray')
else:
    print('something else')

print(np.shape(clim_t2))

# centre and reduce the transpose of the data by column (average by month, across all locations)
clim_t2_centered_reduced = centre_and_reduce(clim_t2[:, 2:].T)

# ************************
# Plot graph of centered and reduced temperature by months

# set up values for graphs
ville = ['Reykjavik', 'Oslo', 'Paris', 'New York', 'Tunis', 'Alger', 'Beyrouth', 'Atlan27N40W', 'Dakar']
values = range(9)
print(values)

jet = plt.get_cmap('jet')
cNorm = colors.Normalize(vmin=0, vmax=values[-1])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

# create new figure
fig = plt.figure(num=None, figsize=(20, 9), dpi=80, facecolor='w', edgecolor='k')
plt.title('Centered & reduced temperatures in 9 regions, from January 1982 to December 2010')
plt.ylabel('Centered & reduced temperatures')
plt.xticks(clim_t2[::12, 0], rotation=45)
plt.xlabel('Years')

for i in values:
    colorVal = scalarMap.to_rgba(values[i])
    plt.plot(clim_t2[:, 0] + (clim_t2[:, 1] / 12), clim_t2_centered_reduced.T[:, i], color=colorVal, label=ville[i])

plt.legend()
plt.show()

fig.savefig('part2_temp_by_month_cr.png')


# ************************
# Calculate linear regression on non-normalised and normalised data for each location
# Take min and max gradient value for both non-normalised and normalised
b1_list_clim_t2 = []
b1_list_clim_t2_centered_reduced = []

for i in np.arange(9):
    b0, b1, s, R2, sigb0, sigb1 = triedtools.linreg(clim_t2[:, 0] + (clim_t2[:, 1] / 12), clim_t2[:, i+2])
    b1_list_clim_t2.append(b1)
    b0, b1, s, R2, sigb0, sigb1 = triedtools.linreg(clim_t2[:, 0] + (clim_t2[:, 1] / 12), clim_t2_centered_reduced[i, :])
    b1_list_clim_t2_centered_reduced.append(b1)

print(max(b1_list_clim_t2))
print(min(b1_list_clim_t2))
print(max(b1_list_clim_t2_centered_reduced))
print(min(b1_list_clim_t2_centered_reduced))


# ************************
# ACP of centered and reduced temperature values

# get eigenvalues & eigenvectors, and project onto principal axes
eigval, eigvec, XU = triedacp.acp(clim_t2_centered_reduced)

# plot eigenvalues as bar chart, and cumulative inertia as line plot
inertia, cumulative_inertia = triedacp.phinertie(eigval)
fig.savefig('part2_bar_eigen_line_cumul_inertia_cr.png')

#  qual : Les qualités de réprésentation des individus par les axes
#  contrib: Les contributions des individus à la formation des axes
qual, contrib = triedacp.qltctr2(XU, eigval)

# verify that the sum of representations of each individual equals one (over each row)
qual_sum = np.sum(qual, 1)
# output is:
# [ 1.  1.  1.  1.  1.  1.  1.  1.  1.]

# verify that the sum of contributions to each axis equals one (over each col)
contrib_sum = np.sum(contrib, 0)
# output is:
# [ 1.  1.  1.  1.  1.]


# ************************
# Correlation circle using prince - all variables

# df = pd.DataFrame(data=clim_t2_centered_reduced)
# # df.columns = ville
# # variables are the 9 locations (use -1 to mean all)
# pca = prince.PCA(df, n_components=2)
# fig1, ax1 = pca.plot_correlation_circle(axes=(0, 1), show_labels=True)
# fig1.show()
# fig1.savefig('part2_correlation_circle_cr.png')

varnames = []
for i in range(np.size(clim_t2_centered_reduced, 1)):
    # varnames.append(str(i))
    varnames.append('')

tt_corcer(clim_t2_centered_reduced, XU, 1, 2, varnames)

# ************************
# Correlation circle using prince - one for each month

# months = ['Janvier', 'Fevrier', 'Mars', 'Avril', 'Mai', 'Juin',
#           'Juillet', 'Aout', 'Septembre', 'Octobre', 'Novembre', 'Decembre']
#
# for i in np.arange(len(months)):
#     df = pd.DataFrame(data=clim_t2_centered_reduced[i::12, :].T)
#     # df.columns = ville
#     # variables are the 9 locations (use -1 to mean all)
#     pca = prince.PCA(df, n_components=2)
#     fig1, ax1 = pca.plot_correlation_circle(axes=(0, 1), show_labels=True)
#     fig1.show()
#     fig1.suptitle(months[i])
#     fig1.savefig('part2_correlation_circle_cr_' + months[i] + '.png')

# ************************
# Correlation circles - one for each month

fig = plt.figure(facecolor='w')

values = np.arange(12).tolist()
jet = plt.get_cmap('jet')
cNorm = colors.Normalize(vmin=0, vmax=values[-1])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

months = ['Janvier', 'Fevrier', 'Mars', 'Avril', 'Mai', 'Juin',
          'Juillet', 'Aout', 'Septembre', 'Octobre', 'Novembre', 'Decembre']

corr_a = []
corr_b = []

for i in range(12):
    fig.add_subplot(4, 3, i+1)
    plt.title(months[i])
    colorVal = scalarMap.to_rgba(values[i])
    # eigval, eigvec, XU = triedacp.acp(clim_t2_centered_reduced[:, i::12])
    a, b = tt_corcer(clim_t2_centered_reduced, XU, 1, 2, varnames[::12], coul=colorVal, markersize=3, fontsize=8, start=i, step=12)
    corr_a.append(a)
    corr_b.append(b)

plt.legend()
plt.show()
fig.savefig('part2_corr_circles_12of.png')

# ************************
# Correlation circles - average pca vector for each month

corr_a_ave = np.array(corr_a).astype(float).mean(axis=1)
corr_b_ave = np.array(corr_b).astype(float).mean(axis=1)

fig = plt.figure(facecolor='w')

values = np.arange(12).tolist()
jet = plt.get_cmap('jet')
cNorm = colors.Normalize(vmin=0, vmax=values[-1])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

months = ['Janvier', 'Fevrier', 'Mars', 'Avril', 'Mai', 'Juin',
          'Juillet', 'Aout', 'Septembre', 'Octobre', 'Novembre', 'Decembre']

tt_cerclecor()

for i in range(12):
    colorVal = scalarMap.to_rgba(values[i])
    plt.plot(corr_a_ave[i], corr_b_ave[i], 'o', c=colorVal, markersize=5, label=months[i]);
    plt.text(corr_a_ave[i] + 0.02, corr_b_ave[i], months[i], fontsize=6);

plt.legend(loc=4)
plt.show()
fig.savefig('part2_corr_circle_average_vectors.png')

# ************************
# Calc contrib of 9 locs on axes 1 and 2, and plot to ensure none contribute much more than others

# get eigenvalues & eigenvectors, and project onto principal axes
eigval, eigvec, XU = triedacp.acp(clim_t2_centered_reduced)

#  qual : Les qualités de réprésentation des individus par les axes
#  contrib: Les contributions des individus à la formation des axes
qual, contrib = triedacp.qltctr2(XU, eigval)

# verify that the sum of representations of each individual equals one (over each row)
qual_sum = np.sum(qual, 1)
# output is:
# [ 1.  1.  1.  1.  1.  1.  1.  1.  1.]

# verify that the sum of contributions to each axis equals one (over each col)
contrib_sum = np.sum(contrib, 0)
# output is:
# [ 1.  1.  1.  1.  1.]

