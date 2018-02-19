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
from prettytable import PrettyTable
import numpy.matlib as nm




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

mat_dict = np.loadtxt('/Users/carl/Dropbox/Docs/Python/PyCharm/TRIED_RNRF_GIT/TRIED_TP4/crabes.dat')
varnames = ['FL', 'RW', 'CL', 'CW', 'BD']

corrCoef = np.corrcoef(mat_dict, rowvar=False)
print(corrCoef)
triedtools.plotby2(mat_dict, mks=5, varnames=varnames)

mat_dict_cenred = centre_and_reduce(mat_dict)

# get eigenvalues & eigenvectors, and project onto principal axes
eigval, eigvec, XU = triedacp.acp(mat_dict_cenred)

# plot eigenvalues as bar chart, and cumulative inertia as line plot
inertia, cumulative_inertia, fig = triedacp.phinertie(eigval)
fig.savefig('inertia_cr.png')

# print out pretty table of X_centered
ptable = PrettyTable()
labels = ['axis1', 'axis2', 'axis3', 'axis4', 'axis5']
ptable.add_column('', labels)
ptable.add_column('Eigen vals', np.round(eigval.astype(float), 3))
ptable.add_column('Inertia', np.round(inertia.astype(float), 3))
ptable.add_column('Cumul. inertia', np.round(cumulative_inertia.astype(float), 3))
ptable.align = 'r'
print(ptable)

# # ************************
# # Plot locations on pc1 and pc2
#
# # create new figure
# fig = plt.figure()
# plt.title('Coordinates of examples on PC1 and PC2')
# plt.ylabel('PC 2')
# plt.xlabel('PC 1')
# plt.axhline(0, color='k')
# plt.axvline(0, color='k')
#
# classnames = ['Group 1', 'Group 2', 'Group 3', 'Group 4']
# values = range(4)
# jet = plt.get_cmap('jet')
# cNorm = colors.Normalize(vmin=0, vmax=values[-1])
# scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
#
# for i in values:
#     colorVal = scalarMap.to_rgba(i)
#     plt.scatter(XU[i*50:(i*50)+50, 0], XU[i*50:(i*50)+50, 1], color=colorVal, label=classnames[i], marker='v', s=20)
#
# plt.legend()
# plt.show()
#
# fig.savefig('pca_cloud_1_2.png')
#
#
# # ************************
# # Plot locations on pc2 and pc3
#
# # create new figure
# fig = plt.figure()
# plt.title('Coordinates of examples on PC2 and PC3')
# plt.ylabel('PC 3')
# plt.xlabel('PC 2')
# plt.axhline(0, color='k')
# plt.axvline(0, color='k')
#
# classnames = ['Group 1', 'Group 2', 'Group 3', 'Group 4']
# values = range(4)
# jet = plt.get_cmap('jet')
# cNorm = colors.Normalize(vmin=0, vmax=values[-1])
# scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
#
# for i in values:
#     colorVal = scalarMap.to_rgba(i)
#     plt.scatter(XU[i*50:(i*50)+50, 1], XU[i*50:(i*50)+50, 2], color=colorVal, label=classnames[i], marker='v', s=20)
#
# plt.legend()
# plt.show()
#
# fig.savefig('pca_cloud_2_3.png')
#
# # ************************
# # Plot locations on pc2 and pc3
#
# # create new figure
# fig = plt.figure()
# plt.title('Coordinates of examples on PC1 and PC3')
# plt.ylabel('PC 3')
# plt.xlabel('PC 1')
# plt.axhline(0, color='k')
# plt.axvline(0, color='k')
#
# classnames = ['Group 1', 'Group 2', 'Group 3', 'Group 4']
# values = range(4)
# jet = plt.get_cmap('jet')
# cNorm = colors.Normalize(vmin=0, vmax=values[-1])
# scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
#
# for i in values:
#     colorVal = scalarMap.to_rgba(i)
#     plt.scatter(XU[i*50:(i*50)+50, 0], XU[i*50:(i*50)+50, 2], color=colorVal, label=classnames[i], marker='v', s=20)
#
# plt.legend()
# plt.show()
#
# fig.savefig('pca_cloud_1_3.png')

# ******** pca subplots
# create new figure
fig = plt.figure()
inc=1
for i in range(5):
    for j in range(5):
        plt.subplot(5, 5, inc)
        plt.title(str(i+1) + '_' + str(j+1))
        # plt.ylabel('PC ' + str(i))
        # plt.xlabel('PC ' + str(j))
        plt.axhline(0, color='k')
        plt.axvline(0, color='k')

        plt.tick_params(
            # axis='x',  # changes apply to the x-axis
            axis='both',  # changes apply to both axis
            which='both',  # both major and minor ticks are affected
            left='off',  # ticks along the left edge are off
            right='off',  # ticks along the right edge are off
            bottom='off',  # ticks along the bottom edge are off
            top='off',  # ticks along the top edge are off
            labelleft='off',  # labels along the left edge are off
            labelright='off',  # labels along the right edge are off
            labeltop='off',  # labels along the top edge are off
            labelbottom='off')  # labels along the bottom edge are off

        classnames = ['Group 1', 'Group 2', 'Group 3', 'Group 4']
        values = range(4)
        jet = plt.get_cmap('jet')
        cNorm = colors.Normalize(vmin=0, vmax=values[-1])
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

        for val in values:
            colorVal = scalarMap.to_rgba(val)
            plt.scatter(XU[val * 50:(val * 50) + 50, i], XU[val * 50:(val * 50) + 50, j], color=colorVal, label=classnames[val],
                        marker='v', s=9)

        inc = inc + 1

# plt.tight_layout()
plt.legend()
plt.show()

fig.savefig('pca_cloud_2by2.png')



# ************************
# Correlation circle
#
# fig = plt.figure(facecolor='w')
# plt.title('Correlation circle')
# a, b = tt_corcer(mat_dict_cenred, XU, 1, 2, varnames, coul='b', markersize=5, fontsize=8)
# plt.show()
# fig.savefig('corr_circle_1-2.png')
#
# fig = plt.figure(facecolor='w')
# plt.title('Correlation circle')
# a, b = tt_corcer(mat_dict_cenred, XU, 2, 3, varnames, coul='b', markersize=5, fontsize=8)
# plt.show()
# fig.savefig('corr_circle_2-3.png')

df = pd.DataFrame(data=mat_dict_cenred)
df.columns = varnames
# variables are the 9 locations (use -1 to mean all)
pca = prince.PCA(df, n_components=-1)
fig1, ax1 = pca.plot_correlation_circle(axes=(0, 1), show_labels=True)
fig1.show()
fig1.savefig('corr_circle_prince_0_1.png')

df = pd.DataFrame(data=mat_dict_cenred)
df.columns = varnames
# variables are the 9 locations (use -1 to mean all)
pca = prince.PCA(df, n_components=-1)
fig1, ax1 = pca.plot_correlation_circle(axes=(1, 2), show_labels=True)
fig1.show()
fig1.savefig('corr_circle_prince_1_2.png')

# clim_t2 = mat_dict['clim_t2']
#
# # find out what type is in the dict
# if isinstance(clim_t2, list):
#     print('list')
# elif isinstance(clim_t2, np.ndarray):
#     print('ndarray')
# else:
#     print('something else')
#
# print(np.shape(clim_t2))
#
# # centre and reduce the transpose of the data by column (average by month, across all locations)
# clim_t2_centered_reduced = centre_and_reduce(clim_t2[:, 2:].T)
#
# # ************************
# # Plot graph of centered and reduced temperature by months
#
# # set up values for graphs
# ville = ['Reykjavik', 'Oslo', 'Paris', 'New York', 'Tunis', 'Alger', 'Beyrouth', 'Atlan27N40W', 'Dakar']
# values = range(9)
# print(values)
#
# jet = plt.get_cmap('jet')
# cNorm = colors.Normalize(vmin=0, vmax=values[-1])
# scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
#
# # create new figure
# fig = plt.figure(num=None, figsize=(20, 9), dpi=80, facecolor='w', edgecolor='k')
# plt.title('Centered & reduced temperatures in 9 regions, from January 1982 to December 2010')
# plt.ylabel('Centered & reduced temperatures')
# plt.xticks(clim_t2[::12, 0], rotation=45)
# plt.xlabel('Years')
#
# for i in values:
#     colorVal = scalarMap.to_rgba(values[i])
#     plt.plot(clim_t2[:, 0] + (clim_t2[:, 1] / 12), clim_t2_centered_reduced.T[:, i], color=colorVal, label=ville[i])
#
# plt.legend()
# plt.show()
#
# fig.savefig('part2_temp_by_month_cr.png')
#
#
# # ************************
# # Calculate linear regression on non-normalised and normalised data for each location
# # Take min and max gradient value for both non-normalised and normalised
# b1_list_clim_t2 = []
# b1_list_clim_t2_centered_reduced = []
#
# for i in np.arange(9):
#     b0, b1, s, R2, sigb0, sigb1 = triedtools.linreg(clim_t2[:, 0] + (clim_t2[:, 1] / 12), clim_t2[:, i+2])
#     b1_list_clim_t2.append(b1)
#     b0, b1, s, R2, sigb0, sigb1 = triedtools.linreg(clim_t2[:, 0] + (clim_t2[:, 1] / 12), clim_t2_centered_reduced[i, :])
#     b1_list_clim_t2_centered_reduced.append(b1)
#
# print(max(b1_list_clim_t2))
# print(min(b1_list_clim_t2))
# print(max(b1_list_clim_t2_centered_reduced))
# print(min(b1_list_clim_t2_centered_reduced))
#
#
# # ************************
# # ACP of centered and reduced temperature values
#
# # get eigenvalues & eigenvectors, and project onto principal axes
# eigval, eigvec, XU = triedacp.acp(clim_t2_centered_reduced)
#
# # plot eigenvalues as bar chart, and cumulative inertia as line plot
# inertia, cumulative_inertia = triedacp.phinertie(eigval)
# fig.savefig('part2_bar_eigen_line_cumul_inertia_cr.png')
#
# #  qual : Les qualités de réprésentation des individus par les axes
# #  contrib: Les contributions des individus à la formation des axes
# qual, contrib = triedacp.qltctr2(XU, eigval)
#
# # verify that the sum of representations of each individual equals one (over each row)
# qual_sum = np.sum(qual, 1)
# # output is:
# # [ 1.  1.  1.  1.  1.  1.  1.  1.  1.]
#
# # verify that the sum of contributions to each axis equals one (over each col)
# contrib_sum = np.sum(contrib, 0)
# # output is:
# # [ 1.  1.  1.  1.  1.]
#
#
# # ************************
# # Correlation circle using prince - all variables
#
# # df = pd.DataFrame(data=clim_t2_centered_reduced)
# # # df.columns = ville
# # # variables are the 9 locations (use -1 to mean all)
# # pca = prince.PCA(df, n_components=2)
# # fig1, ax1 = pca.plot_correlation_circle(axes=(0, 1), show_labels=True)
# # fig1.show()
# # fig1.savefig('part2_correlation_circle_cr.png')
#
# varnames = []
# for i in range(np.size(clim_t2_centered_reduced, 1)):
#     # varnames.append(str(i))
#     varnames.append('')
#
# tt_corcer(clim_t2_centered_reduced, XU, 1, 2, varnames)
#
# # ************************
# # Correlation circle using prince - one for each month
#
# # months = ['Janvier', 'Fevrier', 'Mars', 'Avril', 'Mai', 'Juin',
# #           'Juillet', 'Aout', 'Septembre', 'Octobre', 'Novembre', 'Decembre']
# #
# # for i in np.arange(len(months)):
# #     df = pd.DataFrame(data=clim_t2_centered_reduced[i::12, :].T)
# #     # df.columns = ville
# #     # variables are the 9 locations (use -1 to mean all)
# #     pca = prince.PCA(df, n_components=2)
# #     fig1, ax1 = pca.plot_correlation_circle(axes=(0, 1), show_labels=True)
# #     fig1.show()
# #     fig1.suptitle(months[i])
# #     fig1.savefig('part2_correlation_circle_cr_' + months[i] + '.png')
#
# # ************************
# # Correlation circles - one for each month
#
# fig = plt.figure(facecolor='w')
#
# values = np.arange(12).tolist()
# jet = plt.get_cmap('jet')
# cNorm = colors.Normalize(vmin=0, vmax=values[-1])
# scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
#
# months = ['Janvier', 'Fevrier', 'Mars', 'Avril', 'Mai', 'Juin',
#           'Juillet', 'Aout', 'Septembre', 'Octobre', 'Novembre', 'Decembre']
#
# corr_a = []
# corr_b = []
#
# for i in range(12):
#     fig.add_subplot(4, 3, i+1)
#     plt.title(months[i])
#     colorVal = scalarMap.to_rgba(values[i])
#     # eigval, eigvec, XU = triedacp.acp(clim_t2_centered_reduced[:, i::12])
#     a, b = tt_corcer(clim_t2_centered_reduced, XU, 1, 2, varnames[::12], coul=colorVal, markersize=3, fontsize=8, start=i, step=12)
#     corr_a.append(a)
#     corr_b.append(b)
#
# plt.legend()
# plt.show()
# fig.savefig('part2_corr_circles_12of.png')
#
# # ************************
# # Correlation circles - average pca vector for each month
#
# corr_a_ave = np.array(corr_a).astype(float).mean(axis=1)
# corr_b_ave = np.array(corr_b).astype(float).mean(axis=1)
#
# fig = plt.figure(facecolor='w')
#
# values = np.arange(12).tolist()
# jet = plt.get_cmap('jet')
# cNorm = colors.Normalize(vmin=0, vmax=values[-1])
# scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
#
# months = ['Janvier', 'Fevrier', 'Mars', 'Avril', 'Mai', 'Juin',
#           'Juillet', 'Aout', 'Septembre', 'Octobre', 'Novembre', 'Decembre']
#
# tt_cerclecor()
#
# for i in range(12):
#     colorVal = scalarMap.to_rgba(values[i])
#     plt.plot(corr_a_ave[i], corr_b_ave[i], 'o', c=colorVal, markersize=5, label=months[i]);
#     plt.text(corr_a_ave[i] + 0.02, corr_b_ave[i], months[i], fontsize=6);
#
# plt.legend(loc=4)
# plt.show()
# fig.savefig('part2_corr_circle_average_vectors.png')
#
# # ************************
# # Calc contrib of 9 locs on axes 1 and 2, and plot to ensure none contribute much more than others
#
# # get eigenvalues & eigenvectors, and project onto principal axes
# eigval, eigvec, XU = triedacp.acp(clim_t2_centered_reduced)
# # eigval, eigvec, XU = triedacp.acp(clim_t2[:, 2:].T)
#
# #  qual : Les qualités de réprésentation des individus par les axes
# #  contrib: Les contributions des individus à la formation des axes
# # for first two principal axes / eigenvalues only
# qual, contrib = triedacp.qltctr2(XU, eigval)
#
# # verify that the sum of representations of each individual equals one (over each row)
# qual_sum = np.sum(qual, 1)
# # output is:
# # [ 1.  1.  1.  1.  1.  1.  1.  1.  1.]
#
# # verify that the sum of contributions to each axis equals one (over each col)
# contrib_sum = np.sum(contrib, 0)
# # output is:
# # [ 1.  1.]
#
# # print(contrib[:, 0:2].astype(float))
#
# labels = ['Reykjavik', 'Oslo', 'Paris', 'New York', 'Tunis', 'Alger', 'Beyrouth', 'Atlan27N40W', 'Dakar']
#
# # print out pretty table of X_centered
# ptable = PrettyTable()
# ptable.add_column('Location', labels)
# ptable.add_column('PC1', np.round(contrib[:, 0].astype(float), 3))
# ptable.add_column('PC2', np.round(contrib[:, 1].astype(float), 3))
# ptable.align = 'r'
# print(ptable)
#
#
# # create new figure
# fig = plt.figure()
# plt.title('Contributions of each city to principal components 1 and 2')
# plt.ylabel('Contribution')
# plt.xlabel('Cities')
# values = range(9)
#
# plt.plot(values, np.round(contrib[:, 0].astype(float), 3), 'bo-', markersize=5, label='PC 1')
# plt.plot(values, np.round(contrib[:, 1].astype(float), 3), 'gs-', markersize=5, label='PC 2')
#
# plt.xticks(values, ville[:], rotation=25)
# plt.legend()
# plt.show()
#
# fig.savefig('part2_contributions_1_2.png')
#
# # ************************
# # Plot locations on pc1 and pc2
#
# # create new figure
# fig = plt.figure()
# plt.title('Cities plotted against PC 1 and PC2, with quality of representation')
# plt.ylabel('PC 2')
# plt.xlabel('PC 1')
# plt.axhline(0, color='k')
# plt.axvline(0, color='k')
#
# jet = plt.get_cmap('jet')
# cNorm = colors.Normalize(vmin=0, vmax=values[-1])
# scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
#
# qualfull = np.array(qual[:, 0].astype(float)) + np.array(qual[:, 1].astype(float))
# qualdiff = qualfull - min(qualfull)
# qualrange = qualdiff / max(qualdiff)
# qualmarker = (qualrange + 0.01) * 3
#
#
# s = [20*2**n for n in qualmarker]
#
# for i in range(9):
#     colorVal = scalarMap.to_rgba(values[i])
#     plt.scatter(XU[i, 0], XU[i, 1], color=colorVal, label=ville[i], marker='v', s=s[i])
#     plt.text(XU[i, 0] + 0.4, XU[i, 1] + 0.2, ville[i], fontsize=9)
#     plt.text(XU[i, 0] + 0.4, XU[i, 1] - 0.2, np.round(qualfull[i], 2), fontsize=9)
#
# plt.legend()
# plt.show()
#
# fig.savefig('part2_cities_plotted_with_quality.png')
#
# # ************************
# # Plot scatter of pc1 against cities
#
# # create new figure
# fig = plt.figure()
# plt.title('PC 1 value for all cities')
# plt.ylabel('PC 1')
# plt.xlabel('Cities')
#
# plt.scatter(values, XU[:, 0], color='b', label='PC1', marker='o')
# plt.xticks(values, ville[:], rotation=25)
#
# plt.legend()
# plt.show()
#
# fig.savefig('part2_cities_plotted_with_pc1.png')
#
# # ************************
# # Plot scatter of pc1 against cities
#
# # create new figure
# fig = plt.figure()
# plt.title('PC 2 value for all cities')
# plt.ylabel('PC 2')
# plt.xlabel('Cities')
#
# plt.scatter(values, XU[:, 1], color='b', label='PC 2', marker='o')
# plt.xticks(values, ville[:], rotation=25)
#
# plt.legend()
# plt.show()
#
# fig.savefig('part2_cities_plotted_with_pc2.png')
#
# # ************************
# # Plot scatter of average temperatures of cities
#
# # create new figure
# fig = plt.figure()
# plt.title('Average temperatures of cities over entire period')
# plt.ylabel('Average Temperature')
# plt.xlabel('Cities')
#
# plt.scatter(values, clim_t2[:, 2:].mean(axis=0), color='b', label='Average Temperature', marker='o')
# plt.xticks(values, ville[:], rotation=25)
#
# plt.legend()
# plt.show()
#
# fig.savefig('part2_cities_plotted_with_mean_temp.png')
#
#
# # ************************
# # Calc mean monthly temps for each city
#
# # mean_temps = np.reshape(clim_t2[:, 2:], (9, 12, -1))
#
# mean_temps = []
#
# for i in range(9):
#     l = []
#     for j in range(12):
#         l.append(np.mean(clim_t2[j::12, i+2]))
#     mean_temps.append(l)
#
#
# std_temps = []
#
# for i in range(9):
#     l = []
#     for j in range(12):
#         # l.append(np.std(clim_t2[j::12, i+2]))
#         l.append(np.std(clim_t2[j::12, 2:]))
#     std_temps.append(l)
#
#
# mean_temps_global = []
#
# for j in range(12):
#     mean_temps_global.append(np.mean(clim_t2[j::12, 2:]))
#
#
# norm_temps = []
#
# for i in range(9):
#     l = []
#     for j in range(12):
#         l.append(mean_temps[i][j] - mean_temps_global[j])
#     norm_temps.append(l)
#
# for i in range(9):
#     l = []
#     for j in range(12):
#         norm_temps[i][j] /= std_temps[i][j]
#
# # ************************
# # Plot mean_temps against months
#
# # set up values for graphs
# ville = ['Reykjavik', 'Oslo', 'Paris', 'New York', 'Tunis', 'Alger', 'Beyrouth', 'Atlan27N40W', 'Dakar']
# values_cities = range(9)
# values_months = range(12)
#
# jet = plt.get_cmap('jet')
# cNorm = colors.Normalize(vmin=0, vmax=values_cities[-1])
# scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
#
# # create new figure
# fig = plt.figure()
# plt.title(' Mean monthly temperatures in 9 regions')
# plt.ylabel('Mean temperatures')
# months = ['Janvier', 'Fevrier', 'Mars', 'Avril', 'Mai', 'Juin',
#           'Juillet', 'Aout', 'Septembre', 'Octobre', 'Novembre', 'Decembre']
# plt.xticks(values_months, months, rotation=25)
# plt.xlabel('Months')
#
# for i in values_cities:
#     colorVal = scalarMap.to_rgba(values_cities[i])
#     plt.plot(values_months, mean_temps[i][:], color=colorVal, label=ville[i])
#
# plt.legend()
# plt.show()
#
# fig.savefig('part2_monthly_average_temps_raw.png')
#
# # ************************
# # Plot norm_temps against months
#
# # set up values for graphs
# ville = ['Reykjavik', 'Oslo', 'Paris', 'New York', 'Tunis', 'Alger', 'Beyrouth', 'Atlan27N40W', 'Dakar']
# values_cities = range(9)
# values_months = range(12)
#
# jet = plt.get_cmap('jet')
# cNorm = colors.Normalize(vmin=0, vmax=values_cities[-1])
# scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
#
# # create new figure
# fig = plt.figure()
# plt.title(' Normed mean monthly temperatures in 9 regions')
# plt.ylabel('Normed mean temperatures')
# months = ['Janvier', 'Fevrier', 'Mars', 'Avril', 'Mai', 'Juin',
#           'Juillet', 'Aout', 'Septembre', 'Octobre', 'Novembre', 'Decembre']
# plt.xticks(values_months, months, rotation=25)
# plt.xlabel('Months')
#
# for i in values_cities:
#     colorVal = scalarMap.to_rgba(values_cities[i])
#     plt.plot(values_months, norm_temps[i][:], color=colorVal, label=ville[i])
#
# plt.legend()
# plt.show()
#
# fig.savefig('part2_monthly_average_temps_normed.png')
#
#
# # ************************
# # imad code
#
# # this is the average temperature for each city
# # tMoy = np.mean(villes,axis = 1)
# # tMoy = clim_t2[:, 2:].mean(axis=0)
# tMoy = np.sort(clim_t2[:, 2:].mean(axis=0))
# ville = ['Reykjavik', 'Oslo', 'New York', 'Paris', 'Alger', 'Beyrouth', 'Tunis', 'Atlan27N40W', 'Dakar']
#
# #then the PC1
# PC1 = XU[:,0].astype(float)
# diff = tMoy - PC1
# corrCoef = np.corrcoef(tMoy,PC1)[0][1]
# plt.title("Average monthly temperatures, PC1 values, and their difference")
# plt.plot(values,tMoy,'bo-', label="Average monthly temperatures")
# plt.plot(values,PC1,'ro-', label="PC1 values")
# plt.plot(values,diff,'g-',label="Difference")
# plt.xticks(values, ville, rotation=25)
# plt.legend()
# plt.show()
