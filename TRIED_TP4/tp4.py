import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.io import loadmat
import matplotlib.colors as colors
import matplotlib.cm as cmx
import pandas as pd
import prince
import triedacp


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

# ************************
# Plot graph of temperature by months

# set up values for graphs
ville = ['Reykjavik', 'Oslo', 'Paris', 'New York', 'Tunis', 'Alger', 'Beyrouth', 'Atlan27N40W', 'Dakar']
values = range(9)
print(values)

jet = plt.get_cmap('jet')
cNorm = colors.Normalize(vmin=0, vmax=values[-1])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

# create new figure
fig = plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
plt.title('Temperatures dans 9 regions, de janvier 1982 à décembre 2010')
plt.ylabel('Temperature')
plt.xticks(clim_t2[::12, 0], rotation=45)
plt.xlabel('Années')

for i in values:
    colorVal = scalarMap.to_rgba(values[i])
    plt.plot(clim_t2[:, 0] + (clim_t2[:, 1] / 12), clim_t2[:, i+2], color=colorVal, label=ville[i])

plt.legend()
plt.show()

fig.savefig('temp_by_month.png')

# ************************
# show graph of mean temperatures and std devs for each location (calculated over the entire time range)

mean_temps = np.mean(clim_t2[:, 2::], 0)
std_temps = np.std(clim_t2[:, 2::], 0)

# set up values for graphs
ville = ['Reykjavik', 'Oslo', 'Paris', 'New York', 'Tunis', 'Alger', 'Beyrouth', 'Atlan27N40W', 'Dakar']
values = range(9)

# create new figure
fig = plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
plt.title('Temperatures moyennes dans 9 regions, de janvier 1982 à décembre 2010')
plt.ylabel('Temperature')
plt.xticks(values, ville[:], rotation=45)
plt.xlabel('Années')

plt.scatter(values, mean_temps, color='b', label='Mean')
plt.scatter(values, mean_temps - std_temps, color='g', label='-1 Std dev', marker='v')
plt.scatter(values, mean_temps + std_temps, color='g', label='+1 Std dev', marker='^')

plt.legend()
plt.show()

fig.savefig('mean_temp_entire_range.png')

# ************************
# show graph of mean temperatures and std devs for each location (calculated over the entire time range)

clim_t2_centered = centre_and_reduce(clim_t2[:, 2:])

# ************************
# plot correlation circle with labels

df = pd.DataFrame(data=clim_t2_centered)
df.columns = ville
# variables are the 9 locations (use -1 to mean all)
pca = prince.PCA(df, n_components=-1)
fig1, ax1 = pca.plot_correlation_circle(axes=(0, 1), show_labels=True)
fig1.show()
fig1.savefig('correlation_circle.png')

fig1, ax1 = pca.plot_cumulative_inertia()
fig1.show()
fig1.savefig('inertia_cumulative.png')

fig1, ax1 = pca.plot_inertia()
fig1.show()
fig1.savefig('inertia.png')

# get eigenvalues & eigenvectors, and project onto principal axes
eigval, eigvec, XU = triedacp.acp(clim_t2_centered)

# ************************
# plot eigenvalues as bar chart, and cumulative inertia as line plot

inertia, cumulative_inertia = triedacp.phinertie(eigval)
fig.savefig('bar_eigen_line_cumul_inertia.png')

# ************************
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

# calc mean temps
mean_temps = np.mean(clim_t2[:, 2:], 1)

# create new figure
fig = plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
plt.title('Scatter-plot of instances in variable space')
plt.ylabel('PC2')
plt.xlabel('PC1')
plt.axhline(0, color='k')
plt.axvline(0, color='k')

jet = plt.get_cmap('jet')
cNorm = colors.Normalize(vmin=mean_temps.min(), vmax=mean_temps.max())
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
colorVals = scalarMap.to_rgba(mean_temps)

sc = plt.scatter(XU[:, 0], XU[:, 1], marker='None', c=mean_temps, cmap=jet)

plt.scatter(XU[1::12, 0], XU[1::12, 1], marker='o', c=colorVals[1::12, :], label='Jan')
plt.scatter(XU[2::12, 0], XU[2::12, 1], marker=',', c=colorVals[2::12, :], label='Feb')
plt.scatter(XU[3::12, 0], XU[3::12, 1], marker='v', c=colorVals[3::12, :], label='Mar')
plt.scatter(XU[4::12, 0], XU[4::12, 1], marker='8', c=colorVals[4::12, :], label='Apr')
plt.scatter(XU[5::12, 0], XU[5::12, 1], marker='+', c=colorVals[5::12, :], label='May')
plt.scatter(XU[6::12, 0], XU[6::12, 1], marker='D', c=colorVals[6::12, :], label='Jun')
plt.scatter(XU[7::12, 0], XU[7::12, 1], marker='*', c=colorVals[7::12, :], label='Jul')
plt.scatter(XU[8::12, 0], XU[8::12, 1], marker='_', c=colorVals[8::12, :], label='Aug')
plt.scatter(XU[9::12, 0], XU[9::12, 1], marker='^', c=colorVals[9::12, :], label='Sep')
plt.scatter(XU[10::12, 0], XU[10::12, 1], marker='x', c=colorVals[10::12, :], label='Oct')
plt.scatter(XU[11::12, 0], XU[11::12, 1], marker='|', c=colorVals[11::12, :], label='Nov')
plt.scatter(XU[12::12, 0], XU[12::12, 1], marker='p', c=colorVals[12::12, :], label='Dec')

plt.colorbar(sc)

months = ['Janvier', 'Fevrier', 'Mars', 'Avril', 'Mai', 'Juin',
          'Juillet', 'Aout', 'Septembre', 'Octobre', 'Novembre', 'Decembre']
# N = np.size(XU, 0)
# for i in range(N):
#     plt.text(XU[:, 0], XU[:, 1], months[i % 12])

plt.legend()
plt.show()

fig.savefig('pca-nuage-1-2.png')


