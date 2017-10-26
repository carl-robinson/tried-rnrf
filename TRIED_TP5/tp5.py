import numpy as np
from scipy import stats
from scipy.io import loadmat

import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA

import triedtools


# ************************
# load the data

mat_dict = loadmat('/Users/carl/Dropbox/Docs/Python/PyCharm/TRIED_RNRF_GIT/TRIED_TP5/clim_co2_J1982D2010.mat')
clim_co2 = mat_dict['clim_co2']

# find out what type is in the dict
if isinstance(clim_co2, list):
    print('list')
elif isinstance(clim_co2, np.ndarray):
    print('ndarray')
else:
    print('something else')

print(np.shape(clim_co2))

# ************************
# Calc linreg

# correct years to start from 0, so you don't regress into the past and get wrong b0
X = (clim_co2[:, 0] + (clim_co2[:, 1] / 12)) - min(clim_co2[:, 0])
# return b0,b1,s,R2,sigb0,sigb1
linreg_result = triedtools.linreg(X, clim_co2[:, 2])

# ************************
# Calc corrected co2

# CO2cor = CO2 – b1*tclim, où tclim=[1:taille des données]
steps_in_years = np.arange(np.size(clim_co2, 0)) / 12
co2_corrected = clim_co2[:, 2] - (linreg_result[1] * steps_in_years)

linreg_result_cor = triedtools.linreg(X, co2_corrected)


# ************************
# Plot graph of raw co2 values and linreg

# create new figure
fig = plt.figure()

# set up values for graphs
values = range(9)

jet = plt.get_cmap('jet')
cNorm = colors.Normalize(vmin=0, vmax=values[-1])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

plt.title('CO2 measurements, from January 1982 to December 2010, with linear regression')
plt.ylabel('CO2 measurements (ppm)')
plt.xticks(X[::12], clim_co2[::12, 0].astype(int), rotation=45)
plt.xlabel('Years')

# X = clim_co2[:, 0] + (clim_co2[:, 1] / 12)
# raw co2 vals
plt.plot(X, clim_co2[:, 2], 'bo-', label='CO2 - Raw', markersize=3, linewidth=1)
# linreg line
plt.plot(X, (linreg_result[1] * X) + linreg_result[0], 'g-', label='CO2 - Raw - Linear Regression', linewidth=2)
# corrected vals
plt.plot(X, co2_corrected, 'ro-', label='CO2 - Corrected', markersize=3, linewidth=1)
# linreg line
plt.plot(X, (linreg_result_cor[1] * X) + linreg_result_cor[0], 'y-', label='CO2 - Corrected - Linear Regression', linewidth=2)

plt.legend()
plt.show()

fig.savefig('co2_by_month.png')


# ************************
# Load the temperature data

mat_dict = loadmat('/Users/carl/Dropbox/Docs/Python/PyCharm/TRIED_RNRF_GIT/TRIED_TP5/clim_t2C_J1982D2010.mat')
clim_t2 = mat_dict['clim_t2']
print(np.shape(clim_t2))

# ************************
# Work out the offset between seasonal effect of temp and co2
# Calc corrcoef for 60 month-step offsets (find the max)

# make list of average values across 9 villes for each month (348x1) - t2moy
clim_t2_moy = np.mean(clim_t2[:, 2:], 1)

offset_correlations = []
size = np.size(clim_t2_moy)

for m in range(60):
    corr = np.corrcoef(clim_t2_moy[0:size-m], co2_corrected[m:])
    offset_correlations.append(corr[0, 1])

# plot offset_correlations against m, to find maximum anti-correlation
# create new figure
fig = plt.figure()

plt.title('Correlations of t2moy and corrected co2 offset by M [0, 59]')
plt.ylabel('Correlation')
plt.xticks(range(60), rotation=90, fontsize=7)
plt.xlabel('M')

# X = clim_co2[:, 0] + (clim_co2[:, 1] / 12)
# raw co2 vals
plt.plot(range(60), offset_correlations, 'bo-', label='Correlation', markersize=3, linewidth=1)
plt.legend()
plt.show()

fig.savefig('co2_by_month.png')

print(min(offset_correlations))
print(max(offset_correlations))

# ************************
# Calc overall monthly averages for temp and co2, and plot them together

clim_t2_moy_ave = []
for i in range(12):
    clim_t2_moy_ave.append(np.mean(clim_t2_moy[i::12], 0))
clim_t2_moy_ave = np.array(clim_t2_moy_ave)

co2_corr_ave = []
for i in range(12):
    co2_corr_ave.append(np.mean(co2_corrected[i::12], 0))
co2_corr_ave = np.array(co2_corr_ave)


host = host_subplot(111, axes_class=AA.Axes)
plt.subplots_adjust(right=0.75)

par2 = host.twinx()

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']

host.set_xlabel("Months")
host.set_ylabel("Mean Temperature")
par2.set_ylabel("Mean CO2")

p2, = host.plot(range(12), clim_t2_moy_ave, 'bo-', label='Mean Temperature', markersize=3, linewidth=1)
p3, = par2.plot(range(12), co2_corr_ave, 'go-', label='Mean CO2', markersize=3, linewidth=1)

host.legend()

host.axis["left"].label.set_color(p2.get_color())
par2.axis["right"].label.set_color(p3.get_color())

plt.title('Monthly averages for t2moy and corrected co2')
plt.xticks(range(12), months)

plt.draw()
plt.show()

# ************************
# Scatter plots - mean temp vs co2 raw

fig = plt.figure()
plt.title('Mean Temperature against raw CO2')
plt.ylabel('Raw CO2')
plt.xlabel('Mean Temperature')
plt.scatter(clim_t2_moy, clim_co2[:, 2], color='b', label='CO2 - Raw', s=10)
linreg_result = triedtools.linreg(clim_t2_moy, clim_co2[:, 2])
plt.plot(clim_t2_moy, (linreg_result[1] * clim_t2_moy) + linreg_result[0], 'g-', label='Linear Regression', linewidth=1)
plt.legend()
plt.show()

print('Raw CO2')
print(linreg_result[0])
print(linreg_result[1])

corr = np.corrcoef(clim_t2_moy, clim_co2[:, 2])
print(corr[0, 1])


fig = plt.figure()
plt.title('Mean Temperature against corrected CO2')
plt.ylabel('Corrected CO2')
plt.xlabel('Mean Temperature')
plt.scatter(clim_t2_moy, co2_corrected, color='b', label='CO2 - Corrected', s=10)
# X = (clim_t2_moy - min(clim_t2_moy))
# linreg_result = triedtools.linreg(X, co2_corrected)
linreg_result = triedtools.linreg(clim_t2_moy, co2_corrected)
# plt.plot(X, (linreg_result[1] * X) + linreg_result[0], 'g-', label='Linear Regression', linewidth=1)
plt.plot(clim_t2_moy, (linreg_result[1] * clim_t2_moy) + linreg_result[0], 'g-', label='Linear Regression', linewidth=1)
plt.legend()
plt.show()
print('Corrected CO2')
print(linreg_result[0])
print(linreg_result[1])

corr = np.corrcoef(clim_t2_moy, co2_corrected)
print(corr[0, 1])


fig = plt.figure()
plt.title('Mean Temperature against corrected & offset CO2')
plt.ylabel('Corrected & Offset CO2')
plt.xlabel('Mean Temperature')
co2_corrected_offset = np.roll(co2_corrected, -3)
# co2_corrected_offset[np.size(co2_corrected_offset) - 3:np.size(co2_corrected_offset)] = None
plt.scatter(clim_t2_moy, co2_corrected_offset, color='b', label='CO2 - Corrected & Offset', s=10)

linreg_result = triedtools.linreg(clim_t2_moy, co2_corrected_offset)
plt.plot(clim_t2_moy, (linreg_result[1] * clim_t2_moy) + linreg_result[0], 'g-', label='Linear Regression', linewidth=1)

plt.legend()
plt.show()

print('Corrected & Offset CO2')
print(linreg_result[0])
print(linreg_result[1])

corr = np.corrcoef(clim_t2_moy, co2_corrected_offset)
print(corr[0, 1])


# ************************
# Annual averages for temp and co2

annual_temps = []
for i in range(29):
    annual_temps.append(np.mean(clim_t2_moy[i * 12:(i * 12) + 12]))
print(annual_temps)

annual_co2 = []
clim_co2_col2 = clim_co2[:, 2]
for i in range(29):
    annual_co2.append(np.mean(clim_co2_col2[i * 12:(i * 12) + 12]))
print(annual_co2)

corr = np.corrcoef(annual_temps, annual_co2)
print(corr[0, 1])

# ************************
# Plot annual averages for temp and co2 on same graph

host = host_subplot(111, axes_class=AA.Axes)
# plt.subplots_adjust(right=0.75)

par2 = host.twinx()

host.set_xlabel("Years")
host.set_ylabel("Mean Annual Temperature")
par2.set_ylabel("Mean Annual CO2")

p2, = host.plot(range(1982, 2011), annual_temps, 'bo-', label='Mean Annual Temperature', markersize=3, linewidth=1)
p3, = par2.plot(range(1982, 2011), annual_co2, 'go-', label='Mean Annual CO2', markersize=3, linewidth=1)

host.legend()

host.axis["left"].label.set_color(p2.get_color())
par2.axis["right"].label.set_color(p3.get_color())

plt.title('Mean Annual Temperature and CO2')
plt.draw()
plt.show()

