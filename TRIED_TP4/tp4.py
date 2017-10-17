import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.io import loadmat
import matplotlib.colors as colors
import matplotlib.cm as cmx

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
fig = plt.figure(num=None, figsize=(20, 12), dpi=80, facecolor='w', edgecolor='k')
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