# import numpy as np
from triedpy import triedtools as tls

import TPC01_methodes as tp
from triedpy import triedctk as ctk
from   matplotlib import cm
import numpy as np

# sMap, Xapp, Xapplabels, classnames = tp.app_lettre()

# def confus(sm,Data,Datalabels,classnames,Databmus,visu=False) :
# tp.confus(sMap)

# def classifperf(sm,Xapp,Xapplabels,Xtest=None,Xtestlabels=None) :

# perf = ctk.classifperf(sMap, Xapp, Xapplabels)
#
# print()
# print('performance finale = ', perf)

# PART2
# def app_chiffres(Xapp, classnames, infile) :


datafile = 'x.txt'
# datafiles = ['pg_pd.txt']
# datafile = 'hx_hy_pg_pd.txt'
# datafiles = ['x.txt', 'hx_hy.txt', 'pg_pd.txt', 'hx_hy_pg_pd.txt']

Xapp, Xapplabels, Xtest, Xtestlabels = tp.set_sdata(datafile, 450)

tp.display_pat(Xapp.T, 1, 10)
