# import numpy as np
from triedpy import triedtools as tls

import TPC01_methodes as tp
from triedpy import triedctk as ctk
from   matplotlib import cm
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

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


# ************************
# load the data
xiris_norm = np.loadtxt('xiris_norm.txt')
tiris = np.loadtxt('tiris.txt')

# convert label data to ints 1,2,3
tiris_labels = []
for row in tiris:
    if row[0] == 1:
        tiris_labels.append('Se')
    elif row[1] == 1:
        tiris_labels.append('Ve')
    else:
        tiris_labels.append('Vi')

tiris_labels = np.array(tiris_labels)

varnames = ['sepal height', 'sepal width', 'petal height', 'petal width']
leg = ['Se', 'Ve', 'Vi']

# train_sizes = [0.2, 0.4, 0.6, 0.8]
train_sizes = [25, 50, 75, 100, 125]
accuracy_list = []

# som vars
datafile = 'xiris_norm'
classnames = ('Se', 'Ve', 'Vi')
# classnames = ('1', '2', '3')
# partition = (50, 100, 150, 200, 250, 300, 350, 400, 450)

app_size = 125

# split data
# Xapp, Xapplabels, Xtest, Xtestlabels = train_test_split(xiris_norm, tiris_labels, test_size=1 - t, random_state=42)
# Xapp, asdf, Xtest, Xtestlabels = train_test_split(xiris_norm, tiris_labels, test_size=1 - t, random_state=42)

Xapp = xiris_norm[0:app_size, :]
Xapplabels = tiris_labels[0:app_size]

Xtest = xiris_norm[app_size:, :]
Xtestlabels = tiris_labels[app_size:]

for i in range(1):
    sMap = tp.app_chiffres(Xapp, classnames, datafile)

    Perfapp = ctk.classifperf(sMap, Xapp, Xapplabels)
    Perftest = ctk.classifperf(sMap, Xapp, Xapplabels, Xtest, Xtestlabels)
    print('Papp=%f Ptest=%f' % (Perfapp, Perftest))


    # ********************
    # Xapp
    Tfreq, Ulab = ctk.reflabfreq(sMap, Xapp, Xapplabels)
    CBlabmaj = ctk.cblabvmaj(Tfreq, Ulab)
    CBilabmaj = ctk.label2ind(CBlabmaj, classnames)  # transformation des labels en int
    # fig = ctk.showcarte(sMap,figlarg=12, fighaut=12,shape='s',shapescale=600,colcell = CBilabmaj, text = CBlabmaj,
    #                     sztext = 16, cmap = cm.jet, showcellid = False)
    # fig.tight_layout()
    # fig.savefig(datafile + '_' + str(p) + '_app_carte.png')

    CBLABELS = ctk.cblabfreq(Tfreq, Ulab)
    fig = ctk.showcarte(sMap, figlarg=8, fighaut=6, shape='s', shapescale=400, colcell=CBilabmaj,
                        text=CBLABELS,
                        sztext=11, cmap=cm.jet, showcellid=False, dv=-0.025)
    fig.tight_layout()
    fig.savefig(datafile + '_' + str(i) + '_app_carte_labels.png')

    # showrefpat for app
    MBMUS = ctk.mbmus(sMap, Data=Xapp)
    HITS = ctk.findhits(sMap, bmus=MBMUS)
    # ctk.showrefpat(sMap, Xapp, 16, 16, MBMUS, HITS)
    # print(np.shape(Xtest))
    fig = ctk.showrefpat(sMap, Xapp, 2, 2, MBMUS, HITS, sztext=5, axis='tight', ticks='off')
    fig.tight_layout()
    fig.savefig(datafile + '_' + str(i)  + '_app_refpat.png')

    fig = ctk.showrefactiv(sMap, Xapp[0:10, :], sztext=5)
    fig.tight_layout()
    fig.savefig(datafile + '_' + str(i)  + '_app_refactiv.png')

    MC_train, P_train = tp.confus(sMap, Xapp, Xapplabels, classnames, MBMUS, visu=True)




    # # ********************
    # # Xtest
    # change showcarte labels to CBlabels to show number of data examples used to classify each neuron
    Tfreq, Ulab = ctk.reflabfreq(sMap, Xtest, Xtestlabels)
    CBlabmaj = ctk.cblabvmaj(Tfreq, Ulab)
    CBilabmaj = ctk.label2ind(CBlabmaj, classnames)  # transformation des labels en int
    # fig = ctk.showcarte(sMap,figlarg=12, fighaut=12,shape='s',shapescale=600,colcell = CBilabmaj, text = CBlabmaj,
    #                     sztext = 16, cmap = cm.jet, showcellid = False)
    # fig.tight_layout()
    # fig.savefig(datafile + '_' + str(p) + '_test_carte.png')

    CBLABELS = ctk.cblabfreq(Tfreq, Ulab)
    fig = ctk.showcarte(sMap, figlarg=8, fighaut=6, shape='s', shapescale=400, colcell=CBilabmaj, text=CBLABELS,
                        sztext=11, cmap=cm.jet, showcellid=False, dv=-0.025)
    fig.tight_layout()
    fig.savefig(datafile + '_' + str(i) + '_test_carte_labels.png')

    # showrefpat for test
    MBMUS = ctk.mbmus(sMap, Data=Xtest)
    HITS = ctk.findhits(sMap, bmus=MBMUS)
    # ctk.showrefpat(sMap, Xapp, 16, 16, MBMUS, HITS)
    # print(np.shape(Xtest))
    fig = ctk.showrefpat(sMap, Xtest, 2, 2, MBMUS, HITS, sztext=5, axis='tight', ticks='off')
    fig.tight_layout()
    fig.savefig(datafile + '_' + str(i) + '_test_refpat.png')

    fig = ctk.showrefactiv(sMap, Xtest[0:10, :], sztext=5)
    fig.tight_layout()
    fig.savefig(datafile + '_' + str(i) + '_test_refactiv.png')

    MC_test, P_test = tp.confus(sMap, Xtest, Xtestlabels, classnames, MBMUS, visu=True)

    with open('perf.txt', 'a') as f:
        f.write('*******************\n')
        f.write('ITERATION_' + str(i) + '\n')
        f.write('Papp=' + str(Perfapp) + ', Ptest=' + str(Perftest) + '\n')
        f.write('\n')
        f.write('MC train:\n')
        f.write('P_train = ' + str(P_train) + '\n')
        f.write('\n')
        f.write(str(MC_train) + '\n')
        f.write('MC test:\n')
        f.write('P_test = ' + str(P_test) + '\n')
        f.write('\n')
        f.write(str(MC_test) + '\n')
