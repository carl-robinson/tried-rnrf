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


# datafiles = ['x.txt']
# datafiles = ['pg_pd.txt']
datafiles = ['hx_hy_pg_pd.txt']
# datafiles = ['x.txt', 'hx_hy.txt', 'pg_pd.txt', 'hx_hy_pg_pd.txt']
classnames = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
# partition = (50, 100, 150, 200, 250, 300, 350, 400, 450)
partition = [350, 400, 450]

for p in partition:
    for datafile in datafiles:
        for i in range(5):
            Xapp, Xapplabels, Xtest, Xtestlabels = tp.set_sdata(datafile, p)

            sMap = tp.app_chiffres(Xapp, classnames, datafile)

            Perfapp = ctk.classifperf(sMap, Xapp, Xapplabels)
            Perftest = ctk.classifperf(sMap, Xapp, Xapplabels, Xtest, Xtestlabels)
            print('Papp=%f Ptest=%f'%(Perfapp,Perftest))

            with open('perf' + '_' + str(p) + '.txt', 'a') as f:
                f.write(datafile + '_' + str(i) + '\n')
                f.write('Papp=' + str(Perfapp) + ', Ptest=' + str(Perftest) + '\n')
                f.write('\n')

            # ********************
            # Xapp
            # Tfreq, Ulab = ctk.reflabfreq(sMap, Xapp, Xapplabels)
            # CBlabmaj = ctk.cblabvmaj(Tfreq, Ulab)
            # CBilabmaj = ctk.label2ind(CBlabmaj, classnames)  # transformation des labels en int
            # fig = ctk.showcarte(sMap,figlarg=12, fighaut=12,shape='s',shapescale=600,colcell = CBilabmaj, text = CBlabmaj,
            #                     sztext = 16, cmap = cm.jet, showcellid = False)
            # fig.tight_layout()
            # fig.savefig(datafile + '_' + str(p) + '_app_carte.png')
            #
            # # change showcarte labels to CBlabels to show number of data examples used to classify each neuron
            # CBLABELS = ctk.cblabfreq(Tfreq, Ulab)
            # fig = ctk.showcarte(sMap, figlarg=12, fighaut=12, shape='s', shapescale=600, colcell=CBilabmaj, text=CBLABELS,
            #                     sztext=16, cmap=cm.jet, showcellid=False)
            # fig.tight_layout()
            # fig.savefig(datafile + '_' + str(p) + '_app_carte_labels.png')
            #
            # # showrefpat for train
            # MBMUS = ctk.mbmus(sMap, Data=Xapp)
            # HITS = ctk.findhits(sMap, bmus=MBMUS)
            # # ctk.showrefpat(sMap, Xapp, 16, 16, MBMUS, HITS)
            # fig = ctk.showrefpat(sMap, Xapp, 4, 8, MBMUS, HITS, sztext=5, axis='tight', ticks='off')
            # fig.tight_layout()
            # fig.savefig(datafile + '_' + str(i) + '_app_refpat.png')
            #
            # fig = ctk.showrefactiv(sMap, Xapp[0:10, :], sztext=5)
            # fig.tight_layout()
            # fig.savefig(datafile + '_' + str(i) + '_app_refactiv.png')

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
            fig.savefig(datafile + '_' + str(p) + '_' + str(i) + '_app_carte_labels.png')

            # showrefpat for test
            MBMUS = ctk.mbmus(sMap, Data=Xapp)
            HITS = ctk.findhits(sMap, bmus=MBMUS)
            # ctk.showrefpat(sMap, Xapp, 16, 16, MBMUS, HITS)
            # print(np.shape(Xtest))
            fig = ctk.showrefpat(sMap, Xapp, 8, 8, MBMUS, HITS, sztext=5, axis='tight', ticks='off')
            fig.tight_layout()
            fig.savefig(datafile + '_' + str(p) + '_' + str(i) + '_app_refpat.png')

            fig = ctk.showrefactiv(sMap, Xapp[0:10, :], sztext=5)
            fig.tight_layout()
            fig.savefig(datafile + '_' + str(p) + '_' + str(i) + '_app_refactiv.png')


            #
            # # ********************
            # # Xtest
            # # change showcarte labels to CBlabels to show number of data examples used to classify each neuron
            # Tfreq, Ulab = ctk.reflabfreq(sMap, Xtest, Xtestlabels)
            # CBlabmaj = ctk.cblabvmaj(Tfreq, Ulab)
            # CBilabmaj = ctk.label2ind(CBlabmaj, classnames)  # transformation des labels en int
            # # fig = ctk.showcarte(sMap,figlarg=12, fighaut=12,shape='s',shapescale=600,colcell = CBilabmaj, text = CBlabmaj,
            # #                     sztext = 16, cmap = cm.jet, showcellid = False)
            # # fig.tight_layout()
            # # fig.savefig(datafile + '_' + str(p) + '_test_carte.png')
            #
            # CBLABELS = ctk.cblabfreq(Tfreq, Ulab)
            # fig = ctk.showcarte(sMap, figlarg=8, fighaut=6, shape='s', shapescale=400, colcell=CBilabmaj, text=CBLABELS,
            #                     sztext=11, cmap=cm.jet, showcellid=False, dv=-0.025)
            # fig.tight_layout()
            # fig.savefig(datafile + '_' + str(p) + '_' + str(i) + '_test_carte_labels.png')
            #
            # # showrefpat for test
            # MBMUS = ctk.mbmus(sMap, Data=Xtest)
            # HITS = ctk.findhits(sMap, bmus=MBMUS)
            # # ctk.showrefpat(sMap, Xapp, 16, 16, MBMUS, HITS)
            # # print(np.shape(Xtest))
            # fig = ctk.showrefpat(sMap, Xtest, 8, 8, MBMUS, HITS, sztext=5, axis='tight', ticks='off')
            # fig.tight_layout()
            # fig.savefig(datafile + '_' + str(p) + '_' + str(i) + '_test_refpat.png')
            #
            # fig = ctk.showrefactiv(sMap, Xtest[0:10, :], sztext=5)
            # fig.tight_layout()
            # fig.savefig(datafile + '_' + str(p) + '_' + str(i) + '_test_refactiv.png')
