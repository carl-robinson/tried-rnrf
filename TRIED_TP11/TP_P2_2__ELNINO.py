import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from triedpy import triedtools as tls
from triedpy import triedctk   as ctk
from triedpy import triedsompy as SOM
import TPC04_methodes as tp4meth
from sklearn.metrics import confusion_matrix
import itertools
import sys


plt.close('all')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def app_SST(Xapp,iterations,temperatures):
    initmethod = 'random';  # 'random', 'pca'#
    nlmap = 7;
    ncmap = 7;

    tracking = 'on';  # niveau de suivi de l'apprentissage

    # paramètres 1ere étape :-> Variation rapide de la temperature
    epochs1 = iterations[0];
    radius_ini1 = temperatures[0];
    radius_fin1 = temperatures[1];
    etape1 = [epochs1, radius_ini1, radius_fin1];

    # paramètres 2ème étape :-> Variation fine de la temperature
    epochs2 = iterations[1];
    radius_ini2 = temperatures[1];
    radius_fin2 = temperatures[2];
    etape2 = [epochs2, radius_ini2, radius_fin2];
    classnames = ['El Nino','NON El Nino']
    varnames = ['SST1','SST2','SST3','SST4']
    sMap = SOM.SOM('sMap', Xapp, mapsize=[nlmap, ncmap], norm_method='data', \
                   initmethod=initmethod,varname=varnames)
    sMap.train(etape1=etape1, etape2=etape2, verbose=tracking);
    return sMap


nino = np.loadtxt('el_nino.mat').T
dateNino = nino[0]
tempNino = nino[1:5]
ventNino = nino[5:13]
classnames = ['no','NI']

tempNino108 = tempNino[:,108:].T
sst1 = tempNino[0,108:]
# n = 4
perfOutput = open("perfOutput.txt", "a+")

iterations = [[50,50],[50,100],[100,50],[200,200]]
temperatures = [[5, 1, 0.3], [20.00, 10.0, 0.10]]
#Taille de la fenêtre
Tfen = 2

#Consutruction de la serie chrono
Xapp, Xapplabels, Xtest, Xtestlabels, comp_names= tp4meth.serielabset(sst1,"SST1",Tfen,classname = classnames,Varlab = sst1)

#apprentissage par CT
sMap =  app_SST(Xapp,iterations[2],temperatures[0])

#calcul de la performance
Perfapp,Perftest,MCapp,MCtest = tp4meth.drasticperf2(sMap,Xapp,Xtest,1,classnames,Xapplabels,Xtestlabels)
print("Performance en apprentissage",Perfapp)
print("Performance en Test",Perftest)

print("")
print("############################## Matrice de Confusion  - données d'apprentissage #####################################")
print("")
print(MCapp)
# Plot non-normalized confusion matrix
plt.figure()
class_names = ['Not Detected', 'Not El Nino', 'El Nino']
plot_confusion_matrix(MCapp, classes=class_names,
                      title='Confusion matrix - Training Set', cmap=plt.cm.Greens)


print("")
print("############################## Matrice de Confusion  - données de test ######################################")
print("")
print(MCtest)
plt.figure()
plot_confusion_matrix(MCtest, classes=class_names,
                      title='Confusion matrix - Test Set', cmap=plt.cm.Reds)

plt.show()

# todo !!!!!!!!!!!!!!
# sys.exit()

map = cm.nipy_spectral


#Best matching units
bmusApp = ctk.findbmus(sMap,Data = Xapp)
bmusTest = ctk.findbmus(sMap,Data = Xtest)


#Récupérer les lables par vote majoritaire
Tfreq,Ulab = ctk.reflabfreq(sMap,Xapp,Xapplabels);
CBlabmaj   = ctk.cblabvmaj(Tfreq,Ulab);




#apprentissage
Tfreq,Ulab = ctk.reflabfreq(sMap,Xapp,Xapplabels);
CBlabmaj = ctk.cblabvmaj(Tfreq,Ulab);
CBilabmaj = ctk.label2ind(CBlabmaj, classnames); # transformation des labels en int
# ctk.showcarte(sMap,figlarg=12, fighaut=12,shape='s',shapescale=400,colcell=CBilabmaj+2,text=CBlabmaj,sztext=9
# ,cmap=map,showcellid=False);
#plt.savefig("figures/carte_app.jpeg")




labFreq =ctk.cblabfreq(Tfreq,Ulab)
labFreq =[str.replace(l,' ','\n') for l in labFreq]
ctk.showcarte(sMap,figlarg=12, fighaut=12,shape='s',shapescale=300,colcell=CBilabmaj+2,\
coltext='k',text=labFreq,sztext=9,cmap=map,showcellid=False,dv = -0.017,dh = -0.04);
plt.title("Activation frequency of neurons - Training Set")



#plt.savefig("figures/app_freq_affectation.jpeg")

CBlabmajApp = CBlabmaj
CBilabmajApp = CBilabmaj

#test
Tfreq,Ulab = ctk.reflabfreq(sMap,Xtest,Xtestlabels);
CBlabmaj = ctk.cblabvmaj(Tfreq,Ulab);
CBilabmaj = ctk.label2ind(CBlabmaj, classnames); # transformation des labels en int
#ctk.showcarte(sMap,figlarg=12, fighaut=12,shape='s',shapescale=600,colcell=CBilabmaj,text=CBlabmaj,sztext=16,cmap=cm.jet,showcellid=False);



labFreq =ctk.cblabfreq(Tfreq,Ulab)
labFreq =[str.replace(l,' ','\n') for l in labFreq]
ctk.showcarte(sMap,figlarg=12, fighaut=12,shape='s',shapescale=300,colcell=CBilabmajApp+5,\
coltext='k',text=labFreq,sztext=9,cmap=map,showcellid=False,dv = -0.017,dh = -0.04);
plt.title("Activation frequency of neurons - Test Set")

#plt.savefig("figures/test_freq_affectation.jpeg");



bmusApp = ctk.findbmus(sMap,Data = Xapp)

ctk.showmap(sMap, sztext=11, coltext='k',colbar=True, cmap=cm.jet, interp=None,caxmin=None,
          caxmax=None,axis=None, comp=[1], nodes=None, Labels=None, dh=0, dv=0)

# uncomment this
ctk.showprofils(sMap,Clevel=CBilabmajApp,visu=3,Data=Xapp, Gscale=0.4)
plt.savefig('profils echelle independante + xapp.png')

ctk.showprofils(sMap,scale=2,Clevel=CBilabmajApp,visu=3,Data=Xapp, Gscale=0.6)
plt.savefig('profils echelle commune + xapp.png')

# ctk.showprofils(sMap,Clevel=CBilabmajApp)
# plt.savefig('profils echelle independante.png')
#
# ctk.showprofils(sMap,scale=2,Clevel=CBilabmajApp)
# plt.savefig('profils echelle commune .png')

plt.show()
