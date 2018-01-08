import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from triedpy import triedtools as tls
from triedpy import triedctk   as ctk
from triedpy import triedsompy as SOM
import TPC04_methodes as tp4meth

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

# load with columns as time series, rows as vars
nino = np.loadtxt('el_nino.mat').T
dateNino = nino[0]
# take vars 1 to 5
tempNino = nino[1:5]
ventNino = nino[5:13]
classnames = ['no','NI']

# take all time series from 108 onwards (i guess to remove the -999 vals)
tempNino108 = tempNino[:,108:].T
# get only sst1 var from this
sst1 = tempNino[0,108:]
# n = 4
# should use 'with' block..
perfOutput = open("perfOutput.txt", "a+")

# set up some options for later use
iterations = [[50,50],[50,100],[100,50],[200,200]]
temperatures = [[5, 1, 0.3], [20.00, 10.0, 0.10]]

# 100,5,1--- 50-1-0.3
# temporary = np.arange(0,328,1)

Bestperf =0
# perfTestGlobal  = np.array(())
perfTestGlobal = []

# run 12 times to find global optimum?
for test in range(12):
    perfOutput.write("Execution numero %s \n" %(test+1))
    perfapp = np.zeros(10)
    perftest = np.zeros(10)
    for n in range(1,11):
        xapp, applabels, xtest, testlabels, comp_names = tp4meth.serielabset(sst1 , 'temp', n, sst1 , classnames)
        sMap = app_SST(xapp,iterations[2],temperatures[0])
        perfapp[n-1],perftest[n-1],mCapp,mCtest,=tp4meth.drasticperf2(sMap,xapp,xtest,1,classnames,applabels,testlabels)
        perfOutput.write("Pour n= %s Perf-App : %.3f | Perf-Test : %.3f \n" %(n,perfapp[n-1],perftest[n-1]) )
        if(perftest[n-1]>Bestperf):
            nBest = n
            Bestperf=perftest[n-1]
            sMapBest = sMap
            mCappBest =mCapp
            mCtestBest=mCtest
            Besttest = test+1
    perfTestGlobal += [perftest]
    plt.subplot(3,4,test+1)
    plt.bar(np.arange(1,11),perftest)
    plt.xlabel("Taille de la fenêtre")
    plt.ylabel("Performance en test")
    plt.xticks(np.arange(1,11), np.arange(1,11))
print('Pour une fenetre n= %s Perf Apprentissage: %.2f Perf Test :  %.2f | Test numero %s ' %(nBest, perfapp[nBest-1],Bestperf,Besttest))

# sMap.view_U_matrix()
perfOutput.close()

perfTestGlobal = np.array(perfTestGlobal)
perftMean = np.mean(perfTestGlobal,axis=0)
plt.figure()
plt.bar(np.arange(1,11),perftMean)
plt.xlabel("Taille de la fenêtre")
plt.ylabel("Performance en test Moyenne ")
plt.xticks(np.arange(1, 11), np.arange(1, 11))
plt.title('La performance selon la taille de fenêtre - Maximum performance %.2f pour n= %s ' %(np.max(perftMean),np.argmax(perftMean)+1))
