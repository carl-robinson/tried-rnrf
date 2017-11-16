#--------------------------------------------------------
import sys
import numpy as np
import matplotlib.pyplot as plt
# If triedpy modules are not directly accessible by python,
# set the path where triedpy is located :
if 1 :
    TRIEDPY = "../../../../";   # Mettre le chemin d'accès ... 
    sys.path.append(TRIEDPY);   # ... aux modules triedpy
import triedtools as tls
import trieddeep  as tdp
import TPB01_methodes
#--------------------------------------------------------
# > Fonction schioler bruitée
N = 100;  # Taille des données
np.random.seed(0);    # (Re)positionnement du random (ou pas)
Xi, Yi = TPB01_methodes.schioler(N); # Here is the call to the function
#
# > Initialisation du PMC
# Architecture : nombres de neurones par couches cachées
m  = [5]; # Un couche cachée de 5 neurones
#
# Fonctions de Transfert : mettre autant de Fi que de m+1
F1 = "tah"; F2 = "lin"; 
FF = [F1,F2];
#
# Les poids
WW   = tdp.pmcinit(Xi, Yi, m);
#
# > Learning
nbitemax = 2000;
dfreq    = 200;  
dcurves  = 2;
#
WW = tdp.pmctrain(Xi,Yi,WW,FF, \
     nbitemax=nbitemax, dfreq=dfreq, dcurves=dcurves);
plt.title("Minimisation sur la fonction de coût",fontsize=16);
Y = tdp.pmcout(Xi,WW,FF); # Valeur de sortie du PMC sur l'ensemble d'App.
#
# Error et RMS en Apprentissage :
errq, rms, = tls.errors(Yi, Y,["errq","rms"])
print("\nApprentissage : (Erreur, RMS) = (%f, %f)" %(errq,rms));

# figure des donnees et de la régression
YC = TPB01_methodes.schiolerClean(Xi); # Valeur de la vraie fonction schioler (i.e. non bruitee) 
plt.figure();
plt.plot(Xi,Yi,'b*',markersize=8);    # Bruited datas
plt.plot(Xi,Y, 'r', linewidth=1.5);   # Modèle regressif
plt.plot(Xi,YC,'k', linewidth=1.5);   # Not bruited function
plt.legend(["Datas","PMC","True"],numpoints=1);

# Résultats sur un ensemble de Test (Generalisation)
Xi,Yi = TPB01_methodes.schioler(N);
Y     = tdp.pmcout(Xi,WW,FF);
errq, rms, = tls.errors(Yi, Y,["errq","rms"])
print("Generalisation: (Erreur, RMS) = (%f, %f)" %(errq,rms));

#----------
plt.show();



