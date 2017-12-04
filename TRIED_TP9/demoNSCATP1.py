#----------------------------------------------------------------------
import sys
import numpy as np
import matplotlib.pyplot as plt
from   matplotlib import cm
# If triedpy modules are not directly accessible by python,
# set the path where triedpy is located :
if 1 :
    #TRIEDPY = "../../../../";   # Mettre le chemin d'accès ...
    #TRIEDPY = "C:/Users/Charles/Documents/FAD/FAD_Charles";
    TRIEDPY = "/Users/carl/Dropbox/Docs/Python/PyCharm/TRIED_RNRF_GIT/TRIED_TP9";
    sys.path.append(TRIEDPY);  # ... aux modules triedpy
from   triedpy import triedtools as tls
from   triedpy import trieddeep  as tdp
# ENONCE = "../TPB04_R_diffusiometreNSCAT";   # Pour l'accès aux ressources de l'énonce
ENONCE = ".";   # Pour l'accès aux ressources de l'énonce
sys.path.append(ENONCE);
#----------------------------------------------------------------------
#import TPB04_methodes
#----------------------------------------------------------------------
# Some Conditionnements 
plt.ion();
np.random.seed(2); # for reproductiility
#
#%=====================================================================
# Chargement des donnees brutes,
Vit = np.loadtxt(ENONCE+"/Diffu_Vit.dat")
Inc = np.loadtxt(ENONCE+"/Diffu_Inc.dat")
Sig = np.loadtxt(ENONCE+"/Diffu_Sig.dat")
N   = len(Sig);
#
# Il convient de mettre les données dans le type array (1D)
Vit = Vit.reshape(N,1);
Inc = Inc.reshape(N,1);
Sig = Sig.reshape(N,1);
#
# moyennes et écarts types,
mean_vit = np.mean(Vit); std_vit = np.std(Vit);
mean_inc = np.mean(Inc); std_inc = np.std(Inc);
mean_sig = np.mean(Sig); std_sig = np.std(Sig);
#
# Donnees normalisees et matrice d'entree
VitN    = tls.centred(Vit, mean_vit, std_vit, coef=2/3);
IncN    = tls.centred(Inc, mean_inc, std_inc, coef=2/3);
SigN    = tls.centred(Sig, mean_sig, std_sig, coef=2/3);
#
XN      = np.concatenate((VitN, IncN), axis=1)
#
#----------------------------------------------------------------------
# Séparation des bases en Apprentissage et Validation
def TPB04_split_bases(X,Y,percent=2/3) :
    n = len(XN); #%% Nombre d'exemples

    #%% Nombre d'exemples seconde base
    n_firstset  = int(np.ceil(n * percent));
    n_secondset = n - n_firstset;

    print('     N .............. %d' %(n));
    print('     N_First set .... %d' %(n_firstset));
    print('     N_Second set ... %d' %(n_secondset));

    # Permutation des indices des donnees
    i_perm   = np.random.permutation(n);      #%% indices 1:n 'randomises'
    i_first  = i_perm[0:n_firstset];
    i_second = i_perm[n_firstset:n];

    # 1er ensemble
    Xa = X[i_first,];
    Ya = Y[i_first,];

    # 2eme ensemble
    Xb = X[i_second,];
    Yb = Y[i_second,];

    return Xa, Ya, Xb, Yb

#----------------------------------------------------------------------
# Séparation des bases en Apprentissage et Validation
XaN,YaN,XvN,YvN = TPB04_split_bases(XN, SigN); #save demobases XN SigN XaN YaN XvN YvN;
#load demobases_187
#
#======================================================================
# =============== PARAMETRAGE DU PMC ET APPRENTISSAGE ================= 
#----------------------------------------------------------------------
m = 5;           # Nombre de cellules sur la couche cachées
nbitemax = 500;  # Nombre maximum d'iterations
F1    = "tah";   # Fonction d'activation des neurones de la couche cachée
F2    = "lin";   # Fonction d'activation des neurones de sortie
FF    = [F1,F2]; # Fonctions de Transfert : mettre autant de Fi que de m+1
dfreq = 50;      # Frequence d'affichage  
WW = tdp.pmcinit(XaN,YaN,[m]);  # save demoweini0 W1 W2; % voir aussi save/load demobases
#load demoweini0_m5_5Cit_187
WWmv, ITMINVAL = tdp.pmctrain(XaN,YaN,WW,FF,nbitemax=nbitemax,
                    dfreq=dfreq,dcurves=1,Xv=XvN,Yv=YvN);    
# Erreur Min en Validation
YmvN   = tdp.pmcout(XvN,WWmv,FF);
Evalmv = tls.errors(YvN,YmvN,errtype=["errq"]); print(Evalmv);
#
#======================================================================
# Affichage des donnees et de la Regression du PMC pour une valeur (ou+) 
# de theta et une tolerance (voir help de plot_donnees et de plot_regress)
allT  = [34.5];      # Le(s) angle(s) d'incidence Theta [34 35]
Ntheta= len(allT);
Ttol  = 0.5;         # Tolerance sur l'angle d'incidence Theta
#
def TPB04_plot_donnees(BasX1,BasX2,allx2,x2tol,BasY,Palette=0,Tmark=0,szmark=6) :
    #
    nens=len(allx2);
    #
    if Palette == 0 :
       # Definition d'une palette par defaut pour les couleurs 
       # de chaque ensemble 
       LPalette = cm.jet(np.arange(280));
       Palette = [];
       for i in np.arange(nens) : 
           #ipal = 1+np.floor((i-1)*256/nens)
           ipal = int(1+np.floor((i)*280/nens)); #print(ipal)
           LPi  =  LPalette[ipal,:]; #print(LPi)
           Palette.append(LPi);
    #
    if Tmark == 0 :
        Tmark=[];
        for i in np.arange(nens) : 
           Tmark.append('.');
    #
    Nall=[];
    for i in np.arange(nens) : #1:nens
        x2=allx2[i];    # current selected Value (fixed within the loop)
        # indices de selection autour de x2
        Ix2 = np.where( (BasX2>(x2-x2tol))  &  (BasX2<(x2+x2tol)) )[0];
        # plot des valeurs selectionnees
        plt.plot(BasX1[Ix2],BasY[Ix2],Tmark[i],markersize=szmark,color=Palette[i]);                   
        Nall.append(len(Ix2));
    #    
    return Nall,Palette
#
# On plot d'abord les donnees, 
plt.figure()
Nall, Palette = TPB04_plot_donnees(Vit, Inc, allT, Ttol, Sig);
#
def TPB04_complete_plotdonVSI() : # Pour Completer le plot des donnees proprement dit
    plt.axis('tight')
    plt.ylabel('Sigma0 (dB)');
    xl1 = str(allT);    #!!!!!!! allT et non pas allV
    xl2 = str(Nall);
    plt.xlabel('vitesse\n Theta.[%s] degrés (Tol. %2.2f):  Nb.:(%s)'%(xl1,Ttol,xl2));
    plt.title('Sigma0');
    plt.grid('on');
    if 1 : #plt.colorbar();
        import matplotlib 
        bmap = matplotlib.colors.ListedColormap(Palette); 
        bmap = plt.cm.ScalarMappable(cmap=bmap)
        bmap.set_array([])
        hcb = plt.colorbar(bmap)
        cticks  = np.arange(0,1,1/Ntheta);   
        hcb.set_ticks(cticks+1/(2*Ntheta));  
        YLabel=[]
        for i in np.arange(Ntheta) :          
            YLabel.append('%d m/s'%allT[i]);    
        hcb.set_ticklabels(YLabel)
#
TPB04_complete_plotdonVSI();
#
# Plot de LA regression pour chaque theta retenu (sur la fenetre active) 
# (avec les poids au min de la validation)
def TPB04_demoplot_regress(couleur,WW,FF,VitP,theta,
                 mean_vit,std_vit,mean_inc,std_inc,mean_sig,std_sig) :
    szXP = len(VitP); # Taille des donnees
    #
    # Construire la matrice d'entree normalisee
    VitPN = tls.centred(VitP, mean_vit, std_vit, coef=2/3);
    IncP  = theta * np.ones((szXP,1));
    IncPN = tls.centred(IncP, mean_inc, std_inc, coef=2/3);
    XPN   = np.concatenate((VitPN, IncPN), axis=1)
    #
    # Passe avant (avec les poids adéquats)
    YPN = tdp.pmcout(XPN,WW,FF);
    #
    # Denormalisation de la sortie et plot
    YP  = tls.decentred(YPN, mean_sig, std_sig, crCoef=2/3);
    #
    plt.plot(VitP,YP,'-',color=couleur,linewidth=4);
    plt.plot(VitP,YP,'-k',linewidth=2);
    plt.plot(VitP,YP,'-w',linewidth=1);
    #
    return XPN, YPN
#    
VitP = np.arange(min(Vit),max(Vit),0.2);  # Definition d'une abscisse de vitesse
VitP = np.reshape(VitP,(len(VitP),1))
for i in np.arange(Ntheta) : 
   TPB04_demoplot_regress(Palette[i],WWmv,FF,VitP,allT[i],
                mean_vit,std_vit,mean_inc,std_inc,mean_sig,std_sig);
plt.axis('tight');
#








