import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import triedtools as tls
import trieddeep as tdp
import TPB04_methodes

# Init
plt.ion()
np.random.seed(2) # for reproductiility

#----------------------------------------------------------------------
# Séparation des bases en Apprentissage et Validation
def TPB04_split_bases(X,Y,percent=2/3) :
    n = len(XN) #%% Nombre d'exemples

    #%% Nombre d'exemples seconde base
    n_firstset  = int(np.ceil(n * percent))
    n_secondset = n - n_firstset

    print('     N .............. %d' %(n))
    print('     N_First set .... %d' %(n_firstset))
    print('     N_Second set ... %d' %(n_secondset))

    # Permutation des indices des donnees
    i_perm   = np.random.permutation(n)      #%% indices 1:n 'randomises'
    i_first  = i_perm[0:n_firstset]
    i_second = i_perm[n_firstset:n]

    # 1er ensemble
    Xa = X[i_first,]
    Ya = Y[i_first,]

    # 2eme ensemble
    Xb = X[i_second,]
    Yb = Y[i_second,]

    return Xa, Ya, Xb, Yb
#----------------------------------------------------------------------
# DATA PREP

# Chargement des donnees brutes,
Vit = np.loadtxt("Diffu_Vit.dat")
Inc = np.loadtxt("Diffu_Inc.dat")
Dir = np.loadtxt("Diffu_Dir.dat")
Sig = np.loadtxt("Diffu_Sig.dat")

# Il convient de mettre les données dans le type array (1D)
N = len(Sig)
Vit = Vit.reshape(N, 1)
Inc = Inc.reshape(N, 1)
Dir = Dir.reshape(N, 1)
Sig = Sig.reshape(N, 1)

# moyennes et écarts types,
mean_inc = np.mean(Inc)
mean_vit = np.mean(Vit)
mean_sig = np.mean(Sig)
mean_dir = np.mean(Dir)

std_inc = np.std(Inc)
std_vit = np.std(Vit)
std_sig = np.std(Sig)
std_dir = np.std(Dir)

# Donnees normalisees et matrice d'entree
VitN = tls.centred(Vit, mean_vit, std_vit, coef=2/3)
IncN = tls.centred(Inc, mean_inc, std_inc, coef=2/3)
SigN = tls.centred(Sig, mean_sig, std_sig, coef=2/3)

# take the sines and the cosines of the direction of the wind
Dir_sin = np.sin(np.radians(Dir))
Dir_cos = np.cos(np.radians(Dir))

# concat training data for input into PMC
XN = np.concatenate((Dir_sin, Dir_cos, VitN), axis=1)

# split input into train and validation sets
XaN, YaN, XvN, YvN = TPB04_split_bases(XN, SigN)
#----------------------------------------------------------------------
# PLOT THE DATA

# create new figure
# values = range(3)
# jet = plt.get_cmap('jet')
# cNorm = colors.Normalize(vmin=0, vmax=values[-1])
# scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
#
# labels = ['speed', 'dir_sin', 'dir_cos']
# inputs = [VitN, Dir_sin, Dir_cos]
# for i in values:
#     fig = plt.figure()
#     plt.title(labels[i])
#     plt.ylabel('Sigma0 Normalised')
#     # plt.xticks(k_sizes)
#     plt.xlabel(labels[i])
#
#     colorVal = scalarMap.to_rgba(i)
#     plt.scatter(inputs[i], SigN, color=colorVal, label=labels[i], s=10)
#
#     plt.tight_layout()
#     plt.legend()
#     plt.show()
#     fig.savefig('sigma_vs_' + labels[i] + '.png')

#----------------------------------------------------------------------
# MAKE PMC1 TO GENERATE THE ERROR VALUES

m = 5           # Nombre de cellules sur la couche cachées
nbitemax = 1000  # Nombre maximum d'iterations
F1 = "tah"   # Fonction d'activation des neurones de la couche cachée
F2 = "lin"   # Fonction d'activation des neurones de sortie
FF = [F1,F2] # Fonctions de Transfert : mettre autant de Fi que de m+1
dfreq = 50      # Frequence d'affichage

# initialise and train the PMC
WW = tdp.pmcinit(XaN, YaN, [m])  # save demoweini0 W1 W2 % voir aussi save/load demobases
WWmv1, minval, it_minval, nballit, plt2 = tdp.pmctrain(XaN,YaN,WW,FF,nbitemax=nbitemax,
                                                      dfreq=dfreq, dcurves=1, Xv=XvN, Yv=YvN)
# WWmv1, it_minval = tdp.pmctrain(XaN, YaN, WW, FF, nbitemax=nbitemax, dfreq=dfreq, dcurves=0)

# Get predictions for validation X, using weights from trained PMC
YmvN = tdp.pmcout(XvN, WWmv1, FF)
# Get error between validation Y (sigma) and PMC predicted Y
Evalmv = tls.errors(YvN, YmvN, errtype=["errq"])
print(Evalmv)

#----------------------------------------------------------------------
# PART 2

# calc error between each real sigma (Sig) and its prediction by PMC (train_y_pred), using train data and valid:
# get predictions for ALL the data using PMC trained in previous step
train_y_pred = tdp.pmcout(XN, WWmv1, FF)
# denormalise these predictions
train_y_pred = tls.decentred(train_y_pred, np.mean(Sig), np.std(Sig))
# calc quadratic error
ErrApp = np.square(train_y_pred - Sig)
# normalise these squared errors again (we use this quadratic error as the desired output of second network)
ErrAppN = tls.centred(ErrApp)

# split input into train and validation sets
XaN, YaN, XvN, YvN = TPB04_split_bases(XN, ErrAppN)

#----------------------------------------------------------------------
# MAKE PMC2 TO ESTIMATE VARIANCE OF NOISE

m = 5           # Nombre de cellules sur la couche cachées
nbitemax = 1000  # Nombre maximum d'iterations
F1 = "tah"   # Fonction d'activation des neurones de la couche cachée
F2 = "exp"   # Fonction d'activation des neurones de sortie
FF = [F1, F2] # Fonctions de Transfert : mettre autant de Fi que de m+1
dfreq = 50      # Frequence d'affichage

# initialise and train the PMC - the Yan/YvN are now quadratic errors
WW = tdp.pmcinit(XN, ErrAppN, [m])  # save demoweini0 W1 W2 % voir aussi save/load demobases  (1)

# ?? should I train second network on whole dataset or with split data?  (1) train with ALL
WWmv2, it_minval = tdp.pmctrain(XN, ErrAppN, WW, FF, nbitemax=nbitemax, dfreq=dfreq, dcurves=0)

# Get predictions for whole dataset, using weights from trained PMC2  (2)
# ?? should I generate predictions and then quadratic error for whole dataset?  (2) yes
YmN = tdp.pmcout(XN, WWmv2, FF)
# denormalise these predictions
train_y_pred = tls.decentred(YmN, np.mean(ErrApp), np.std(ErrApp))
# Get error between validation Y (sigma) and PMC predicted Y
Evalmv = tls.errors(SigN, YmN, errtype=["errq"])
print(Evalmv)

# **********************

def TPB04_plot_regress(couleur,WW,FF,DirP,vitesse,
                 mean_vit,std_vit,mean_y,std_y) :
    szXP = len(DirP); # Taille des donnees
    #
    # Construire la matrice d'entree normalisee
    DirPNsin = np.sin(DirP * np.pi / 180);
    DirPNcos = np.cos(DirP * np.pi / 180);
    VitP     = vitesse * np.ones((szXP,1));
    VitPN    = tls.centred(VitP, mean_vit, std_vit, 2/3);
    XPN      = np.concatenate((DirPNsin, DirPNcos, VitPN), axis=1)
    #
    # Passe avant (avec les poids adéquats)
    YPN = tdp.pmcout(XPN,WW,FF);
    #
    # Denormalisation de la sortie et plot
    YP = tls.decentred(YPN, mean_y, std_y, 2/3);
    #
    plt.plot(DirP,YP,'-',color=couleur,linewidth=4);
    plt.plot(DirP,YP,'-k',linewidth=2);
    plt.plot(DirP,YP,'-w',linewidth=1);
    plt.tight_layout()
    #
    return XPN, YPN


def TPB04_plot_encadre(XPN,YPN,couleur,WW,FF,BasXP,mean_sig,std_sig,mean_sqerr,std_sqerr) :
    # La Variance (apprise sur l'erreur quadratique)
    YsPN = tdp.pmcout(XPN,WW,FF);
    YsP  = tls.decentred(YsPN, mean_sqerr, std_sqerr);
    #
    # L'encadrement
    IntervEncadre = np.sqrt(YsP);
    YP      = tls.decentred(YPN, mean_sig, std_sig, crCoef=2/3);
    SiginfP = YP-IntervEncadre;
    SigsupP = YP+IntervEncadre;
    #
    # Plot de l'Encadrement
    plt.plot(BasXP,SiginfP,'-k',linewidth=1.5);
    plt.plot(BasXP,SigsupP,'-k',linewidth=1.5);
    plt.plot(BasXP,SiginfP,'--',color=couleur,linewidth=1.5);
    plt.plot(BasXP,SigsupP,'--',color=couleur,linewidth=1.5);
    plt.tight_layout()

    return IntervEncadre


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
    plt.tight_layout()
    #
    return XPN, YPN

# just returns the number of elements in BasX2 within Enc of allx2[i]
def TPB04_encs(BasX2,allx2,Enc, i) :
    x2=allx2[i];    # current selected Value (fixed within the loop)
    Ix2 = np.where( (BasX2>(x2-Enc))  &  (BasX2<(x2+Enc)) )[0];
    return len(Ix2)

# ********************** Direction & Wind Speed

allspeeds  = [4,8,12,16,20];      # speed intervals
Nspeeds= len(allspeeds);
Ttol  = 1;         # Tolerance sur l'angle d'incidence Theta

# calc mean and stddev of quadratic error, to use with TPB04_plot_encadre
err_mean = np.mean(ErrApp)
err_std = np.std(ErrApp)

azimutP = np.arange(min(Dir),max(Dir),1);  # Definition d'une abscisse de wind direction
azimutP = np.reshape(azimutP,(len(azimutP),1))
Nall_list = []
Enc_count_list = []
for i in np.arange(Nspeeds) :
    fig = plt.figure()
    Nall, Palette = TPB04_methodes.TPB04_plot_donnees(Dir, Vit, allspeeds, Ttol, Sig);
    Nall_list.append(Nall)
    TPB04_methodes.TPB04_complete_plotdonDSV(allspeeds, Nall, Ttol, Palette, Nspeeds)
    XPN, YPN = TPB04_plot_regress(Palette[i],WWmv1,['tah', 'lin'],azimutP,allspeeds[i],mean_vit,std_vit,mean_sig,std_sig)
    Enc = TPB04_plot_encadre(XPN,YPN,Palette[i],WWmv2,['tah','exp'],azimutP,mean_sig,std_sig,err_mean,err_std)
    Enc_count_list.append(TPB04_encs(Vit, allspeeds, Enc, i))
    plt.tight_layout()
    fig.savefig('data_and_regression_' + str(i) + '.png')
print('Number inside bracket =' + str(Nall))
print('Number inside enc =' + str(Enc_count_list))
Nall = np.array(Nall)
NEnc = np.array(Enc_count_list)
print('% in encad=' + str((NEnc / Nall) * 100))
# plt.axis('tight')

# ********************** Plot quadratic error & regression of it
# plot test on its own
fig = plt.figure()
plt.title('Quadratic error & regression')
plt.xlabel('Sigma')
plt.ylabel('Quadratic error')
# plt.xlim(-2.2, 2.2)
# plt.ylim(-1.65, 1.6)

plt.scatter(Sig, ErrApp, color='r', label='Quadratic error', edgecolors='w', s=40)

# sigmaP = np.arange(min(Sig),max(Sig),0.1);  # Definition d'une abscisse de wind direction
# sigmaP = np.reshape(sigmaP,(len(sigmaP),1))
# XPN, YPN = TPB04_plot_regress(Palette[i],WWmv2,['tah', 'exp'],sigmaP,100,mean_sig,std_sig,err_mean,err_std)

# sigmaP = np.arange(min(Sig),max(Sig),0.1);  # Definition d'une abscisse de wind direction
# sigmaP = np.reshape(sigmaP,(len(sigmaP),1))
# from operator import itemgetter
# zip_list = [list(x) for x in zip(*sorted(zip(Sig, train_y_pred), key=itemgetter(0)))]
# plt.plot(zip_list[0], zip_list[1], color='b', label='Regression')

plt.legend()
plt.tight_layout()
plt.show()
fig.savefig('quadratic_error.png')


# # ********************** Angle Incidence & Wind Speed
#
# # allangles  = [20, 30, 40, 50, 60];      # angle intervals
# allangles  = [34.5];      # angle intervals
# Nangles= len(allangles);
# Ttol  = 1;         # Tolerance sur l'angle d'incidence Theta
# plt.figure()
# Nall, Palette = TPB04_methodes.TPB04_plot_donnees(Inc, Vit, allangles, Ttol, Sig);
# TPB04_methodes.TPB04_complete_plotdonDSV(allangles, Nall, Ttol, Palette, Nangles)
#
# # calc mean and stddev of quadratic error, to use with TPB04_plot_encadre
# err_mean = np.mean(ErrApp)
# err_std = np.std(ErrApp)
#
# speedP = np.arange(min(Vit),max(Vit),0.2);  # Definition d'une abscisse de angle of incidence
# speedP = np.reshape(speedP,(len(speedP),1))
#
# for i in np.arange(Nangles) :
#     # plt.figure()
#     XPN, YPN = TPB04_demoplot_regress(Palette[i],WWmv1,['tah', 'lin'],speedP,allangles[i],
#                                   mean_vit,std_vit,mean_inc,std_inc, mean_sig,std_sig)
#     TPB04_plot_encadre(XPN,YPN,Palette[i],WWmv2,['tah','exp'],speedP,mean_sig,std_sig,err_mean,err_std)
# plt.axis('tight')

