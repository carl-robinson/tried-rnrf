#-----------------------------------------------------------
# If triedpy modules are not directly accessible by python,
# set the path where triedpy is located :
if 1 :
    import sys
    TRIEDPY = "../../../../";  # Mettre le chemin d'accès ... 
    sys.path.append(TRIEDPY);  # ... aux modules triedpy
#
import numpy as np
import TPB03_methodes
import datetime
#-----------------------------------------------------------
# Les Données
# Choisir le fichier des données d'entrée à utiliser :
x = np.loadtxt("x.txt");
#x = np.loadtxt("hx.txt");
#x = np.loadtxt("hx_hy.txt");
#x = np.loadtxt("pb_ph.txt");
#x = np.loadtxt("pg_pd.txt");
#x = np.loadtxt("hx_hy_pb_ph.txt");
#x = np.loadtxt("hx_hy_pg_pd.txt");
# Fichier des sorties
t = np.loadtxt("t.txt");
#
#-----------------------------------------------------------
struct1     = 3;    # 1 :=> Structure lineaire sans couche cachee
                    # 2 :=> Sigmoide sans couche cachee
                    # 3 :=> Sigmoide avec une couche cachee
hidden_layer=10;    # Nombre de neuronnes caches

from_lr  =   0;     # Learning set starting point
to_lr    = 300;     # Leasrning set ending point
from_val = 300;     # Validation set starting point
to_val   = 400;     # Validation set ending point

lr        = 10.0;    # Learning rate (Pas d'apprentissage) - def: 0.1
n_epochs  = 100000; # Number of whole training set presentation
plot_flag = 1;      # 1 :=> Linear plot 
                    # 2 :=> Log plot                 
ecf  = 10;          # Error Computation Frequency (evry ecf iterations)
gdf  = 200;         # Graphic Display Frequency (evry gdf erreur computation)

rep1 = 0;           # 0 : Initialisation aléatoire des poids
                    # 1 : Poids obtenus précédemment (en fin d'app) %=>l'archi doit etre la meme
                    # 2 : Poids initiaux précédent %=>l'archi doit etre la meme
#-----------------------------------------------------------
#np.random.seed(0);
WWmv, minval, itmv, nballit, plt = TPB03_methodes.trainingter(x,t,hidden_layer,struct1,from_lr, \
       to_lr,from_val,to_val,lr,n_epochs,plot_flag,rep1,ecf,gdf);
#-----------------------------------------------------------

plt.savefig(str(struct1) + '_' + str(hidden_layer) + '_' + str(from_lr) + '_' + str(to_lr) + '_' + str(from_val) + '_' +
            str(to_val) + '_' + str(lr) + '_' + str(n_epochs) + '_' + str(plot_flag) + '_' + str(np.around(minval, 5)) + '_'
            + str(itmv) + '_' + str(nballit) + '_' + '{:%Y%m%d_%H%M%S}'.format(datetime.datetime.now()) + '.png')
