#-----------------------------------------------------------
# If triedpy modules are not directly accessible by python,
# set the path where triedpy is located :
if 1 :
    import sys
    TRIEDPY = "../../../../";  # Mettre le chemin d'accès ... 
    sys.path.append(TRIEDPY);  # ... aux modules triedpy
#
import TPB03_methodes
import numpy as np
import datetime
import sys

hidden_function_list = ['lin', 'tgh']
mask_dimension_list = [3, 5, 8, 10, 12]

for hidden_function in hidden_function_list:
    for mask_dimension in mask_dimension_list:
        for i in range(5):
            #-----------------------------------------------------------
            from_lr  =   0;     # Learning set starting point
            to_lr    = 300;     # Learning set ending point
            from_val = 300;     # Validation set starting point
            to_val   = 400;     # Validation set ending point

            lr        = 0.1;    # Learning rate (Pas d'apprentissage)
            n_epochs  = 100;    # Number of whole training set presentation
            plot_flag = 1;      # 1 :=> Linear plot
                                # 2 :=> Log plot
            # hidden_function='lin'; # 'lin' 'tgh'
            # mask_dimension = 3; # Taille du masque carre
            ecf  = 10;          # Error Computation Frequency (evry ecf iterations)
            gdf  = 1;           # Graphic Display Frequency (evry gdf erreur computation)
            #-----------------------------------------------------------

            # saving sys out to file
            savestring = hidden_function + '_' + str(mask_dimension) + '_' + '{:%Y%m%d_%H%M%S}'.format(datetime.datetime.now())
            orig_stdout = sys.stdout
            f = open(savestring + '.txt', 'w')
            sys.stdout = f

            # Les Données
            x = np.loadtxt("x.txt");
            t = np.loadtxt("t.txt");
            #
            # Apprentissage Avec validation ----------------------------
            #np.random.seed(0);
            w1mv,b1mv,w2mv,b2mv,itmv = TPB03_methodes.trainshared(x,t,hidden_function,from_lr, \
                   to_lr,from_val,to_val,mask_dimension,lr,n_epochs,plot_flag,ecf,gdf);
            #
            # Affichage de l'activation de la carte de caracteristique
            fr=1; to=12; # indices (from, to) des formes à visualiser
            plt = TPB03_methodes.n_pattern_showing(hidden_function,w1mv,b1mv,w2mv,b2mv,x,fr,to);

            plt.savefig(savestring + '.png')

            sys.stdout = orig_stdout
            f.close()

