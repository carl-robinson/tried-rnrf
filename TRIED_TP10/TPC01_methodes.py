import sys
import numpy as np
import matplotlib.pyplot as plt
from   matplotlib import cm
#
from   triedpy import triedtools as tls
from   triedpy import triedctk   as ctk
#
#----------------------------------------------------------------------
def app_lettre() :
    '''
    % Code pour l'apprentissage en 2 phases de la Carte
    % Topologique, dans lequel il convient d'adapter, en particulier,
    % le choix des donnees (lettre Z ou F) et la taille de la carte.
    En sortie :
    sMap       : La structure de la carte
    Xapp, Xapplabels : L'ensemble d'apprentissage et le labels (classes) associés
    classnames : Nom des classes (qui on servis à la labelisation)
    '''
    from   triedpy import triedsompy as SOM
    # Positionnement du rand pour avoir ou pas le meme alea a chaque fois
    seed = 0; np.random.seed(seed);
    #
    # Les Données :
    # Creation du jeu de donnees d'apprentissage et de leurs structures
    # (Choisir la fonction Zcreadata ou Fcreadata selon la lettre Z ou F)
    Napp  = 500; # Taille des donnees
    if 1 :   # Lettre Z
        classnames = ['T','B','D']; #Z: 'T' correspond à la bar du haut
        # comme Top, 'B', à celle du bas comme Bottom et 'D' à la Diagonale
        Xapp,Xapplabels,cnames = Zcreadata(Napp,classnames=classnames);
    elif 0 : # Lettre F
        classnames = ['T','M','L']; #F: 'T' correspond à la bar du haut
        # 'M', à celle du Milieu et 'L' à la bar gauche, comme Left
        Xapp,Xapplabels,cnames = Fcreadata(Napp,classnames=classnames);
    #
    if 0 : # Affichage des seules données
        plt.figure();
        lettreplot(Xapp);
    #
    # Me rapelle pas si une structure des données est necessaire avec Sompy ?
    #sDapp = som_data_struct(Xapp,'name','lettre', 'labels',labs, 'comp_names',cnames);
    #
    #==================================================================
    # la CARTE TOPOLOGIQUE
    #------------------------------------------------------------------
    # Choix des dimensions de la CT : remplacer les 0  par les valeurs
    # souhaitees des nombres de lignes (nlmap) et de colonne (ncmap)
    nlmap = 12;  ncmap = 12; # Nombre de lignes et nombre de colones
    if nlmap<=0 or ncmap<=0 :
        print("app_lettre : mauvais choix de dimension pour la carte");
        sys.exit(0);
    #
    # Creation d'une structure de carte initialisee (référents non initialisés)
    initmethod='random'; # 'random', 'pca'
    # initmethod='pca'; # 'random', 'pca'
    sMap  = SOM.SOM('sMap', Xapp, mapsize=[nlmap, ncmap], norm_method='data', \
                  initmethod=initmethod, varname=classnames)
    #____________________
    # Affichage de l'etat initial de la carte dans l'espace des donnees
    if 1 : # Peut ne pas être utile si on n'a pas besoin de voir l'état initial
        sMap.init_map(); # Initialisation des référents (Weight vectors of SOM)
        plt.figure();
        lettreplot(Xapp);
        ctk.showmapping(sMap, Xapp, bmus=[], seecellid=1, subp=False,override=True);
        plt.title("Etat Initial");
    #
    #==================================================================
    # APPRENTISSAGE
    #------------------------------------------------------------------
    tracking = 'on';  # niveau de suivi de l'apprentissage
    #____________
    # paramètres 1ere étape :-> Variation rapide de la temperature
    epochs1 = 20; radius_ini1 =5.00;  radius_fin1 = 1.25;
    etape1=[epochs1,radius_ini1,radius_fin1];
    #
    # paramètres 2ème étape :-> Variation fine de la temperature
    epochs2 = 50; radius_ini2 =1.25;  radius_fin2 = 0.10;
    etape2=[epochs2,radius_ini2,radius_fin2];
    #
    # Avec Sompy, les paramètres des 2 étapes sont passés en même temps pour l'
    # apprentissage de la carte.
    sMap.train(etape1=etape1,etape2=etape2, verbose=tracking);
    # show final state
    if 1 : # Peut ne pas être utile si on n'a pas besoin de voir l'état initial
        plt.figure();
        lettreplot(Xapp);
        ctk.showmapping(sMap, Xapp, bmus=[], seecellid=1, subp=False,override=True);
        plt.title("Etat Finale");
    #
    print('Map[%dx%d](%d,%2.2f,%2.2f)(%d,%2.2f,%2.2f) '
      %(nlmap,ncmap,epochs1,radius_ini1,radius_fin1,
         epochs2,radius_ini2,radius_fin2),end='');
    #
    return sMap, Xapp, Xapplabels, classnames
#----------------------------------------------------------------------
def  splitNlabs(N,classnames=None) :
    N1 = int(np.ceil(N/3));
    N2 = int(np.floor(N/3));
    N3 = N - N1 - N2;
    labs=np.empty(N).astype(str)
    if classnames is not None :
       labs[0:N1]     = classnames[0];
       labs[N1:N1+N2] = classnames[1];
       labs[N1+N2:N]  = classnames[2];
    else : # noms des classes par defaut
       labs[0:N1]     = ['A']; # %{'A'};  %{'un'};    %{'C1'}
       labs[N1:N1+N2] = ['B']; # %{'B'};  %{'deux'};  %{'C2'}
       labs[N1+N2:N]  = ['C']; # %{'C'};  %{'trois'}; %{'C3'}
    cnames = ['X', 'Y'];
    return N1,N2,N3,labs,cnames

def Zcreadata(N,classnames=None) :
    ''' Creation d'un jeu de donnees simulees en dimension 2 et en
    % forme de lettre Z.
    €n Entree :
    % N          : Nombre de points de donnees
    % classnames : Vecteur des noms des classes
    % Sorties :
    % X      : Jeu de donnees 2D en forme de lettre Z
    % labs   : Labels des donnees selon 3 classes chacune
    %          associee aux points selon leurs appartenances
    %          aux barres qui forment la lettre Z :
    %            - barre du haut   :<->: 1ere classe
    %            - barre du bas    :<->: 2eme classe
    %            - barre diagonale :<->: 3eme classe
    % cnames : Noms des variables des 2 dimensions
    '''
    N1,N2,N3,labs,cnames = splitNlabs(N,classnames)
    #                        ax         ay       bx        by
    X1 = tls.gen2duni(N1,[0.10,        1.00],[0.45,       1.50],     0);
    X2 = tls.gen2duni(N2,[0.09,       -0.95],[0.44,      -0.45],     0);
    X3 = tls.gen2duni(N3,[0.16214, -0.45882],[0.28214, 1.04118],-10.00);
    X  = np.concatenate((X1,X2,X3));
    return X, labs, cnames

def Fcreadata(N,classnames=None) :
    ''' Creation d'un jeu de donnees simulees en dimension 2 et en
    % forme de lettre f.
    €n Entree :
    % N          : Nombre de points de donnees
    % classnames : Vecteur des noms des classes
    % Sorties :
    % X      : Jeu de donnees 2D en forme de lettre f
    % labs   : Labels des donnees selon 3 classes chacune
    %          associee aux points selon leurs appartenances
    %          aux barres qui forment la lettre F :
    %            - barre du haut   :<->: 1ere classe
    %            - barre du milieu :<->: 2eme classe
    %            - barre de gauche :<->: 3eme classe
    % cnames : Noms des variables des 2 dimensions
    '''
    N1,N2,N3,labs,cnames = splitNlabs(N,classnames)
    #                        ax    ay   bx    by
    X1 = tls.gen2duni(N1,[0.2,  1.0],[0.5, 1.5], 0);
    X2 = tls.gen2duni(N2,[0.2,  0.0],[0.5, 0.5], 0);
    X3 = tls.gen2duni(N3,[0.1, -1.0],[0.2, 1.5], 0);
    X  = np.concatenate((X1,X2,X3));
    return X, labs, cnames

def lettreplot(Data) :
    ''' Affichage des donnees qui representent la lettre Z ou F
    % en 3 classes selon un schema de partitionnement pre-etabli.
    % Entrees :
    % Data    : Donnees a afficher
    %
    % Note : Il n'y a pas de passage de parametre pour le choix des
    %        couleurs ou des formes. L'utilisateur qui le souhaite
    %        peut a son gre changer cela ou intervenir directement
    %        dans ce code.
    '''
    Ndata=len(Data);
    #
    N1 = int(np.ceil(Ndata/3));   # Garder la coherence de cette ...
    N2 = int(np.floor(Ndata/3));  # ... partition avec les fonctions ...
    N3 = Ndata - N1 - N2;         # ... Zcreadata et Fcreadata.
    #
    Dx = Data[0:N1,:];
    #plt.plot(Dx[:,0],Dx[:,1],'vk',markersize=5);
    plt.scatter(Dx[:,0],Dx[:,1],marker='v',s=50,c='c');
    Dx = Data[N1:N1+N2,:];
    #plt.plot(Dx[:,0],Dx[:,1],'^k',markersize=5);
    plt.scatter(Dx[:,0],Dx[:,1],marker='^',s=50,c='g');
    Dx = Data[N1+N2:Ndata,:];
    #plt.plot(Dx[:,0],Dx[:,1],'dk',markersize=5);
    plt.scatter(Dx[:,0],Dx[:,1],marker='d',s=50,c='b');
    plt.axis( [min(Data[:,0])-0.1, max(Data[:,0])+0.1, min(Data[:,1])-0.1, max(Data[:,1])+0.1]);
#
def confus(sm,Data,Datalabels,classnames,Databmus,visu=False) :
    ''' Matrice de confusion (MC) pour une carte topologique
    En Entrée :
    sm          : La structure de la carte
    Data        : Les Données avec lesquels les référents seront labellisés
    Datalabels  : Les labels associées aux données (Data)
    classenames : Liste de string de label de classe
    Databmus    : bmus ... but ...
    visu        : Si True MC sera affichée, par défaut elle ne l'est pas
    En Sortie :
    MC   : La matrice de confusion
    Perf : La performance
    '''
    if len(Data) != len(Datalabels) :
        print("confus: Il doit y avoir autant de labels que de données")
        sys.exit(0);
    # Labelisation des reférents par vote majoritaire
    Tfreq,Ulab = ctk.reflabfreq(sm,Data,Datalabels);
    CBlabmaj   = ctk.cblabvmaj(Tfreq,Ulab);
    #
    # Matrice de confusion  : A la différence de ctk_confus de la
    # version matlab, avec ctk.confus :
    # - Si les référents ont déjà été labellisés, leurs labels (CBlabmaj)
    #      peuvent etre passés, sinon ils seront déterminés
    # - Si les BMUs des données (Databmus) ont déjà été déterminés, ils
    #      peuvent etre passés, sinon ils seront déterminés
    MC, Perf = ctk.confus(sm,Data,Datalabels,classnames,
                        CBlabels=CBlabmaj,Databmus=Databmus,visu=visu);
    return MC, Perf

def ctclassif_Umat(sm,Data,Datalabels,classnames,shapescale=400,sztext=11) :
    ''' Exemple de script pour la presentation :
    %   - d'une carte topologique (labellisee par vote majoritaire) avec :
    %       - une taille des neurones proportionnelle au nombre d'elements qu'ils captent (hits)
    %       - les indices des neurones
    %       - le decompte, par neurone, des donnees qu'il capte par label
    %       - l'affectation du label de classe des neurones selon le vote majoritaire
    %       - une couleur des neurones differente selon leur label (ou classe)
    %  - et de la matrice U
    En Entrée :
    sm          : La structure de la carte
    Data        : Les Données avec lesquels les référents seront labellisés
    Datalabels  : Les labels associées aux données (Data)
    classenames : Liste de string de label de classe
    shapescale  : Permet de régler la taille des de la repréentation des neurones
    sztext      : Taille du texte
    '''
    if len(Data) != len(Datalabels) :
        print("ctclassif_Umat: Il doit y avoir autant de labels que de données")
        sys.exit(0);
    # Classification avec la labellisation
    Tfreq,Ulab = ctk.reflabfreq(sm,Data,Datalabels);
    CBlabmaj   = ctk.cblabvmaj(Tfreq,Ulab);
    CBlabfreq  = ctk.cblabfreq(Tfreq,Ulab,csep='\n');
    cblabtext  = tls.concstrlist(CBlabfreq,CBlabmaj);
    CBilabmaj = ctk.label2ind(CBlabmaj, classnames);

    ctk.showcarte(sm,hits='Yes',shape='h',colcell=CBilabmaj,shapescale=shapescale,
              text=cblabtext,dv=-0.03,dh=-0.04,sztext=sztext,showcellid=True);
              # utiliser dv, dh pour ajuster l'emplacement du texte
   #_______________________
    # U-matrix
    sm.view_U_matrix(distance2=2, row_normalized='No', show_data='Yes', \
                     contooor='Yes', blob='No', save='No', save_dir='');
    plt.suptitle("U-matrice", fontsize=16);

#======================================================================
def set_sdata(infile,Napp) :
    ''' Mise en forme des structures de donnees avec labellisation pour
    % l'apprentissage et le test
    % Entrees :
    % infile : (string) Nom du fichier des donnees
    % Napp   : Taille des donnees pour l'ensemble d'apprentissage.
    %          (Les donnees restantes seront affectees au test)
    % Sorties :
    % Xapp, Xapplabels  : L'ensemble d'apprentissage et les labels associés
    % Xtest, Xtestlabels: L'ensemble de test et les labels associés
    '''
    # Chargement des donnees
    Xdata = np.loadtxt(infile); # (256,480)
    Ndata = np.size(Xdata,1);   # Taille totale des donnees
    Ntest = Ndata - Napp;       # Taille ensemble de test
    if Napp >= Ndata :
        print('set_sdata: Warning: Napp required(%d) restricted to %d\n'%(Napp,Ndata));
        print('                    Ntest set to zero\n');
        Ntest = 0;

    # Definition des labels
    Xlab = (np.mod(np.arange(Ndata),10)).astype(str); #print(np.shape(Xlab)) : (480,)
    #
    # Donnees d'apprentissage
    Xapp  = Xdata[:,0:Napp].T;       # Ensemble d'apprentissage
    #
    # Labellisation des donnees d'apprentissage
    Xapplabels = Xlab[0:Napp]; #        print(np.shape(Xapplabels)) :(340,)
    #
    # Donnees de test
    if Ntest > 0 :
        Xtest = Xdata[:,Napp:Ndata].T; # Ensemble de test
        #
        # Labellisation des donnees de test
        Xtestlabels = Xlab[Napp:Ndata];
    #
    return Xapp, Xapplabels, Xtest, Xtestlabels
#
#----------------------------------------------------------------------
def app_chiffres(Xapp, classnames, infile) :
    ''' Exemple de code de definition et d'apprentissage en 2 phases
    % d'une Carte Topologique (TP sur la reconnaissance de chiffres)
    En Entree :
    Xapp       : Les données d'apprentissage
    classnames : Liste des nom de classe
    infile     : Le nom du fichier d'où a été extrait Xapp (ne sert que
                 le print final)
    '''
    from   triedpy import triedsompy as SOM
    #==================================================================
    # Definition de la CARTE TOPOLOGIQUE
    #------------------------------------------------------------------
    nlmap  = 5;  ncmap = 5; # Nombre de lignes et nombre de colones
    #
    # Creation d'une structure de carte
#    initmethod='pca'; # 'random', 'pca'
    initmethod='random'; # 'random', 'pca'
    sMap  = SOM.SOM('sMap', Xapp, mapsize=[nlmap, ncmap], norm_method='data', \
                  initmethod=initmethod, varname=classnames)
    #
    #==================================================================
    # APPRENTISSAGE
    #------------------------------------------------------------------
    tracking = 'on';  # niveau de suivi de l'apprentissage
    #____________
    # paramètres 1ere étape :-> Variation rapide de la temperature
    # in1 = 20.0
    # in2 = 10.0
    # in3 = 0.10
    in1 = 20.0
    in2 = 10.0
    in3 = 0.1
    epochs1 = 50; radius_ini1 = in1;  radius_fin1 = in2;
    etape1=[epochs1,radius_ini1,radius_fin1];
    #
    # paramètres 2ème étape :-> Variation fine de la temperature
    epochs2 = 100; radius_ini2 = in2;  radius_fin2 = in3;
    etape2=[epochs2,radius_ini2,radius_fin2];
    #
    # Avec Sompy, les paramètres des 2 étapes sont passés en même temps pour l'
    # apprentissage de la carte.
    sMap.train(etape1=etape1,etape2=etape2, verbose=tracking);
    #
    print('file %s (%d,%2.2f,%2.2f)(%d,%2.2f,%2.2f) '
      %(infile,epochs1,radius_ini1,radius_fin1,
         epochs2,radius_ini2,radius_fin2),end='');
    #
    return sMap
#----------------------------------------------------------------------
def display_pat(x,fr,to) :
    ''' Visualisation des chiffres manuscrits (avec les donnees brutes)
    % Entrees :
    % x      : Matrice (256xN) d'echantillon de chiffres manuscrits
    % fr, to : Indices de debut (from) et de fin (to) de la fourchette
               des chiffres à visualiser.
    '''
    tls.shownpat2D(x.T,[16,16],fr,to,subp=True,cmap=cm.gray_r);
#----------------------------------------------------------------------






