#-----------------------------------------------------------
import numpy as np
from   matplotlib import cm
from   triedpy    import triedtools as tls
from   triedpy    import trieddeep  as tdp
#====================== méthodes communes ========================
def display_pat(x,fr,to) :
    tls.shownpat2D(x.T,[16,16],fr,to,subp=True,cmap=cm.gray_r);

#============== Méthodes pour les partie 1 et 2 ==================
def trainingter(x,t,hidden_layer,struct1,from_lr, \
     to_lr,from_val,to_val,lr,n_epochs,plot_flag,rep1,ecf,gdf) :

    x    = x.T;   #<Convention
    Xapp = x[from_lr: to_lr,:];
    Xval = x[from_val:to_val,:];
    #
    t    = t.T;   #<Convention
    Yapp = t[from_lr:to_lr,:];
    Yval = t[from_val:to_val,:]; 

    # Architecture
    if struct1 == 1 :  
        m  = [];
        F1 = "lin"
        FF = [F1];
    elif struct1 == 2 :  
        m  = [];
        F1 = "tah"
        FF = [F1];
    elif struct1 == 3 :  
        m  = [hidden_layer];
        F1 = "tah"; F2="lin";
        FF = [F1,F2];
    else :
        print("trainingter: bad struct1 value, must be 1, 2 or 3");
        sys.exit(0);

    # Init des poids
    if rep1==0 :   # Initialisation aléatoire des poids
        WW = tdp.pmcinit(Xapp, Yapp, m);
        np.save("poids0", WW); 
    elif rep1==1 : # Poids obtenus précédemment (en fin d'app)
        WW = np.load("pmcendwei.npy");
    elif rep1==2 : # Poids initiaux précédent
        WW = np.load("poids0.npy");

    # Autres Paramètres d'Apprentissage et d'affichage
    alpha    = 0.0;
    dperf    = 1;
    dprint   = 0;
    ntfpamin = 200; #nbitamin = 200;
    pval     = 1;
    dval     = 1;
    weivar_seuil = 0.0;
    #
    WWmv, minval, itmv, nballit, plt = tdp.pmctrain(Xapp, Yapp, WW, FF, nbitemax=n_epochs, gstep=lr, \
                     alpha=alpha,  weivar_seuil=weivar_seuil, \
                     tfp=ecf, dfreq=gdf, dprint=dprint, dcurves=plot_flag, dperf=dperf, \
                     Xv=Xval,Yv=Yval, pval=pval, ntfpamin=ntfpamin,dval=dval);
    return WWmv, minval, itmv, nballit, plt

#================= Méthodes pour la partie 3 =====================
def n_pattern_showing(hidden_function,w1,b1,w2,b2,x,fr,to) :
    '''Pour éviter à l'utilisateur de formater xshape, mask, skip et se
       mettre au plus près de l'exo d'origine '''
    x = x.T;
    xshape  = [16, 16]; # Dimension 2D des patterns d'entrées
    skip    = [2, 2];   # Saut (vertical (en ligne), et horizontal(en colonne)) à appliquer au déplacement du masque
    sidevalue = -1;     # Valeur à attribuer aux points extérieurs à la forme d'entrée.
    plt = tdp.wsmshowccpat(hidden_function,w1,b1,w2,b2,x,fr,to,xshape,skip,sidevalue);
    return plt
    
def errperf_computing(hidden_function,w1,b1,w2,b2,x,t) :
    '''Pour éviter à l'utilisateur de formater xshape, mask, skip et se
       mettre au plus près de l'exo d'origine '''
    x = x.T;
    t = t.T;
    xshape  = [16, 16]; # Dimension 2D des patterns d'entrées
    skip    = [2, 2];   # Saut (vertical (en ligne), et horizontal(en colonne)) à appliquer au déplacement du masque
    sidevalue = -1;     # Valeur à attribuer aux points extérieurs à la forme d'entrée.
    Y    = tdp.wsmout(hidden_function,w1,b1,w2,b2,x,xshape,skip,sidevalue);
    Err  = np.sum((t - Y)**2) / np.size(t); # erreur quadratique normalisée
    Perf = tls.classperf(t,Y,miss=1);       # Compute miss Perf      
    return Err, Perf

def trainshared(x,t,hidden_function,from_lr, \
       to_lr,from_val,to_val,mask_dimension,lr,n_epochs,plot_flag,ecf,gdf) :
    x = x.T;   #<Convention
    Xapp = x[from_lr: to_lr,:];
    Xval = x[from_val:to_val,:];
    t = t.T;   #<Convention
    Yapp = t[from_lr :to_lr,:];
    Yval = t[from_val:to_val,:];

    # Définition de la convolution (Caracteristic Map size)
    xshape  = [16, 16]; # Dimension 2D des patterns d'entrées
    # dimension du mask de convolution
    mask    = [mask_dimension, mask_dimension];
    skip    = [2, 2];  # Saut (vertical (en ligne), et horizontal(en colonne)) à appliquer au déplacement du masque
    sidevalue = -1;    # Valeur à attribuer aux points extérieurs à la forme d'entrée.

    # Init des poids
    dimout   = np.size(Yapp,1);
    w1,b1,w2,b2 = tdp.wsminit(xshape,mask,skip,dimout);

    # Autres Paramètres d'Apprentissage et d'affichage
    Fhid      = hidden_function;
    pval      = 1;  # Appréciation de min en val : 0:=>sur l'erreur; 1:=>sur la perf (revoir cas pval=0)
    ntfpamin  = 2;  # nbitamin  = 2;
    # flag pour l'affichage
    dprint    = 1;
    dcurves   = 0;
    dperf     = 1;     # 0; (si dprint==0 et dcurves==0 alors dperf ne sert pas)
    dval      = 1;     # display ou pas des éléments de validation
    #
    w1mv,b1mv,w2mv,b2mv,itmv = tdp.wsmtrain(Xapp,Yapp,w1,b1,w2,b2,Fhid,xshape,skip, \
              sidevalue=sidevalue,n_epochs=n_epochs,lr=lr,tfp=ecf,dfreq=gdf,            \
              dprint=dprint, dcurves=dcurves,dperf=dperf,                       \
              Xv=Xval,Yv=Yval,pval=pval,ntfpamin=ntfpamin,dval=dval);

    # Err et Perf sur l'ensemble d'App avec les poids obtenus en fin de procédure
    w1app, b1app, w2app, b2app = np.load("wsmendwei.npy");
    Y    = tdp.wsmout(hidden_function,w1app,b1app,w2app,b2app,Xapp,xshape,skip,sidevalue);
    Err  = np.sum((Yapp - Y)**2) / np.size(Yapp); # erreur quadratique normalisée
    Perf = tls.classperf(Yapp,Y,miss=1);          # Compute miss Perf   
    print("Ens. d'App.[%3d, %3d] : Err=%.6f; Perf=%.6f" %(from_lr+1,to_lr,Err,Perf));

    # Err et Perf sur l'ensemble de validation en son minumum
    Y    = tdp.wsmout(hidden_function,w1mv,b1mv,w2mv,b2mv,Xval,xshape,skip,sidevalue);
    Err  = np.sum((Yval - Y)**2) / np.size(Yval); # erreur quadratique normalisée
    Perf = tls.classperf(Yval,Y,miss=1);          # Compute miss Perf 
    print("Ens de Val.[%3d, %3d] : Err=%.6f; Perf=%.6f" %(from_val+1,to_val,Err,Perf));
    
    return w1mv,b1mv,w2mv,b2mv,int(itmv)
    
#-----------------------------------------------------------------
#


