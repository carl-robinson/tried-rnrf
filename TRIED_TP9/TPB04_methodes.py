# -*- coding: utf-8 -*-
#----------------------------------------------------------------------
import numpy as np
import scipy.io
from   scipy import stats
import matplotlib.pyplot as plt
from   matplotlib import cm
#----------------------------------------------------------------------
#import TPB04_methodes
def TPB04_errscat_cor(Sig,Ymv,Vbase,Gap) :
    # 1) Erreur par Intervalle
    print('......................................................');
    print(':     Tableau des Erreurs s/données denormalisées    :');
    print(': -------------------------------------------------- :');
    print(':   Variable  Nb.    RMS    RMSrel.  BIAIS    SigMoy');
    #
    Max = max(Vbase);
    Inf = 0;  Sup=Inf+Gap;
    i   = 0;
    Coul=np.zeros(len(Vbase));
    plt.figure()
    while Inf < Max  : # Pour chaque intervalle
      Ix    = np.where((Vbase>=Inf) & (Vbase<Sup))[0] ;
      Nb    = len(Ix); #print('Nb=',Nb);
      Y1_v  = Ymv[Ix]; 
      Sig_v = Sig[Ix];
      rms,biais,rmsrel = tls.errors(Sig_v,Y1_v,errtype=['rms','biais','rmsrel']);
      print('   [%3d,%3d[  %3d  %2.4f  % 2.4f  % 2.4f  % 2.4f'
                   %(Inf,Sup,Nb,rms,rmsrel,biais,np.mean(Sig_v)));
      Inf = Sup; Sup=Sup+Gap;
      #
      i=i+1; Coul[Ix]=i*Gap; # Echelle de couleur pour le diagramme de dispersion
    print('\n');
    #
    #------------------------------------------------------------------
    # 2) Diagramme de dispersion de Sigma0 par intervalle
    plt.scatter(Sig,Ymv,c=Coul,s=5,edgecolors='none');
    plt.colorbar();
    plt.plot([min(Ymv), max(Sig)],[min(Ymv), max(Sig)],color='k',linewidth=1.5); 
    plt.axis('tight'); # equal ; 
    plt.xlabel('Valeurs de sorties Sigma0 de reference (dB)');
    plt.ylabel('Sorties Sigma0 du PMC : valeurs estimées (dB)');
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
           #ipal= 1+np.floor((i-1)*256/nens)
           ipal = int(1+np.floor((i)*280/nens)); #print(ipal)
           LPi  = LPalette[ipal,:]; #print(LPi)
           Palette.append(LPi);
    #
    if Tmark == 0 :
        Tmark=[];
        for i in np.arange(nens) : 
           Tmark.append('.');
    #
    Nall=[];
    # Nnotin = []
    for i in np.arange(nens) : #1:nens
        x2=allx2[i];    # current selected Value (fixed within the loop)
        # indices de selection autour de x2
        Ix2 = np.where( (BasX2>(x2-x2tol))  &  (BasX2<(x2+x2tol)) )[0];
        # Iv2 = np.where( (BasX2<(x2-x2tol))  |  (BasX2>(x2+x2tol)) )[0];
        # plot des valeurs selectionnees
        plt.plot(BasX1[Ix2],BasY[Ix2],Tmark[i],markersize=szmark,color=Palette[i]);                   
        Nall.append(len(Ix2));
        # Nnotin.append(len(Iv2));
    #    
    return Nall,Palette
#----------------------------------------------------------------------
#
def TPB04_complete_plotdonDSV(allV, Nall, Vtol, Palette, Nvit) : # Pour Completer le plot des donnees proprement dit
    plt.axis([0, 360, -60, 5]); 
    plt.ylabel('Sigma0 (dB)');
    xl1 = str(allV);
    xl2 = str(Nall);
    plt.xlabel('angle d''azimut.\n  Vit.[%s] ms (Tol. %2.2f):  Nb.:(%s)'%(xl1,Vtol,xl2));
    plt.title('Sigma0');
    plt.grid('on');
    if 1 : #plt.colorbar();
        import matplotlib 
        bmap   = matplotlib.colors.ListedColormap(Palette); 
        bmap   = plt.cm.ScalarMappable(cmap=bmap)
        bmap.set_array([])
        hcb    = plt.colorbar(bmap)
        cticks = np.arange(0,1,1/Nvit)
        hcb.set_ticks(cticks+1/(2*Nvit))
        YLabel=[]
        for i in np.arange(Nvit) :
            YLabel.append('%d m/s'%allV[i]);
        hcb.set_ticklabels(YLabel)
#
#
#----------------------------------------------------------------------
# Plot de LA regression pour chaque vitesse retenues (sur la fenetre active)
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
    #
    return XPN, YPN
##----------------------------------------------------------------------    
#
def TPB04_plot_encadre(XPN,YPN,couleur,WW,FF,BasXP,mean_sig,std_sig,mean_sqerr,std_sqerr) :
    # La Variance (apprise sur l'erreur quadratique)
    YsPN = tdp.pmcout(XPN,WW,FF);
    YsP  = tls.decentred(YsPN, mean_sqerr, std_sqerr, 2/3);
    #
    # L'encadrement
    IntervEncadre = 2*np.sqrt(YsP);
    YP      = tls.decentred(YPN, mean_sig, std_sig, 2/3);
    SiginfP = YP-IntervEncadre;
    SigsupP = YP+IntervEncadre;
    #
    # Plot de l'Encadrement
    plt.plot(BasXP,SiginfP,'-k',linewidth=1.5);
    plt.plot(BasXP,SigsupP,'-k',linewidth=1.5);
    plt.plot(BasXP,SiginfP,'--',color=couleur,linewidth=1.5);
    plt.plot(BasXP,SigsupP,'--',color=couleur,linewidth=1.5);
#



