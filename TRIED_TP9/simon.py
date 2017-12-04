allT  = [4,8,12,16,20];      # Le(s) angle(s) d'incidence Theta [34 35]
Ntheta= len(allT);
Ttol  = 1;         # Tolerance sur l'angle d'incidence Theta
plt.figure()
Nall, Palette = met.TPB04_plot_donnees(azimut, speed, allT, Ttol, sigma0);
#plt.xticks(np.arange(min(sigma0)-1, max(sigma0)+1, 1.0))

def TPB04_complete_plotdonVSI() : # Pour Completer le plot des donnees proprement dit
    plt.axis('tight')
    plt.ylabel('Sigma0 (dB)');
    xl1 = str(allT);    #!!!!!!! allT et non pas allV
    xl2 = str(Nall);
    plt.xlabel('Azimut\n Vitesse.[%s] m/s (Tol. %2.2f):  Nb.:(%s)'%(xl1,Ttol,xl2));
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
        
TPB04_complete_plotdonVSI()

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
    
azimutP = np.arange(min(azimut),max(azimut),1);  # Definition d'une abscisse de vitesse
azimutP = np.reshape(azimutP,(len(azimutP),1))
for i in np.arange(Ntheta) : 
    plt.figure()
    Nall, Palette = met.TPB04_plot_donnees(azimut, speed, allT, Ttol, sigma0);
    TPB04_complete_plotdonVSI()
    XPN, YPN = TPB04_demoplot_regress(Palette[i],WW1aux,['tah', 'lin'],azimutP,allT[i], \
                 azimut_mean,azimut_std,speed_mean,speed_std,sigma0_mean,sigma0_std);
    TPB04_plot_encadre(XPN,YPN,Palette[i],WW2aux,['tah','exp'],azimutP,sigma0_mean, \
                       sigma0_std,err2_meanaux,err2_stdaux)
plt.axis('tight')