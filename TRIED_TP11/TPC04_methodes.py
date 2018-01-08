import numpy as np
from triedpy import triedtools as tls
from triedpy import triedctk   as ctk
#----------------------------------------------------------------------
def serielabset(Var,varname,tfen,Varlab,classname) :
    ''' Construction de series chronologiques a partir
    % des termes d'une variable selon une taille de fenetre donnee.
    % - Labellisation, en 2 classes seulement, des series selon un 
    %   critere de norm sur une variable.     
    % - Repartition des series en ensembles d'apprentissage et de test
    % Entrees   :
    % Var       : Vecteur de la variable à partir duquel on va construire la
    %             serie chronologique
    % varname   : Nom de la variable
    % tfen      : taille de la fenetre à appliquer
    % Varlab    : variable sur laquelle on applique le critère de norm pour
    %             la labellisation.
    % classname : noms des (deux) labels de classe 

    % Sorties   :
    % Xapp      : L'ensemble des données d'apprentissage
    % applabels : les labels associés aux données d'apprentissage
    % Xtest     : L'ensemble des données de test
    % testlabels: Les labels associés aux données de test
    % comp_names: Les noms des composantes de la serie sous la forme
    %             varname(T-n).
    % Notes :
    % 1) On applique le critère de labellisation de la façon suivante : 
    %       si norm(Varlab) > 1
    %           alors le label est celui du 2eme nom de classname
    %           sinon le label est celui du 1er nom de classname
    %       sachant que la formule utilisée pour norm(Varlab) est :
    %         norm(Varlab) = (Varlab - mean(Varlab)) / std(Varlab);
    % 2) Les ensembles d'apprentissage et de test sont a priori 
    %    constitues a raison de 3/4 des données disponibles pour
    %    le 1er et du quart restant pour le second. Cette repartition
    %    peut etre modifiee.
    '''
    # some init
    seuil_norm = 1;        # Seuil d'etiquettage du phenomene el nino.
    Nvar       = len(Var); # nombre d'elements dans la variable Var
    #
    # Formation de la serie en fonction de tfen
    Nserie = Nvar-(tfen); # Taille de la serie
    Xserie = [];
    comp_names = np.empty((tfen)).astype(str)        
    for i in np.arange((tfen)) :           
        Xserie = np.append(Xserie,Var[i:i+Nserie]);
        # au passage on formate les noms des composantes
        comp_names[i] = "%s(T-%d)"%(varname,tfen-i);
    Xserie =  np.reshape(Xserie, (tfen,Nserie)).T  
    # On re-formate le dernier car T-0 c'est pas joli
    comp_names[i]  = "%s(T)"%(varname);
    #
    # --------------------------------------------------------
    # Etiquettage des données (Nino or not Nino) selon que 
    # la variable choisie normalisée > seuil
    #
    VarN  = (Varlab - np.mean(Varlab)) / np.std(Varlab,ddof=1);
    # ... Puis on positionne la labellisation simplement en t+1, ... 
    VarN  = VarN[tfen:Nvar];
    # Application du seuil de normalisation "Nino"
    INino = np.where(VarN >= seuil_norm)[0];
    #
    # Labellisation Non "Nino" ou "NON Nino"
    labs = np.empty(Nserie).astype(str)
    labs[0:Nserie] = 'no'; # Initialisation : tout a "Non Nino"
    labs[INino]    = 'NI'; # Evenement "Nino"
    NinoLabels     = labs;
    #
    # --------------------------------------------------------
    # Formation des ensembles d'Apprentissage et de Test
    Iperm  = np.random.permutation(Nserie);
    Napp   = int(round(Nserie*0.75+0.0000001)); # 3/4 pour l'apprentissage
    Xapp   = Xserie[Iperm[0:Napp],:];
    Xtest  = Xserie[Iperm[Napp:Nserie],:];  # le reste 1/4 pour le test
    Ntest  = len(Xtest);
    #
    # repartition des labels
    applabels = NinoLabels[Iperm[0:Napp]]
    testlabels= NinoLabels[Iperm[Napp:Nserie]]
    #
    return Xapp, applabels, Xtest, testlabels, comp_names
#----------------------------------------------------------------------
def drasticperf2(sMap,Xapp,Xtest,Class,classname,applabels,testlabels) :
    ''' Calcul de performance en classification d'une carte topologique
    % pour un ensemble d'apprentissage et de un ensemble de test sachant
    % que c'est avec le 1er qu'on realise la labellisation des référents.
    % Le calcul de performance est effectuer avec les termes de la matrice
    % de confusion pour une seule classe C donnee selon la formule :
    %    1-((C prevu a tord + C non prevu)/nombre total de C reel) 
    %
    % Entrees :
    % sMap      : Structure de la carte toppologiquer
    % Xapp      : L'ensemble des données d'apprentissage
    % Xtest     : L'ensemble des données de test
    % Class     : Indice de la classe concernee
    % classname : Vecteur des noms des classes
    % applabels : les labels associés aux données d'apprentissage
    % testlabels: Les labels associés aux données de test
    %
    % Sorties :
    % Perfapp  : La performance en apprentissage
    % Perftest : La performance en test
    % MCapp    : La matrice de confusion sur l'ensemble d'apprentissage
    % MCtest   : La matrice de confusion sur l'ensemble de test
    '''
    # Labellisation des referents de la carte selon un vote majoritaire
    # (necessaire pour calculer la matrice de confusion, puis la perf)
    Tfreq,Ulab = ctk.reflabfreq(sMap,Xapp,applabels);
    CBlabmaj   = ctk.cblabvmaj(Tfreq,Ulab);
    #
    # Matrice de confusion et performance sur les donnees de test
    bmustst    = ctk.mbmus(sMap, Data=Xtest);
    Tstmaplabs = ctk.mapclassif(sMap,CBlabmaj,Xtest,bmustst);
    Tstilab    = ctk.label2ind(testlabels, classname); 
    Tstmapilab = ctk.label2ind(Tstmaplabs, classname);
    MCtest     = tls.matconf(Tstilab, Tstmapilab, visu=0);
    # vérifs : On suppose que tout est ok ...
    # Calcul de la performance en test
    #Perf = 1 - (MC[1,2]+MC[2,1]+MC[2,3])/MC[2,4];
    SomColC = np.sum(MCtest[:,Class])
    SomLigC = np.sum(MCtest[Class,:])
    Perftest = (1 - ( (  SomColC-MCtest[Class,Class]  \
                       + SomLigC-MCtest[Class,Class]  \
                      ) / SomLigC                 \
               )     )*100;

    # Matrice de confusion et performance sur les donnees de d'apprentissage
    bmusapp    = ctk.mbmus(sMap, Data=Xapp);
    Appmaplabs = ctk.mapclassif(sMap,CBlabmaj,Xapp,bmusapp); 
    Appilab    = ctk.label2ind(applabels, classname); 
    Appmapilab = ctk.label2ind(Appmaplabs, classname);
    MCapp      = tls.matconf(Appilab, Appmapilab, visu=0);
    # vérifs : On suppose que tout est ok ...
    SomColC = np.sum(MCapp[:,Class])
    SomLigC = np.sum(MCapp[Class,:])
    Perfapp = (1 - ( (  SomColC-MCapp[Class,Class]  \
                       + SomLigC-MCapp[Class,Class]  \
                      ) / SomLigC                 \
               )     )*100;
  
    return Perfapp,Perftest,MCapp,MCtest
#----------------------------------------------------------------------
    
