import numpy as np


def schiolerClean(X) :
    Y = np.sin(np.pi*X)*((X>-1) & (X<1));
    return Y;

def schioler(N,sigma=0.2) :
    ''' Calcul de la fonction schioler définie comme suit :
    | Sur les abscisses les points suivent une distribution uniforme sur [-2, 2];
    | y = f(x) + delta  où :
    |     f(x) =  sin (pi.x)    sur  ] -1, 1 [
    |     f(x) =  0             sur  [ -2,-1 ]   U  [1,  2]
    |     avec delta un bruit qui suit une distribution normale N(0 ; sigma**2)
    |
    |   X, Y = schioler(N,sigma)
    |
    | N     : Nombre de données (points d'abscisses) à générer
    | sigma : Sigma de la loi Normale (N(0 ; sigma**2)) suivie par le bruit à ajouter
    |         aux données (0.2 est la valeur par defaut qui correspond à l'énoncé)
    | En sortie :
    | X : Les valeurs d'abscisse tirées aléatoirement
    | Y : Les valeurs de sortie de la fonction
    '''       
    X = 4*(np.random.rand(N,1)-0.5)
    X = np.sort(X,0);
    
    #Y = np.sin(np.pi*X)*((X>-1) & (X<1));
    Y = schiolerClean(X);

    if sigma != 0 :# Add noise       
        #Ynoise = Y + sigma * np.random.randn(N,1);
        Y = Y + sigma * np.random.randn(N,1);
        #return X, Y, Ynoise
    #else :
        #return X, Y
    return X, Y


def silverClean(X) :
    Y = np.sin(2*np.pi*(1-X)**2)*(X>0);   
    return Y;

def silverman(N,sigma=0.05) :
    ''' Calcul de la fonction silverman définie comme suit :
    | Sur les abscisses les points suivent une distribution uniforme sur [-0.25, 1];
    | y = f(x) + delta(x)  où :
    |     f(x) =  sin(2.pi(1-.x)**2)   sur  ] 0, 1 [
    |     f(x) =  0                    sur  [-0.25, 0 ]
    |     delta(x) un bruit qui suit une distribution normale N(0 ; sigma(x)**2), avec
    |     avec la variance sigma(x)**2 =  0.0025   si x <= 0.05
    | 			               =  x**2     si x >  0.05
    | X, Y = silverman(N,sigma)
    |
    | N     : Nombre de données (points d'abscisses) à générer
    | sigma : Sigma de la loi Normale (N(0 ; sigma**2)) suivie par le bruit à ajouter
    |         aux données. Par défaut l'écart type (sigma) de 0.05 correspond à une
    |         variance de 0.0025
    | En sortie :
    | X : Les valeurs d'abscisse tirées aléatoirement
    | Y : Les valeurs de sortie de la fonction
    '''       
    X = np.random.rand(N,1);
    X = X*1.25-0.25;   
    X = np.sort(X,0);
    
    #Y = np.sin(2*np.pi*(1-X)**2)*(X>0);
    Y = silverClean(X);
    
    if sigma != 0 : # Add noise
        fsig    = np.copy(X);
        I       = np.where(X<0.05)[0];
        fsig[I] = 0.05;
        fsig    = fsig**2;
        Y  = Y + fsig * np.random.randn(N,1);
    return X, Y












