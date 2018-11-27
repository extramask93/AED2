import numpy as np
from scipy.io import arff
import pandas as pd
import math

def freq(V):
    Slownik = dict.fromkeys(np.unique(V),0)
    for e in V:
        Slownik[e] += 1
    return Slownik
def freq2(vx,vy):
    unikalne1 = np.unique(vx)
    Slownik1 = dict.fromkeys(unikalne1,0)
    
    unikalne2 = np.unique(vy)
    Slownik2 = dict.fromkeys(unikalne2,0)

    for e in range(0,len(Slownik2)):
        Slownik2[unikalne2[e]] = dict.fromkeys(np.unique(vx),0)

    for e in range(len(vx)):
        Slownik2[vy[e]][vx[e]] += 1

    return Slownik2
def entropy(_Y,_X,pr = False):
    if(len(_X)==0):
        uni = np.unique(_Y);
        mianownik = 0;
        sumator = 0;
        for e in uni:
            mianownik+=freq(_Y)[e]
        liczniki = []
        for e in uni:
            liczniki.append(freq(_Y)[e])
        prawdowpodobienstwa = []
        for e in liczniki:
            prawdowpodobienstwa.append(e/mianownik)
        for e in prawdowpodobienstwa:
            sumator+= e * math.log(e,2)
        sumator = -sumator
        return sumator
        
    else:
        fy = freq(_Y)
        fx = freq(_X)
        fyx = freq2(_Y,_X)
        #--------------conditional probability print-----------------------
        if(pr):
            for item in fyx.keys():
                for item2 in fyx[item].keys():
                    print("P({0}|{1}) = {2}".format(item,item2,fyx[item][item2]/fy[item2]))
        #------------------Entropia warunkowa zmiennej losowej Y---------------------------
        sumexternal = 0
        for item in fyx.keys():
            probabilityx = fx[item] / sum(fx.values())
            suminteranal = 0
            for item2 in fyx[item].keys():
                a = (fyx[item][item2]/fx[item])
                if(a !=0 ):
                    suminteranal = suminteranal + (a * math.log2(a))
            sumexternal = sumexternal + probabilityx * suminteranal
        #-------------------------------------------------------------
        return -sumexternal
def infogain(X,Y):
    ey = entropy(Y,[])
    eyx = entropy(Y,X)
    return ey-eyx

if __name__ == "__main__":
    ################CONTACT LENSES###############################
    #data = arff.loadarff("contact-lenses.arff")
    #D1 = pd.DataFrame(data[0])
    #X = np.array(D1)
    #X = X.astype('U13')
    #D1 = X[:, -1]
    #for i in range(0,len(X[0])-2):
    #    D2 = X[:, i]
    #    print(infogain(D1,D2))
    #############ZOO#############################################
    data = arff.loadarff("zoo.arff")
    D1 = pd.DataFrame(data[0])
    X = np.array(D1)
    X = X.astype('U13')
    D1 = X[:, -1]
    for i in range(0,len(X[0])-2):
        D2 = X[:, i]
        print(infogain(D1,D2))