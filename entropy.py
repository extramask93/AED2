import numpy as np
import matplotlib.pyplot as plt
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


def entropy(_Y,_X = None):
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
        y1 = np.unique(_X)
        fy = freq(_Y)
        fx = freq(_X)
        print("y: "+str(fy))
        print("x: "+str(fx))
        summ1 = 0
        for item in fy.keys():
            summ1 = summ1 + fy[item]/sum(fy.values())
        summ1 = -summ1
        summ2 = 0
        fyx = freq2(_Y,_X)
        print(fyx)
        #--------------conditional probability-----------------------
        for item in fyx.keys():
            for item2 in fyx[item].keys():
                print("P({0}|{1}) = {2}".format(item,item2,fyx[item][item2]/fy[item2]))
        #------------------probability of X---------------------------
        probabilityx = 0
        sumexternal = 0
        for item in fyx.keys():
            probabilityx = fx[item] / sum(fx.values())
            suminteranal = 0
            for item2 in fyx[item].keys():
                a = (fyx[item][item2]/fx[item])
                if(a !=0 ):
                    suminteranal = suminteranal + (a * math.log2(a))
            sumexternal = sumexternal + probabilityx * suminteranal
        print("probability of x: "+ str(-sumexternal))
        #-------------------------------------------------------------

        return -sumexternal
def infogain(X,Y):
    ey = entropy(Y,[])
    eyx = entropy(Y,X)
    return ey-eyx
if __name__ == "__main__":
    data = arff.loadarff("contact-lenses.arff")
    D1 = pd.DataFrame(data[0])
    X = np.array(D1)
    X = X.astype('U13')
    D1 = X[:, -1]
    D2 = X[:, -3]
    print(entropy(D1,[]) - entropy(D1,D2))
    #0.7772925846688998 prod
    #print(entropy(D1,D2))


