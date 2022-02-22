from matplotlib.pyplot import axis
import numpy as np


def distance(metric='euclidean', **kwargs):
    '''
    distance permet de calculer la distance d'un échantillon par rapport aux autres.

    Paramètres
    ---------------------
    :metric: {'Euclidean', 'Manhattan', 'Minkowski'}
        methode à utiliser pour calculer la distance 
    '''
    X_train = kwargs.get("X_train", None)
    X_test = kwargs.get("X_test", None)
    p = kwargs.get('p', 2)

    if metric.lower() == 'euclidean':
        if len(X_test) > 1:
            d = np.sqrt(np.sum((X_train-X_test)**2, axis=1))
            return d
        else:
            d = []
            for x in X_test:
                d.append(np.sqrt(np.sum((X_train-x)**2, axis=1)))
            return d
    elif metric.lower() == 'manhattan':
        if len(X_test) > 1:
            d = np.sqrt(np.sum(abs(X_train-X_test)))
            return d
        else:
            d = []
            for x in X_test:
                d.append(np.sqrt(np.sum(abs(X_train-X_test))))
            return d
    elif metric.lower() == 'minkowski':
        if len(X_test) > 1:
            d = pow(np.sum(abs(X_train-X_test)**p), 1/p)
            return d
        else:
            d = []
            for x in X_test:
                d.append(pow(np.sum(abs(X_train-X_test)**p), 1/p))
            return d
    else:
        raise ValueError(f"metric prend en uniquement comme valeur 'euclidean', 'manhattan', ou 'minkowski' (saisie {metric})")


def KNN(data_test, data_train, label_train, k=1, **kwargs):
    metric = kwargs.get("metric", None)

    d = distance(metric, X_test=data_test, X_train=data_train)
    ind = np.argsort(d,axis=1)[:,:k] # k : nombre de voisin

    ppv = []
    for i in ind:
        ppv.append(label_train.iloc[i].mode()) 