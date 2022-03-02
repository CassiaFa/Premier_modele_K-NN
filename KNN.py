import numpy as np
from sklearn.metrics import confusion_matrix

class KNN:

    def __init__(self):
        self.__label_train = []
        self.__X_train = []
        self.__metric = 'euclidean'
        self.__p = []
        self.__label = []
        self.__confusion_matrix = []
        self.__format_model = {}

    
    def distance(self, **kwargs):
        '''
        distance permet de calculer la distance d'un échantillon par rapport aux autres.

        Paramètres
        ---------------------
        :metric: {'Euclidean', 'Manhattan', 'Minkowski'}
            methode à utiliser pour calculer la distance 
        '''

        X_test = kwargs.get("X_test", None)

        if self.__metric.lower() == 'euclidean':
            if len(X_test) == 1:
                d = np.sqrt(np.sum((self.__X_train-X_test)**2, axis=1))
                return np.array(d)
            else:
                d = []
                for x in X_test:
                    d.append(np.sqrt(np.sum((self.__X_train-x)**2, axis=1)))
                return np.array(d)
        elif self.__metric.lower() == 'manhattan':
            if len(X_test) == 1:
                d = np.sqrt(np.sum(abs(self.__X_train-X_test)))
                return np.array(d)
            else:
                d = []
                for x in X_test:
                    d.append(np.sqrt(np.sum(abs(self.__X_train-X_test))))
                return d
        elif self.__metric.lower() == 'minkowski':
            if len(X_test) == 1:
                d = pow(np.sum(abs(self.__X_train-X_test)**self.__p), 1/self.__p)
                return np.array(d)
            else:
                d = []
                for x in X_test:
                    d.append(pow(np.sum(abs(self.__X_train-X_test)**self.__p), 1/self.__p))
                return np.array(d)
        else:
            raise ValueError(f"metric prend en uniquement comme valeur 'euclidean', 'manhattan', ou 'minkowski' (saisie {self.__metric})")


    def target_format(self, Y_train):
        self.__label = Y_train.sort_values().unique()

        cpt = 1
        for k in self.__label:
            self.__format_model[k] = cpt
            cpt +=1
        
        target_formated = Y_train.replace(self.__format_model).values

        return target_formated


    def train(self, X_train, label_train, **kwargs):
        self.__p = kwargs.get('p', 2)
        self.__metric = kwargs.get("metric", "euclidiean")
        self.__label_train = self.target_format(label_train) # Formatage des labels
        self.__X_train = X_train


    def prediction(self, X_test, k=5):
        d = self.distance(X_test = X_test)

        if d.ndim == 1:
            d = d.reshape(1,-1)

        # ind = np.argsort(d,axis=0)[:k,:] # k : nombre de voisin
        ind = np.argsort(d,axis=1)[:,:k] # k : nombre de voisin

        ppv = []
        for i in ind:
            ppv.append(self.__label_train[i]) # .mode()
        
        # ppv = list(map(list, zip(*ppv))) # transposé liste
        ppv = np.array(ppv)

        proba = []
        for i in range(1,3+1):
            proba.append(np.count_nonzero(ppv == i, axis=1)/k)

        proba = np.array(proba).T

        # proba_A = np.count_nonzero(ppv == 1, axis=0)/k

        # proba_B = np.count_nonzero(ppv == 2, axis=0)/k

        # proba_C = np.count_nonzero(ppv == 3, axis=0)/k

        # proba = np.array([proba_A, proba_B, proba_C])

        y_pred = np.argmax(proba, axis=1)+1

        return y_pred

    def accuracy(self, y_test, y_pred, **kwargs):
        
        y_test = y_test.replace(self.__format_model).values
        self.__confusion_matrix = confusion_matrix(y_test, y_pred)
        error_rate = (1 - np.trace(self.__confusion_matrix)/np.sum(self.__confusion_matrix))*100
        accuracy = 100-error_rate

        if kwargs.get('resume', False):
            self.resume(accuracy, error_rate)


        return accuracy, error_rate

    
    def resume(self, accuracy, error_rate):
        if not(accuracy):
            print("Le calcul de l'accuracy est nécessaire, utiliser la fonction d'instance pour la calculer")
        else:
    
            print("\n ======  Résumé des métrique  =====\n")
            print("----------------------")
            print("Matrice de confusion")
            print(self.__confusion_matrix)

            print("\n=======================")
            print(f"\nAccuracy : {accuracy}")
            print(f"\nTaux d'erreur : {error_rate}")
            print("----------------------")

            cpt = 0
            for l in self.__label:
                P = self.__confusion_matrix[cpt,cpt]/self.__confusion_matrix.sum(axis=0)[cpt]
                R = self.__confusion_matrix[cpt,cpt]/self.__confusion_matrix.sum(axis=1)[cpt]

                cpt += 1
                print(f"\nPrécision classe {l} : {P}")
                print(f"\nSensibilité classe {l} : {R}")
                print("----------------------")

            print("\n=======================\n")