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
        metric: {'Euclidean', 'Manhattan', 'Minkowski'}
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
                d = np.sqrt(np.sum(abs(self.__X_train-X_test), axis=1))
                return np.array(d)
            else:
                d = []
                for x in X_test:
                    d.append(np.sqrt(np.sum(abs(self.__X_train-x), axis=1)))
                return np.array(d)
        elif self.__metric.lower() == 'minkowski':
            if len(X_test) == 1:
                d = pow(np.sum(abs(self.__X_train-X_test)**self.__p, axis=1), 1/self.__p)
                return np.array(d)
            else:
                d = []
                for x in X_test:
                    d.append(pow(np.sum(abs(self.__X_train-x)**self.__p, axis=1), 1/self.__p))
                return np.array(d)
        else:
            raise ValueError(f"metric prend en uniquement comme valeur 'euclidean', 'manhattan', ou 'minkowski' (saisie {self.__metric})")


    def target_format(self, Y_train):
        '''
        La fonction target_format() permet d'extraire les différents labels composants la matrice cible, et de les encodés numériquement.
        
        Paramètres
        ---------------------
        Y_train : *type : array*, matrice contenant les labels que le modèle cherchera à predire
        '''

        self.__label = Y_train.sort_values().unique()

        cpt = 1
        for k in self.__label:
            self.__format_model[k] = cpt
            cpt +=1
        
        target_formated = Y_train.replace(self.__format_model).values

        return target_formated


    def train(self, X_train, label_train, **kwargs):
        '''
        La fonction train() permet de remplir les variables d'instance, cela peut-être comparer à l'entrainement du modèle.
        
        Paramètres
        ---------------------
        X_train : type : array, matrice contenant les données d'entraînement qui serviront pour estimer les prédictions réalisé. 
        
        label_train : *type : array*, matrice contenant les labels que le modèle cherchera à predire

        **kwargs :
        - metric : {'Euclidean', 'Manhattan', 'Minkowski'}
            methode à utiliser pour calculer la distance 
        - **p** : valeur de puissance pour la méthode 'Minkowski    
        '''

        self.__p = kwargs.get('p', 2) # récupération de p, si vide 2 par défaut
        self.__metric = kwargs.get("metric", "euclidiean") # récupération de la métrique choisie, "euclidean" par défaut
        self.__label_train = self.target_format(label_train) # Formatage des labels 
        self.__X_train = X_train # Enregistrement des données d'entraînement


    def prediction(self, X_test, k=5):
        '''
        La fonction prediction() permet de prédir les targets d'un jeu de données test.
        
        Paramètres
        ---------------------
        X_test : type : array, matrice contenant les données de test à partir desquel seront faites les prédictions. 
        
        k : le nombre de voisin à utiliser pour la comparaison de distance. Par défaut le k est de 5.   
        '''
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

        y_pred = np.argmax(proba, axis=1)+1

        return y_pred

    def accuracy(self, y_test, y_pred, **kwargs):
        '''
        La fonction accuracy() permet d'estimer l'accuracy du modèle.
        
        Paramètres
        ---------------------
        y_test : type : array, les labels associés au données test. 
        
        y_pred : *type : array*, les labels prédits par le model

        **kwargs :
        - resume : type : boolean, défaut = False, permet d'afficher dans la console un résumé des estimation de  performances.
        '''
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