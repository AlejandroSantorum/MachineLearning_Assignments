################################################################################
#   Authors:
#       · Alejandro Santorum Varela - alejandro.santorum@estudiante.uam.es
#       · Jose Manuel Chacon Aguilera - josem.chacon@estudiante.uam.es
#   File: Clasificador.py
#   Date: Dec. 20, 2020
#   Project: Assignment 3 Fundamentals of Machine Learning
#   File Description: Implementation of class 'Clasificador'. The rest of
#       classifiers inherit that class and implements its own training
#       and validation methods.
#
################################################################################

from abc import ABCMeta,abstractmethod
import numpy as np
import random
import scipy
from scipy.stats import norm
import math
from collections import Counter

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier



class Clasificador:

    # Clase abstracta
    __metaclass__ = ABCMeta

    # Metodos abstractos que se implementan en casa clasificador concreto
    @abstractmethod
    # TODO: esta funcion debe ser implementada en cada clasificador concreto
    # datosTrain: matriz numpy con los datos de entrenamiento
    # atributosDiscretos: array bool con la indicatriz de los atributos nominales
    # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion de variables discretas
    def entrenamiento(self,datosTrain,atributosDiscretos,diccionario):
        pass


    @abstractmethod
    # TODO: esta funcion debe ser implementada en cada clasificador concreto
    # devuelve un numpy array con las predicciones
    def clasifica(self,datosTest,atributosDiscretos,diccionario):
        pass


    # Obtiene el numero de aciertos y errores para calcular la tasa de fallo
    # datos is numpy array
    # pred is numpy array
    def error(self, datos, pred):
        # Aqui se compara la prediccion (pred) con las clases reales y se calcula el error
        hit_rate = sum(datos == pred)/len(pred)
        return 1-hit_rate # miss rate is the complement of hit rate


    # Realiza una clasificacion utilizando una estrategia de particionado determinada
    # TODO: implementar esta funcion
    def validacion(self, particionado, dataset, clasificador, seed=None):
        # Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
        # - Para validacion cruzada: en el bucle hasta nv entrenamos el clasificador con la particion de train i
        # y obtenemos el error en la particion de test i
        # - Para validacion simple (hold-out): entrenamos el clasificador con la particion de train
        # y obtenemos el error en la particion test. Otra opci�n es repetir la validaci�n simple un n�mero especificado de veces, obteniendo en cada una un error. Finalmente se calcular�a la media.
        random.seed(seed)
        np.random.shuffle(dataset.datos)

        particionado.creaParticiones(dataset.datos, seed)

        errores = []

        for particion in particionado.particiones:
            # Partitioning
            datostrain = dataset.datos[particion.indicesTrain, :]
            datostest = dataset.datos[particion.indicesTest, :]
            # Training
            clasificador.entrenamiento(datostrain, dataset.nominalAtributos, dataset.diccionario)
            # Predicting
            pred = clasificador.clasifica(datostest, dataset.nominalAtributos, dataset.diccionario)
            # Testing error
            ydatos = datostest[:,-1]
            err = clasificador.error(ydatos, pred)

            errores.append(err)
        # We return the array of errores. We can calculate later its mean and std
        return errores



##############################################################################
##############################################################################

class ClasificadorNaiveBayes(Clasificador):

    def __init__(self, laplace=True):
        self.prior_probs = []
        self.likelihoods = []
        self.laplace = laplace


    def _multinomialNB(self, xdata, ydata, feat_idx, diccionario):
        n_xi = len(diccionario[feat_idx])
        n_classes = len(diccionario[-1])
        theta_mtx = np.zeros((n_xi, n_classes))

        for value in diccionario[feat_idx]:
            feat_val_idx = diccionario[feat_idx][value]
            for class_name in diccionario[-1]:
                class_idx = diccionario[-1][class_name]
                # Calculating likelihood probability
                theta_mtx[feat_val_idx][class_idx] = sum((xdata[:,feat_idx] == feat_val_idx)&(ydata == class_idx))/sum(ydata == class_idx)

        # applying laplace correction
        if self.laplace and 0 in theta_mtx:
            theta_mtx += np.ones((n_xi, n_classes))

        return theta_mtx


    def _gaussianNB(self, xdata, ydata, feat_idx, diccionario):
        n_classes = len(diccionario[-1])

        theta_mtx = np.zeros((n_classes, 2)) # 2 columns: mean and variance for each class

        for class_name in diccionario[-1]:
            class_idx = diccionario[-1][class_name]
            # We calculate the mean coditioned to each possible class
            mean_sum = sum(elem for (idx, elem) in enumerate(xdata[:,feat_idx]) if ydata[idx]==class_idx)
            mean_class = mean_sum/sum(ydata == class_idx)
            # We calculate the variance conditioned to each possible class
            var_sum = sum((elem-mean_class)**2 for (idx, elem) in enumerate(xdata[:,feat_idx]) if ydata[idx]==class_idx)
            var_class = var_sum/sum(ydata == class_idx)

            theta_mtx[class_idx][0] = mean_class
            theta_mtx[class_idx][1] = var_class

        return theta_mtx


    def entrenamiento(self,datostrain,atributosDiscretos,diccionario):
        xdata = datostrain[:,:-1] # all rows, all columns but last one
        ydata = datostrain[:,-1]  # all rows, just last column

        m, n = xdata.shape     # number of examples, number of features
        n_classes = len(diccionario[-1])  # number of different classes

        # Calculating prior probabilities
        self.prior_probs = np.zeros(n_classes) # initializing array of prior probs with zeros
        for class_name in diccionario[-1]:
            class_idx = diccionario[-1][class_name]
            self.prior_probs[class_idx] = sum((class_idx == ydata))/m # P(y=i) = count(ydata==i)/len(ydata)

        likelihoods_list = []
        # Calculating likelihoods
        for feat_idx in range(n):
            if atributosDiscretos[feat_idx]:
                # calculating frequentist probs for discrete features
                theta_mtx = self._multinomialNB(xdata, ydata, feat_idx, diccionario)
            else:
                # calculating means and variances for continuous features
                theta_mtx = self._gaussianNB(xdata, ydata, feat_idx, diccionario)

            likelihoods_list.append(theta_mtx)

        self.likelihoods = np.asarray(likelihoods_list, dtype="object")


    # TODO: implementar
    def clasifica(self,datostest,atributosDiscretos,diccionario):
        xdata = datostest[:,:-1] # all rows, all columns but last one
        ydata = datostest[:,-1]  # all rows, just last column

        ndata, n_feat = xdata.shape     # number of examples, number of features
        n_classes = len(diccionario[-1])  # number of different classes

        pred = []
        for i in range(ndata):
            classes_probs = []
            for k in range(n_classes):
                class_p = self.prior_probs[k]
                for feat_idx in range(n_feat):
                    if atributosDiscretos[feat_idx]:
                        # calculating posterior probability
                        class_p *= self.likelihoods[feat_idx][int(xdata[i][feat_idx])][k]
                    else:
                        mean = self.likelihoods[feat_idx][k][0]
                        var = self.likelihoods[feat_idx][k][1]
                        # calculating posterior probability
                        class_p *= norm.pdf(xdata[i][feat_idx], loc=mean, scale=math.sqrt(var))
                classes_probs.append(class_p)
            pred.append(classes_probs.index(max(classes_probs)))

        return np.asarray(pred, dtype="object")



# scikit-learn NaiveBayes classifier encapsulated in our own general class
class ClasificadorNaiveBayesSK(Clasificador):

    def __init__(self, laplace=True, gaussian_feat=True):
        if gaussian_feat:
            self.clf = GaussianNB()
        else:
            self.clf = MultinomialNB(alpha=int(laplace))


    def entrenamiento(self,datostrain,atributosDiscretos,diccionario):
        xdata = datostrain[:,:-1] # all rows, all columns but last one
        ydata = datostrain[:,-1]  # all rows, just last column

        self.clf.fit(xdata,ydata)


    # TODO: implementar
    def clasifica(self,datostest,atributosDiscretos,diccionario):
        xdata = datostest[:,:-1] # all rows, all columns but last one

        return self.clf.predict(xdata)



##############################################################################
##############################################################################

class ClasificadorVecinosProximos(Clasificador):

    def __init__(self, K=5, dist='euclidean'):
        self.K = K
        self.dist = dist
        self.Sigma = None
        self.invSigma = None
        self.xtrain = None
        self.ytrain = None


    def entrenamiento(self,datosTrain,atributosDiscretos,diccionario):
        # We suppose data is already normalized
        self.xtrain = datosTrain[:,:-1] # all rows, all columns but last one
        self.ytrain = datosTrain[:,-1]  # all rows, just last column
        if self.dist == 'mahalanobis':
            # we use only train set to estimate covariance matrix
            self.Sigma = np.cov(self.xtrain, rowvar=False)
            self.invSigma = np.linalg.inv(self.Sigma)


    def clasifica(self,datosTest,atributosDiscretos,diccionario):
        xtest = datosTest[:,:-1] # all rows, all columns but last one
        ytest = datosTest[:,-1]  # all rows, just last column

        ntest, n_feat = xtest.shape  # number of examples, number of features
        ntrain = self.xtrain.shape[0]

        pred = []
        for idx_test in range(ntest):
            distances = []
            for idx_train in range(ntrain):
                if self.dist == 'euclidean':
                    # calculating euclidean distance between a test set example and a train set example
                    distances.append(math.sqrt(np.sum((xtest[idx_test,:]-self.xtrain[idx_train,:])**2)))
                elif self.dist == 'manhattan':
                    # calculating manhattan distance between a test set example and a train set example
                    distances.append(np.sum(np.absolute(xtest[idx_test,:]-self.xtrain[idx_train,:])))
                elif self.dist == 'mahalanobis':
                    # calculating mahalanobis distance between a test set example and a train set example
                    distances.append(scipy.spatial.distance.mahalanobis(xtest[idx_test,:],\
                                                                        self.xtrain[idx_train,:], self.invSigma))
                else:
                    raise ValueError('The introduced distance is not available')
            # Sorting distances list
            sorted_dist = sorted(distances)
            pred_aux = []
            for d in sorted_dist[:self.K]: # Getting K-nearest neighbors
                idx = distances.index(d)
                pred_aux.append(self.ytrain[idx])
            # Getting most common class
            [(p_class, times_class)] = Counter(pred_aux).most_common(1)
            pred.append(p_class)

        return np.asarray(pred, dtype="object")



class ClasificadorVecinosProximosSK(Clasificador):

    def __init__(self, K=5, dist='euclidean'):
        self.K = K
        self.dist = dist
        self.clf = None # initialized in training method


    def entrenamiento(self,datosTrain,atributosDiscretos,diccionario):
        xdata = datosTrain[:,:-1] # all rows, all columns but last one
        ydata = datosTrain[:,-1]  # all rows, just last column
        
        if self.dist == 'mahalanobis':
            # mahalanobis distance needs an additional param, covariance matrix
            self.clf = KNeighborsClassifier(n_neighbors=self.K, metric=self.dist,\
                                            metric_params={'V': np.cov(xdata, rowvar=False)})
        else:
            self.clf = KNeighborsClassifier(n_neighbors=self.K, metric=self.dist)

        self.clf.fit(xdata,ydata)


    # TODO: implementar
    def clasifica(self,datosTest,atributosDiscretos,diccionario):
        xdata = datosTest[:,:-1] # all rows, all columns but last one

        return self.clf.predict(xdata)



##############################################################################
##############################################################################

def sigmoid(z):
    return 1/(1+math.exp(-z))



class ClasificadorRegresionLogistica(Clasificador):

    def __init__(self, learning_rate=0.1, nepochs=100):
        self.W = None
        self.lr = learning_rate
        self.nepochs = nepochs


    def entrenamiento(self, datosTrain, atributosDiscretos, diccionario):
        n_data = datosTrain.shape[0] # number of train examples
        intercept = np.ones((n_data, 1))
        # all rows, all columns but last one, adding intercept (column of ones)
        xdata = np.hstack((intercept, datosTrain[:,:-1]))
        ydata = datosTrain[:,-1] # all rows, just last column

        n_feat = xdata.shape[1] # number of features

        # Generating random initial weights in [-0.5, 0.5]
        # Remeber n_feat includes intercept column
        self.W = np.random.uniform(-0.5, 0.5, size=n_feat)

        for epoch in range(self.nepochs):
            for j in range(n_data):
                # evaluating sigmoid function in W.T * x_j
                sig = sigmoid(np.dot(xdata[j,:], self.W))
                # updating weights
                self.W = self.W - self.lr*(sig - ydata[j])*xdata[j,:]


    def clasifica(self, datosTest, atributosDiscretos, diccionario):
        n_data = datosTest.shape[0] # number of test examples
        intercept = np.ones((n_data, 1))
        # all rows, all columns but last one, adding intercept (column of ones)
        xdata = np.hstack((intercept, datosTest[:,:-1]))
        ydata = datosTest[:,-1] # all rows, just last column

        n_feat = xdata.shape[1] # number of features

        pred = []
        for j in range(n_data):
            # calculate probability of belonging to class 1
            sig = sigmoid(np.dot(self.W, xdata[j,:]))
            if sig >= 0.5: # predicted class 1
                pred.append(1)
            else: # predicted class 0
                pred.append(0)

        return np.asarray(pred, dtype="object")




class ClasificadorRegresionLogisticaSK(Clasificador):

    def __init__(self, learning_rate=0.1, nepochs=100, sgd=False):
        self.lr = learning_rate
        self.sgd = sgd
        if sgd:
            # loss = 'log' => equivalent to logistic regression
            self.clf = SGDClassifier(loss='log', penalty=None,
                                     learning_rate='constant', eta0=learning_rate,
                                     max_iter=nepochs, tol=1e-4)
        else:
            self.clf = LogisticRegression(solver='lbfgs', max_iter=nepochs)


    def entrenamiento(self,datosTrain,atributosDiscretos,diccionario):
        xdata = datosTrain[:,:-1] # all rows, all columns but last one
        ydata = datosTrain[:,-1]  # all rows, just last column

        self.clf.fit(xdata,ydata)


    def clasifica(self,datosTest,atributosDiscretos,diccionario):
        xdata = datosTest[:,:-1] # all rows, all columns but last one

        return self.clf.predict(xdata)



##############################################################################
##############################################################################


class AlgoritmoGenetico(Clasificador):
    

    def __init__(self, n_population=100, max_rules=5, nepochs=100, cross_prob=0.7, bitflip_prob=None, add_rule_prob=0.4, elite_perc=0.05):
        self.n_population = n_population
        self.max_rules = max_rules
        self.nepochs = nepochs
        self.cross_prob = cross_prob
        self.bitflip_prob = bitflip_prob
        self.elite_perc = elite_perc
        self.population = []
        self.rules_len = None
        self.best_solution = None
        self.max_prior = None
        if add_rule_prob >= 0.5:
            raise ValueError('Parameter \'add_rule_prob\' must be in [0, 0.5). It cannot be greater or equal than 0.5')
        else:
            self.add_rule_prob = add_rule_prob



    def __init_population(self):
        # Creating initial population of size 'self.n-population'
        for i in range(self.n_population):
            # Creating a new individual, represented as a list of rules
            new_individual = []
            # The number of rules of the individual is chosen randomly,
            # between 1 (minimum number of rules) and 'self.max_rules'
            n_rules = random.randint(1, self.max_rules)
            for j in range(n_rules): # Creating 'n_rules' new rules
                # Asserting that new rule cannot have EVERY gene equal 0 or 1
                new_rule = []
                while sum(new_rule)==0 or sum(new_rule)==len(new_rule):
                    # Creating a new rule as a random binary string of size 'feat_size' + 1 (predicted class)
                    new_rule = random.choices([0,1], k=self.rules_len)
                # Inserting new rule to the new individual
                new_individual.append(new_rule)

            # Inserting new individual to initial population
            self.population.append(new_individual)


    @staticmethod
    def __clf_example(individual, example, diccionario):
        n_feat = len(diccionario)-1 # number of features. Last dictionary is the class dict
        # Keeping track of predicted class by each rule
        predicted_classes = []
        for rule in individual:
            # a rule is a binary list, but a feature is represented by multiple bits
            cum_idx = 0
            activation = 1
            # rule is activated if every feature matches
            for j in range(n_feat):
                # length of current feature
                len_feat = len(diccionario[j])
                # comparing feature encode with its corresponding rule

                feat_and = np.bitwise_and(example[cum_idx:cum_idx+len_feat], rule[cum_idx:cum_idx+len_feat])
                if sum(feat_and) == 0:
                    # if the rule does not match, the rule does not activate
                    activation = 0
                    break
                else:
                    # jumping to next feature
                    cum_idx += len_feat

            if activation: # saving predicted class by this rule
                predicted_classes.append(rule[-1])

        s = sum(predicted_classes)
        l = len(predicted_classes)/2
         # if the sum is larger than the half size of the list, it means the number of 1's is larger than 0's
        if s > l:
            return 1 # predicted class = 1
        # if the sum is smaller than the half size of the list, it means the number of 0's is larger than 1's
        elif s < l:
            return 0 # predicted class = 0
        elif s == l:
            return -1 # the number of rules predicting class 1 is the same than the number of rules predicting class 0 
        else:
            # zero rules activated
            return None


    
    def __fitness(self, individual, xdata, ydata, diccionario):
        n_examples, feat_size = xdata.shape # number of examples and length of each rule (after oneHotEncode)

        n_hits = 0 # number of training examples classified correctly by given individual

        for i in range(n_examples):
            # For each example, we check if its well classifed or not
            current_example = xdata[i,:]
            # Checking the predicted class for the current data example given an individual
            predicted_class = AlgoritmoGenetico.__clf_example(individual, current_example, diccionario)

            if predicted_class==1 and ydata[i]==1:
                n_hits += 1
            elif predicted_class==0 and ydata[i]==0:
                n_hits += 1

        # returning individual fitness, i.e., number of hits / total examples
        return n_hits/n_examples


    
    def __calculate_population_fitness(self, xdata, ydata, diccionario):
        fitness_list = []

        for individual in self.population:
            fitness_list.append(self.__fitness(individual, xdata, ydata, diccionario))

        return fitness_list



    def __elitism(self, fitness_list, n_elite_inds=None):
        elite = []

        if n_elite_inds:
            # Choose n_elite_inds (integer) best individuals
            elite_size = n_elite_inds
        else:
            # Choose self.elite_perc (percentage) best individuals
            elite_size = math.floor(self.elite_perc * len(self.population))

        population_copy = self.population.copy()
        fitness_list_copy = fitness_list.copy()

        n_extracted = 0
        while n_extracted < elite_size:
            # getting maximum fitness value
            best_fitness_val = max(fitness_list_copy)
            # getting index of individual with the highest fitness
            best_fitness_ind = fitness_list_copy.index(best_fitness_val)
            # getting individual (and removing it from population) with highest fitness
            best_ind = population_copy.pop(best_fitness_ind)
            # deleting its fitness from fitness_list
            fitness_list_copy.remove(best_fitness_val)
            # adding best individual to elite set
            elite.append(best_ind)
            n_extracted += 1
        
        return elite

    

    def __parent_selection(self, fitness_list, n_elite_inds=None):

        if n_elite_inds:
            # Choose n_elite_inds (integer) best individuals
            elite_size = n_elite_inds
        else:
            # Choose self.elite_perc (percentage) best individuals
            elite_size = math.floor(self.elite_perc * len(self.population))

        # total fitness
        s = sum(fitness_list)
        # list of weighted fitness of each individual
        weighted_probs = [f/s for f in fitness_list]
        ## np.random.choice(a, n, replace=False, p=None)
        #       a := list of items to choose
        #       n := number of items to choose
        #       replace: if True, items are chosen with replacement
        #       p: the probabilities associated with each entry in a. Otherwise, they are chosen uniformly.
        ##
        return np.random.choice(self.population, len(self.population)-elite_size, replace=True, p=weighted_probs)

    

    def __crossover(self, parents, inter_rule_cross=False):
        descendents = [] # list of descendets after crossovers

        n_inds = len(parents) # number of parents

        # looping over every pair of parents. If there is an odd number of parents, the last one is not crossed
        for i in range(0, n_inds, 2):
            if i != n_inds-1:
                # selecting two parents
                P1 = parents[i]
                P2 = parents[i+1]
                # flipping a coin with probability 'cross_prob' of getting 1 and with 1-'cross_prob' of getting 0
                cross = np.random.choice([1,0], p=[self.cross_prob, 1-self.cross_prob])
                if cross:
                    n_rules1 = len(P1) # number of rules of parent1
                    n_rules2 = len(P2) # number of rules of parent2
                    cross_rule1 = np.random.randint(0, n_rules1) # selecting a rule to cross of parent1
                    cross_rule2 = np.random.randint(0, n_rules2) # selecting a rule to cross of parent2
                    # Now we switch between inter-rule crossover or intra-rule crossover
                    # Selecting index of crossover
                    if inter_rule_cross:
                        cross_idx = 0 # inter-rule crossover
                    else:
                        cross_idx = np.random.randint(1, self.rules_len-1) # intra-rule crossover
                    # Crossing parents to get descendets
                    child1 = P1[:cross_rule1] + [P1[cross_rule1][:cross_idx] + P2[cross_rule2][cross_idx:]] + P2[cross_rule2+1:]
                    child2 = P2[:cross_rule2] + [P2[cross_rule2][:cross_idx] + P1[cross_rule1][cross_idx:]] + P1[cross_rule1+1:]
                    if len(child1) > self.max_rules:
                        child1 = child1[:self.max_rules]
                    if len(child2) > self.max_rules:
                        child2 = child2[:self.max_rules]
                    # adding new childs to the descendents list
                    descendents.append(child1)
                    descendents.append(child2)  
                else:
                    # if we obtain 0, crossover does not occur, so the parents get to next step directly
                    descendents.append(P1)
                    descendents.append(P2)
            else:
                # there is an odd number of parents, so the last one gets to next step directly
                descendents.append(parents[-1])

        return descendents



    def __mutation_bitflip(self, parents):
        for individual in parents:
            for rule in individual:
                for bit in rule:
                    # flipping a coin with probability 'bitflip_prob' of getting 1 and with 1-'cross_prob' of getting 0
                    bitflip = np.random.choice([1,0], p=[self.bitflip_prob, 1-self.bitflip_prob])
                    if bitflip: # flipping
                        if bit==1: bit=0
                        else: bit=1

        return parents


    
    def __mutation_add_delete_rule(self, parents):
        for individual in parents:
            add_del = np.random.choice([1,-1, 0], p=[self.add_rule_prob, self.add_rule_prob, 1-2*self.add_rule_prob])
            if add_del == 1 and len(individual) < self.max_rules: # Adding new rule
                # Asserting that new rule cannot have EVERY gene equal 0 or 1
                new_rule = []
                while sum(new_rule)==0 or sum(new_rule)==len(new_rule):
                    # Creating a new rule as a random binary string of size 'feat_size' + 1 (predicted class)
                    new_rule = random.choices([0,1], k=self.rules_len)
                individual.append(new_rule)
            
            elif add_del == -1: # Deleting a random rule from individual
                if len(individual) > 1: # we cannot delete a rule from an individual if it has an unique rule
                    del individual[np.random.randint(0, len(individual))]
            
        return parents

    

    def __mutation(self, parents):
        descendents = self.__mutation_bitflip(parents)
        descendents = self.__mutation_add_delete_rule(descendents)
        return descendents



    def __survivor_selection(self, parents, elite_inds):
        # elite individuals get to next generation directly
        return parents + elite_inds




    def entrenamiento(self,datosTrain,atributosDiscretos,diccionario):
        xdata = datosTrain[:,:-1] # all rows, all columns but last one
        ydata = datosTrain[:,-1]  # all rows, just last column (class)

        n_examples, feat_size = xdata.shape

        n_class_1 = sum(ydata)
        if n_class_1 > n_examples:
            self.max_prior = 1
        else:
            self.max_prior = 0

        # Now that we got the data, we can determine the rule length
        self.rules_len = feat_size+1 # + 1 due to class value

        # If the bitflip probability is not determined, it is set to 1/rule_length in order to
        #   flip 1 bit per rule (on average).
        if self.bitflip_prob is None:
            self.bitflip_prob = 1/self.rules_len

        # Creating initial population
        self.__init_population()

        for i in range(self.nepochs):
            print("Epoca ", i+1)
            fitness_list = self.__calculate_population_fitness(xdata, ydata, diccionario)
            # Elitism: best individuals get to next generation directly
            elite_inds = self.__elitism(fitness_list)
            # Parent selection: potential next generation is chosen randomly with replacement
            #                   proportional to fitness
            parents = self.__parent_selection(fitness_list)
            # Crossover: progenitors are transformed crossing two individuals, to generate new solutions
            descendets = self.__crossover(parents)
            # Mutation: progenitors are transformed making alterations on its genes
            descendets = self.__mutation(descendets)
            # Survivor selection: next generation is the union of progenitors + elite individuals
            survivors = self.__survivor_selection(descendets, elite_inds)
            # Next generation
            self.population = survivors

        # Getting best solution from final population. Just survives the best individual
        fitness_list = self.__calculate_population_fitness(xdata, ydata, diccionario)
        self.best_solution = self.__elitism(fitness_list, n_elite_inds=1)[0]



    def clasifica(self,datosTest,atributosDiscretos,diccionario):
        xdata = datosTest[:,:-1] # all rows, all columns but last one
        ydata = datosTest[:,-1]  # all rows, just last column (class)

        pred = []

        for example in xdata:
            predicted_class = AlgoritmoGenetico.__clf_example(self.best_solution, example, diccionario)


            if predicted_class is None:
                pred.append(-1)
            elif predicted_class == -1: # same number of votes
                pred.append(self.max_prior)
            else:
                pred.append(predicted_class)

        return np.asarray(pred, dtype="object")

        

