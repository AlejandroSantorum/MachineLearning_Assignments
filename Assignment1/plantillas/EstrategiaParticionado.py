from abc import ABCMeta,abstractmethod


class Particion():

    # Esta clase mantiene la lista de �ndices de Train y Test para cada partici�n del conjunto de particiones
    def __init__(self):
        self.indicesTrain=[]
        self.indicesTest=[]

#####################################################################################################

class EstrategiaParticionado:

    # Clase abstracta
    __metaclass__ = ABCMeta

    # Atributos: deben rellenarse adecuadamente para cada estrategia concreta. Se pasan en el constructor

    @abstractmethod
    # TODO: esta funcion deben ser implementadas en cada estrategia concreta
    def creaParticiones(self,datos,seed=None):
        pass


#####################################################################################################

class ValidacionSimple(EstrategiaParticionado):

    # Crea particiones segun el metodo tradicional de division de los datos segun el porcentaje deseado y el n�mero de ejecuciones deseado
    # Devuelve una lista de particiones (clase Particion)
    # TODO: implementar
    def creaParticiones(self,datos,seed=None):
        random.seed(seed)
        pass


#####################################################################################################
class ValidacionCruzada(EstrategiaParticionado):

    # Crea particiones segun el metodo de validacion cruzada.
    # El conjunto de entrenamiento se crea con las nfolds-1 particiones y el de test con la particion restante
    # Esta funcion devuelve una lista de particiones (clase Particion)
    # TODO: implementar
    def creaParticiones(self,datos,seed=None):
        random.seed(seed)
        pass
