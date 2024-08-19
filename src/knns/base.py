from abc import ABC, abstractmethod
import numpy as np

class KNNSBase(ABC):
    @abstractmethod
    def insert(self, data):
        pass
    
    @abstractmethod
    def search(self, query, k=1):
        pass

    def get_distance(self, e1, e2):
        #return 1 - (np.dot(e1, e2)/(np.linalg.norm(e1)*np.linalg.norm(e2))) ## 1-(cossine similarity)
        return np.linalg.norm(e1-e2) ## Euclidean distance (L2 norm)