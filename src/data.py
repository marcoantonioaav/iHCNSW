import numpy as np
import tensorflow_datasets as tfds

from knns.exhaustive import ExhaustiveKnn

class Dataset():
    def __init__(self) -> None:
        self.db = []
        self.test = []

    def load_from_tfds(self, name):
        self.test = list(tfds.load(name, split='test'))
        self.db = list(tfds.load(name, split='database'))

    def get_test_size(self):
        return len(self.test)

    def get_db_embeddings(self):
        return [i["embedding"] for i in self.db]

    def get_test_embedding(self, test_index):
        return self.test[test_index]["embedding"]

    def get_test_recall(self, test_index, search_result):
        k = len(search_result)
        gold_result = [i for i in self.test[test_index]["neighbors"]["index"][:k]]
        found = 0
        for i in range(k):
            if gold_result.count(search_result[i][0]) != 0:
                found += 1
        return found/k

def generate_embeddings(n, size=4):
    embeddings = []
    for i in range(n):
        embeddings.append(generate_embedding(size))
    return embeddings

def generate_embedding(size=4):
    return np.random.rand(size)

if __name__=="__main__":
    ds = Dataset()
    print("loading dataset...")
    ds.load_from_tfds('sift1m') 
    print("complete")
    k = 5
    print("getting embeddings...")
    query = ds.get_test_embedding(0)
    data = ds.get_db_embeddings()
    print("complete")
    print("indexing data...")
    knn = ExhaustiveKnn()
    knn.insert(data)
    print("complete")
    print("searching...")
    knn_results = knn.search(query, k)
    print("complete")
    print(knn_results)
    print("validating...")
    print(ds.get_test_recall(0, knn_results))
    print("complete")