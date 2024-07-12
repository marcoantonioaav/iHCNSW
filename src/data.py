import numpy as np
import tensorflow_datasets as tfds
import os
from knns.exhaustive import ExhaustiveKnn

class Dataset():
    def __init__(self) -> None:
        self.db = []
        self.test_embeddings = []
        self.test_neighbors = []

    def load_from_tfds(self, name):
        if os.path.isfile(f"data/tfds_db_{name}.npy"):
            self.test_embeddings = np.load(f"data/tfds_test_embeddings_{name}.npy").tolist()
            self.test_neighbors = np.load(f"data/tfds_test_neighbors_{name}.npy").tolist()
            self.db = np.load(f"data/tfds_db_{name}.npy").tolist()
        else:
            tfds_test = list(tfds.load(name, split='test'))
            tfds_db = list(tfds.load(name, split='database'))
            self.test_embeddings = [i["embedding"] for i in tfds_test]
            self.test_neighbors = [i["neighbors"]["index"] for i in tfds_test]
            self.db = [i["embedding"] for i in tfds_db]
            np.save(f"data/tfds_test_embeddings_{name}.npy", self.test_embeddings)
            np.save(f"data/tfds_test_neighbors_{name}.npy", self.test_neighbors)
            np.save(f"data/tfds_db_{name}.npy", self.db)

    def get_test_size(self):
        return len(self.test_embeddings)

    def get_db_embeddings(self):
        return self.db

    def get_test_embedding(self, test_index):
        return self.test_embeddings[test_index]

    def get_test_recall(self, test_index, search_result):
        k = len(search_result)
        gold_result = [i for i in self.test_neighbors[test_index][:k]]
        found = 0
        for i in range(k):
            if gold_result.count(search_result[i][0]) != 0:
                found += 1
        return found/k

def generate_embeddings(n, size=128):
    embeddings = []
    for i in range(n):
        embeddings.append(generate_embedding(size))
    return embeddings

def generate_embedding(size=128):
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