import numpy as np
from sklearn.decomposition import PCA
import tensorflow_datasets as tfds
import os
from knns.exhaustive import ExhaustiveKnn

np.random.seed(0)

class Dataset():
    def __init__(self) -> None:
        self.db = []
        self.test_embeddings = []
        self.test_neighbors = []
        self.pca_db = []

    def load_from_tfds(self, name):
        if os.path.isfile(f"../data/tfds_db_{name}.npy"):
            self.test_embeddings = np.load(f"../data/tfds_test_embeddings_{name}.npy").tolist()
            self.test_embeddings = [np.array(i) for i in self.test_embeddings]
            self.test_neighbors = np.load(f"../data/tfds_test_neighbors_{name}.npy").tolist()
            self.db = np.load(f"../data/tfds_db_{name}.npy").tolist()
            self.db = [np.array(i) for i in self.db]
            self.pca_db = np.load(f"../data/tfds_db_{name}2d.npy").tolist()
            self.pca_db = [np.array(i) for i in self.pca_db]
        else:
            tfds_test = list(tfds.load(name, split='test'))
            tfds_db = list(tfds.load(name, split='database'))
            self.test_embeddings = [np.array(i["embedding"]) for i in tfds_test]
            self.test_neighbors = [np.array(i["neighbors"]["index"]) for i in tfds_test]
            self.db = [np.array(i["embedding"]) for i in tfds_db]
            np.save(f"../data/tfds_test_embeddings_{name}.npy", self.test_embeddings)
            np.save(f"../data/tfds_test_neighbors_{name}.npy", self.test_neighbors)
            np.save(f"../data/tfds_db_{name}.npy", self.db)

            pca_embeddings = np.concatenate([self.db, self.test_embeddings])
            pca = PCA(n_components=2)
            pca.fit(pca_embeddings)
            transformed = pca.transform(pca_embeddings)
            t_db = transformed[0:1000000]
            np.save(f"../data/tfds_db_{name}2d.npy", t_db)

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
            if search_result[i][0] in gold_result:
                found += 1
        return found/k

def generate_embeddings(n, size=128):
    embeddings = []
    for i in range(n):
        embeddings.append(generate_embedding(size))
    return embeddings

def generate_embedding(size=128):
    return np.array(np.random.rand(size))

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