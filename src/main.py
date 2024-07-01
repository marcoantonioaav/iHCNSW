from knns.exhaustive import ExhaustiveKnn
import data
from knns.ivf import IVF
from time import time

def test_ann_w_generated_embeddings(ann, n=200, k=10):
    db_embeddings = data.generate_embeddings(n)
    query = data.generate_embedding()

    knn = ExhaustiveKnn()
    knn.insert(db_embeddings)
    gold_results = knn.search(query, k)

    ann.insert(db_embeddings)
    ann_results = ann.search(query, k)

    found = 0
    for i in range(k):
        if gold_results.count(ann_results[i]) != 0:
            found += 1
    print(f"recall = {found/k}")

def test_ann(ann, tfds_name='sift1m', k=10):
    dataset = data.Dataset()
    dataset.load_from_tfds('sift1m') 
    db_embeddings = dataset.get_db_embeddings()

    print(f"Test: algorithm={ann.__class__.__name__} dataset={tfds_name} k={k}")
    start = time()
    ann.insert(db_embeddings)
    end = time()
    print(f"time(index)={round(end-start, 4)}")

    test_size = dataset.get_test_size()
    print(f"testing {test_size} query embeddings")
    for i in range(test_size):
        query = dataset.get_test_embedding(i)
        start = time()
        ann_results = ann.search(query, k)
        end = time()
        recall = dataset.get_test_recall(i, ann_results)
        print(f"test {i}: recall={recall} time={round(end-start, 4)}")

if __name__=="__main__":
    #test_ann(ExhaustiveKnn())
    #test_ann_w_generated_embeddings(IVF(n_buckets=6, n_probes=3))
    test_ann(IVF(n_buckets=15, n_probes=3))
    
