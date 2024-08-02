from knns.exhaustive import ExhaustiveKnn
import data
from knns.hnsw import HNSW
from knns.ivf import IVF
from time import time
import cProfile
import pstats

def test_ann_w_generated_embeddings(ann, embedding_size=128, db_size=5000, test_size=5, k=10):
    db_embeddings = data.generate_embeddings(db_size, size=embedding_size)
    
    knn = ExhaustiveKnn()
    knn.insert(db_embeddings)
    
    print(f"Test: algorithm={ann.__class__.__name__} dataset=generated(random) k={k}")
    start = time()
    ann.insert(db_embeddings)
    end = time()
    print(f"time(index)={round(end-start, 4)}")

    print(f"testing {test_size} query embeddings")
    tests = []
    for test_index in range(test_size):
        query = data.generate_embedding(size=embedding_size)
        gold_results = knn.search(query, k)

        start = time()
        ann_results = ann.search(query, k)
        end = time()

        found = 0
        for i in range(k):
            if gold_results.count(ann_results[i]) != 0:
                found += 1
        recall = found/k
        tests.append((recall, end-start))
        print(f"test {test_index}: recall={recall} time={round(end-start, 4)}")
    return tests

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
    for test_index in range(test_size):
        query = dataset.get_test_embedding(test_index)
        start = time()
        ann_results = ann.search(query, k)
        end = time()
        recall = dataset.get_test_recall(test_index, ann_results)
        print(f"test {test_index}: recall={recall} time={round(end-start, 4)}")

def profile_search(ann, embedding_size=128, db_size=200, k=10):
    db_embeddings = data.generate_embeddings(db_size, size=embedding_size)
    
    knn = ExhaustiveKnn()
    knn.insert(db_embeddings)

    ann.insert(db_embeddings)

    query = data.generate_embedding(size=embedding_size)
    gold_results = knn.search(query, k)

    with cProfile.Profile() as profile:
        ann_results = ann.search(query, k)

    found = 0
    for i in range(k):
        if gold_results.count(ann_results[i]) != 0:
            found += 1
    recall = found/k
    print(f"recall={recall}")

    stats = pstats.Stats(profile)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()


if __name__=="__main__":
    #with cProfile.Profile() as profile:
        #test_ann(ExhaustiveKnn())
        test_ann_w_generated_embeddings(IVF(n_buckets=6, n_probes=3))
        #test_ann(IVF(n_buckets=15, n_probes=3))
        #test_ann_w_generated_embeddings(HNSW(m=20, m_max=50, ef_construction=5, mL=0))
        #test_ann_w_generated_embeddings(HNSW())# 468126 get_distance
        #test_ann_w_generated_embeddings(HNSW(m=25, m_max=30, ef_construction=70, ef=100))
        test_ann(HNSW())
        #test_ann(HNSW(ef_construction=5))
        test_ann_w_generated_embeddings(ExhaustiveKnn()) #1200 get_distance

    #stats = pstats.Stats(profile)
    #stats.sort_stats(pstats.SortKey.TIME)
    #stats.print_stats()

    #profile_search(HNSW(m=25, m_max=30, ef_construction=70, ef=100))
    #profile_search(HNSW()) # 3491 search (55948 function calls in 0.032 seconds) / 512249 insert
    #profile_search(ExhaustiveKnn()) # 200 search / 0 insert