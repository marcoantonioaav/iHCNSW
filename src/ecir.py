import pickle
from threading import Thread

from data import Dataset
from knns.bkmeans_hnsw import BisectingKmeansHNSW
from knns.hnsw import HNSW

DB_NAME = 'sift1m'

def index_hnsw_and_hcnsw(dataset, seed):
    print(f"indexing with seed={seed}")

    hnsw = HNSW(m=20, ef_construction=100, ef=100, seed=seed)
    hnsw.insert(dataset.get_db_embeddings())
    with open(f'../data/hnsw_{DB_NAME}_seed{seed}.pkl', 'wb') as file:
        pickle.dump(hnsw, file, pickle.HIGHEST_PROTOCOL)

    hcnsw = BisectingKmeansHNSW(m=20, ef_construction=100, ef=100, max_clusters=1024, random_seed=seed)
    hcnsw.insert(dataset.get_db_embeddings())
    with open(f'../data/hcnsw_{DB_NAME}_seed{seed}.pkl', 'wb') as file:
        pickle.dump(hcnsw, file, pickle.HIGHEST_PROTOCOL)

if __name__=="__main__":
    dataset = Dataset()
    dataset.load_from_tfds(DB_NAME) 

    thread_count = 8
    pc_id = 0
    for iteration_id, first_seed in enumerate(range(0, 100, thread_count)):
        if iteration_id%3 == pc_id:
            seeds = range(first_seed, first_seed+thread_count)
            threads = []
            for seed in seeds:
                thread = Thread(target=index_hnsw_and_hcnsw, args=(dataset, seed, ))
                threads.append(thread)
                thread.start()
            for thread in threads:
                thread.join()