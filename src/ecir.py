import pickle
from sys import argv
from threading import Thread

from data import Dataset
from knns.bkmeans_hnsw import BisectingKmeansHNSW
from knns.exhaustive import ExhaustiveKnn
from knns.hnsw import HNSW

DB_NAME = 'sift1m'
BATCH_SIZE = 360
MAX_PCS = 3
EXPERIMENT_SIZE = 10000

def index_exhaustive(dataset, seed):
    print(f"indexing exhaustive with seed={seed}")

    exhaustive = ExhaustiveKnn()
    exhaustive.insert(dataset.get_db_embeddings())
    with open(f'../data/exhaustive_{DB_NAME}_seed{seed}.pkl', 'wb') as file:
        pickle.dump(exhaustive, file, pickle.HIGHEST_PROTOCOL)

def index_hnsw_and_hcnsw(dataset, seed):
    print(f"indexing hnsw with seed={seed}")

    hnsw = HNSW(m=20, ef_construction=100, ef=100, seed=seed)
    hnsw.insert(dataset.get_db_embeddings())
    with open(f'../data/hnsw_{DB_NAME}_seed{seed}.pkl', 'wb') as file:
        pickle.dump(hnsw, file, pickle.HIGHEST_PROTOCOL)

    print(f"indexing hcnsw with seed={seed}")

    hcnsw = BisectingKmeansHNSW(m=20, ef_construction=100, ef=100, max_clusters=1024, random_seed=seed)
    hcnsw.insert(dataset.get_db_embeddings())
    with open(f'../data/hcnsw_{DB_NAME}_seed{seed}.pkl', 'wb') as file:
        pickle.dump(hcnsw, file, pickle.HIGHEST_PROTOCOL)

if __name__=="__main__":
    pc_id = int(argv[1])
    thread_count = int(argv[2])
    type = argv[3]

    dataset = Dataset()
    dataset.load_from_tfds(DB_NAME) 

    for iteration_id, batch_start in enumerate(range(0, EXPERIMENT_SIZE, BATCH_SIZE)):
        if iteration_id%MAX_PCS == pc_id:
            for first_seed in range(batch_start, batch_start+BATCH_SIZE, thread_count):
                seeds = range(first_seed, first_seed+thread_count)               
                threads = []
                for seed in seeds:
                    if type == 'test':
                        thread = Thread(target=index_exhaustive, args=(dataset, seed, ))
                    else:
                        thread = Thread(target=index_hnsw_and_hcnsw, args=(dataset, seed, ))
                    threads.append(thread)
                    thread.start()
                for thread in threads:
                    thread.join()