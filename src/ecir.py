import os
import pickle
from sys import argv
from threading import Event, Thread

import numpy as np

from knns.hcnsw import HCNSW
from knns.exhaustive import ExhaustiveKnn
from knns.hnsw import HNSW

DB_NAME = 'sift1m'
BATCH_SIZE = 360
MAX_PCS = 3
EXPERIMENT_SIZE = 10000

def load_db_embeddings(dataset_name):
    if os.path.isfile(f"../data/tfds_db_{dataset_name}.npy"):
        db = np.load(f"../data/tfds_db_{dataset_name}.npy").tolist()
        db = [np.array(i) for i in db]
    else:
        import tensorflow_datasets as tfds
        tfds_db = list(tfds.load(dataset_name, split='database'))
        db = [np.array(i["embedding"]) for i in tfds_db]
        del tfds_db
        np.save(f"../data/tfds_db_{dataset_name}.npy", db)
    return db

def index_exhaustive(seed, loading_event):
    db_embeddings = load_db_embeddings(DB_NAME)
    loading_event.set()

    print(f"indexing exhaustive with seed={seed}")

    exhaustive = ExhaustiveKnn()
    exhaustive.insert(db_embeddings)
    with open(f'../data/exhaustive_{DB_NAME}_seed{seed}.pkl', 'wb') as file:
        pickle.dump(exhaustive, file, pickle.HIGHEST_PROTOCOL)

def index_hnsw_and_hcnsw(seed, loading_event):
    db_embeddings = load_db_embeddings(DB_NAME)
    loading_event.set()

    print(f"indexing hnsw with seed={seed}")

    hnsw = HNSW(m=20, ef_construction=100, ef=100, random_seed=seed)
    hnsw.insert(db_embeddings)
    with open(f'../data/hnsw_{DB_NAME}_seed{seed}.pkl', 'wb') as file:
        pickle.dump(hnsw, file, pickle.HIGHEST_PROTOCOL)

    print(f"indexing hcnsw with seed={seed}")

    hcnsw = HCNSW(m=20, ef_construction=100, ef=100, max_clusters=1024, random_seed=seed)
    hcnsw.insert(db_embeddings)
    with open(f'../data/hcnsw_{DB_NAME}_seed{seed}.pkl', 'wb') as file:
        pickle.dump(hcnsw, file, pickle.HIGHEST_PROTOCOL)

if __name__=="__main__":
    pc_id = int(argv[1])
    thread_count = int(argv[2])
    type = argv[3]
    for iteration_id, batch_start in enumerate(range(0, EXPERIMENT_SIZE, BATCH_SIZE)):
        if iteration_id%MAX_PCS == pc_id:
            for first_seed in range(batch_start, batch_start+BATCH_SIZE, thread_count):
                seeds = range(first_seed, first_seed+thread_count)               
                threads = []
                for seed in seeds:
                    loading_event = Event()
                    if type == 'test':
                        thread = Thread(target=index_exhaustive, args=(seed, loading_event, ))
                    else:
                        thread = Thread(target=index_hnsw_and_hcnsw, args=(seed, loading_event, ))
                    threads.append(thread)
                    thread.start()
                    loading_event.wait()
                for thread in threads:
                    thread.join()