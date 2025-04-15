from time import time
from sklearn.cluster import BisectingKMeans
import numpy as np
from random import sample, shuffle, seed

from knns.hnsw import HNSW

class HCNSW(HNSW):
    def __init__(self, m=5, m_max0='auto', ef_construction=30, mL='auto', ef=30, max_clusters=1024, random_seed=0) -> None:
        super().__init__(m, m_max0, ef_construction, mL, ef, random_seed)
        self.max_clusters = max_clusters
        self.seed = random_seed
        seed(self.seed)

    def insert(self, data):
        self.graph.insert_data(data)

        candidates = list(range(len(data)))
        indexes_and_layers = []
        layer = 0
        layer_increment_size = 1
        while layer_increment_size <= self.max_clusters and len(candidates) > 0:
            kmeans = BisectingKMeans(n_clusters=layer_increment_size, bisecting_strategy='largest_cluster', random_state=self.seed)
            kmeans.fit(data)
            centroids = kmeans.cluster_centers_
            for centroid in centroids:
                candidate_distances = [(i, self.get_distance(data[i], centroid)) for i in candidates]
                best_candidate = self.get_nearest(candidate_distances)
                indexes_and_layers.append((best_candidate, layer))
                candidates.remove(best_candidate)
            print(f"layers selected: {len(data) - len(candidates)}")
            layer += 1
            layer_increment_size = min(self.m*layer_increment_size, len(candidates))
        
        while len(candidates) > 0:
            layer_candidates = sample(candidates, layer_increment_size)
            for layer_candidate in layer_candidates:
                indexes_and_layers.append((layer_candidate, layer))
                candidates.remove(layer_candidate)
            layer += 1
            print(f"layers selected: {len(data) - len(candidates)}")
            layer_increment_size = min(self.m*layer_increment_size, len(candidates))
        indexes_and_layers = [(i, (layer-1)-l) for i, l in indexes_and_layers]
        shuffle(indexes_and_layers) 

        start = time()
        for i, (index, layer) in enumerate(indexes_and_layers):
            self.insert_element(index, data[index], layer)
            if self.use_ui and i%100 == 0:
                end = time()
                print(f"{i}: {end-start}s")
                start = time()

    def insert_element(self, index, q, l):
        w = []
        ep = self.graph.enter_point_index
        L = self.graph.height-1
        for lc in range(L, l, -1):
            w = self.search_layer(q, ep=ep, ef=1, lc=lc)
            ep = self.get_nearest(w)
        for lc in range(min(L, l), -1, -1): 
            w = self.search_layer(q, ep=ep, ef=self.ef_construction, lc=lc)
            neighbors = self.select_neighbors_simple(q, w, self.m, lc)
            self.graph.set_bidirectional_links(index, neighbors, lc)
            for e in neighbors:
                e_node = self.graph.get_node(e)
                e_conn = e_node.get_neighbors(lc)
                if len(e_conn) > self.get_m_max(lc):
                    e_conn = [(e_c, self.get_distance(e_node.embedding, self.graph.get_node(e_c).embedding)) for e_c in e_conn]
                    e_new_conn = self.select_neighbors_simple(e, e_conn, self.get_m_max(lc), lc)
                    e_node.set_neighbors(e_new_conn, lc)
            ep = w[0][0]
        if l > L:
            neighbors = []
            self.graph.get_node(index).set_neighbors(neighbors, l)
            self.graph.height = l+1
            self.graph.enter_point_index = index
