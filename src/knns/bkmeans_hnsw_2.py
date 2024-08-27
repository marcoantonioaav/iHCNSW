from math import inf
from time import time
from sklearn.cluster import BisectingKMeans
import numpy as np
from random import sample, shuffle

from knns.hnsw import HNSW

class BisectingKmeansHNSW2(HNSW):
    def __init__(self, m=5, m_max0='auto', ef_construction=30, mL='auto', ef=30, max_clusters=1024, max_iterations=inf) -> None:
        super().__init__(m, m_max0, ef_construction, mL, ef)
        self.max_clusters = max_clusters
        self.cluster_labels = []
        self.max_iterations = max_iterations

    def insert(self, data):
        self.graph.insert_data(data)

        candidates = list(range(len(data)))
        indexes_and_layers = []
        layer = 0
        layer_increment_size = 1
        while layer_increment_size <= self.max_clusters and len(candidates) > 0:
            kmeans = BisectingKMeans(n_clusters=layer_increment_size, bisecting_strategy='largest_cluster', random_state=0)
            kmeans.fit(data)
            self.cluster_labels.append(kmeans.labels_)
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

    def search(self, query, k=1):
        previous_result = []
        result = self.search_iteration(query, k)
        pruned = []
        min_quantization = self.min_quantization(result)
        self.i = 0
        while self.i <= self.max_iterations and min_quantization[1] > 0 and not self.equal_lists(result, previous_result):
            previous_result = result
            if not min_quantization in pruned:
                pruned.append(min_quantization)
            iteration_result = self.search_iteration(query, k, prune=pruned)
            result = self.get_k_nearest(self.merge_results(iteration_result, previous_result), k, return_distances=True)
            self.i += 1
        return result

    def merge_results(self, l1, l2):
        l3 = [e for e in l1]
        for e in l2:
            if not e in l3:
                l3.append(e)
        return l3

    def equal_lists(self, l1, l2):
        for e in l1:
            if not e in l2:
                return False
        return len(l1)==len(l2)
    
    def min_quantization(self, result):
        quantization = (0, 0)
        for deepness, labels in enumerate(self.cluster_labels):
            label = labels[result[0][0]]
            for index, distance in result:
                if labels[index] != label:
                    return quantization
            quantization = (label, deepness)
        return quantization

    def search_iteration(self, query, k, prune=[]):
        W = []
        ep = self.graph.enter_point_index
        L = self.graph.height-1
        for lc in range(L, 0, -1):
            W = self.search_layer(query, ep=ep, ef=1, lc=lc, prune=prune)
            ep = self.get_nearest(W)
        W = self.search_layer(query, ep=ep, ef=self.ef, lc=0, prune=prune)
        return self.get_k_nearest(W, k, return_distances=True)
    
    def search_layer(self, q, ep, ef, lc, prune=[]):
        ep_distance = self.get_distance(q, self.graph.get_node(ep).embedding)
        ep_and_distance = (ep, ep_distance)
        v = [ep]
        C = [ep_and_distance]
        W = [ep_and_distance]
        while len(C) > 0:
            c, c_distance = self.get_nearest(C, return_distances=True)
            C.remove((c, c_distance))
            f, f_distance = self.get_furthest(W)
            if c_distance > f_distance:
                break
            for e in self.graph.get_node(c).get_neighbors(lc):
                if not e in v and not self.is_pruned(e, prune):
                    v.append(e) 
                    e_distance = self.get_distance(q, self.graph.get_node(e).embedding)
                    if e_distance < f_distance or len(W) < ef:
                        e_and_distance = (e, e_distance)
                        C.append(e_and_distance)
                        W.append(e_and_distance)
                        if len(W) > ef:
                           W.remove((f, f_distance))
                        f, f_distance = self.get_furthest(W)
        return W 
        
    def is_pruned(self, e, prune):
        for p in prune:
            if self.cluster_labels[p[1]][e] == p[0]:
                return True
        return False