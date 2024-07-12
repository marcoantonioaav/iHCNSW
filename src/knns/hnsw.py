from math import floor, log
from time import time
from knns.base import KNNSBase
from random import randint, uniform

class HNSW_Node:
    def __init__(self, index, embedding, neighbors) -> None:
        self.index = index
        self.embedding = embedding
        self.neighbors = neighbors

class HNSW_Graph():
    def __init__(self) -> None:
        self.layers = [[]]
        self.enter_point_index = None

    def insert_node(self, node, layer) -> None:
        self.layers[layer].append(node)
        for index in node.neighbors:
            neighbor = self.get_node(index, layer)
            neighbor.neighbors.append(node.index)

    def set_neighbors(self, index, layer, neighbors):
        node = self.get_node(index, layer)
        for neighbor_index in node.neighbors:
            neighbor = self.get_node(neighbor_index, layer)
            neighbor.neighbors.remove(index)
        node.neighbors = neighbors
        for neighbor_index in node.neighbors:
            neighbor = self.get_node(neighbor_index, layer)
            neighbor.neighbors.append(index)

    def get_node(self, index, layer) -> HNSW_Node:
        return next(node for node in self.layers[layer] if node.index == index)

    def get_height(self) -> int:
        return len(self.layers)
    

class HNSW(KNNSBase):
    def __init__(self, m=5, m_max=5, ef_construction=30, mL=1, ef=30) -> None:
        super().__init__()
        self.graph = HNSW_Graph()
        self.plain_data = []
        
        self.m = m
        self.m_max = m_max
        self.ef_construction = ef_construction
        self.mL = mL

        self.ef = ef

    def insert(self, data):
        self.graph.layers[0].append(HNSW_Node(0, data[0], []))
        self.graph.enter_point_index = 0
        start = time()
        for i, e in enumerate(data):
            if i%100 == 0:
                end = time()
                print(f"{i}: {end-start}s")
                start = time()
                
            self.plain_data.append(e)
            if i!=0:
                self.insert_element(i, e)

    def search(self, query, k=1):
        W = []
        ep = self.graph.enter_point_index
        L = self.graph.get_height()-1
        for lc in range(L, 1):
            W = self.search_layer(query, ep=ep, ef=1, lc=lc)
            ep = self.get_nearest(query, W)
        W = self.search_layer(query, ep=ep, ef=self.ef, lc=0)
        return self.get_k_nearest_result(query, W, k)
    
    def get_k_nearest_result(self, query, node_indexes, k=1):
        results = []
        for index in node_indexes:
            embedding = self.plain_data[index]
            distance = self.get_distance(query, embedding)
            if len(results) < k:
                results.append((index, distance))
                results.sort(key=lambda tup: tup[1])
            else:
                if distance < results[-1][1]:
                    results.remove(results[-1])
                    results.append((index, distance))
                    results.sort(key=lambda tup: tup[1])
        return results
            
    def insert_element(self, index, q):
        w = []
        ep = self.graph.enter_point_index
        L = self.graph.get_height()-1
        l = floor(-log(uniform(0, 1))*self.mL)
        for lc in range(L, l+1, -1):
            w = self.search_layer(q, ep=ep, ef=1, lc=lc)
            ep = self.get_nearest(q, w)
        for lc in range(min(L, l), -1, -1):
            w = self.search_layer(q, ep=ep, ef=self.ef_construction, lc=lc)
            neighbors = self.select_neighbors(q, w, self.m)
            self.graph.insert_node(HNSW_Node(index, q, neighbors), lc)
            for e in neighbors:
                e_conn =  self.graph.get_node(e, lc).neighbors
                if len(e_conn) > self.m_max:
                    e_new_conn = self.select_neighbors(e, e_conn, self.m_max)
                    self.graph.set_neighbors(e, lc, e_new_conn)
            ep = w[0]
        if l > L:
            self.graph.enter_point_index = index

    def search_layer(self, q, ep, ef, lc):
        v = [ep]
        C = [ep]
        W = [ep]
        while len(C) > 0:
            c = self.get_nearest(q, C)
            C.remove(c)
            f = self.get_furthest(q, W)
            c_node = self.graph.get_node(c, lc)
            f_embedding = self.graph.get_node(f, lc).embedding
            if self.get_distance(c_node.embedding, q) > self.get_distance(f_embedding, q):
                break
            for e in c_node.neighbors:
                if v.count(e) == 0:
                    v.append(e)
                    f = self.get_furthest(q, W)
                    e_node = self.graph.get_node(e, lc)
                    f_embedding = self.graph.get_node(f, lc).embedding
                    if self.get_distance(e_node.embedding, q) < self.get_distance(f_embedding, q) or len(W) < ef:
                        C.append(e)
                        W.append(e)
                        if len(W) > ef:
                           W.remove(self.get_furthest(q, W))
        return W 
    
    def select_neighbors(self, q, C, M):
        return self.get_nearest(q, C, M)
    
    def get_nearest(self, embedding, node_indexes, k=1):
        distances = []
        for index in node_indexes:
            distances.append((index, self.get_distance(self.plain_data[index], embedding)))
        distances.sort(key=lambda tup: tup[1])
        if k == 1:
            return distances[0][0]
        return [t[0] for t in distances][:k]
    
    def get_furthest(self, embedding, node_indexes, k=1):
        distances = []
        for index in node_indexes:
            node = self.graph.get_node(index, 0)
            distances.append((index, self.get_distance(self.plain_data[index], embedding)))
        distances.sort(key=lambda tup: tup[1], reverse=True)
        if k == 1:
            return distances[0][0]
        return [t[0] for t in distances][:k]