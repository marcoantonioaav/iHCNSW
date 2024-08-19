from math import floor, log
from time import time

from knns.base import KNNSBase
from random import uniform, seed
from ui import print_indexing_frame

seed(0)

class HNSW_Node:
    def __init__(self, embedding) -> None:
        self.embedding = embedding
        self.layer_neighbors = [[]]

    def set_neighbors(self, neighbors, layer):
        for lc in range(self.get_height(), layer+1):
            new_layer = []
            self.layer_neighbors.append(new_layer)
        self.layer_neighbors[layer] = neighbors

    def add_neighbor(self, neighbor, layer):
        self.layer_neighbors[layer].append(neighbor)

    def get_neighbors(self, layer):
        return self.layer_neighbors[layer]

    def get_height(self):
        return len(self.layer_neighbors)

class HNSW_Graph():
    def __init__(self) -> None:
        self.nodes = []
        self.enter_point_index = None
        self.height = 0

    def insert_data(self, data):
        for e in data:
            self.nodes.append(HNSW_Node(e))

    def remove_bidirectional_links(self, index, layer):
        node = self.get_node(index)
        for neighbor in node.get_neighbors(layer):
            self.get_node(neighbor).get_neighbors(layer).remove(index)
        neighbors = []
        node.set_neighbors(neighbors, layer)

    def set_bidirectional_links(self, index, neighbors, layer):
        node = self.get_node(index)
        node.set_neighbors(neighbors, layer)
        for neighbor in neighbors:
            if neighbor != index:
                node = self.get_node(neighbor)
                node.add_neighbor(index, layer)
        if layer+1 > self.height:
            self.height = layer+1

    def get_node(self, index) -> HNSW_Node:
        return self.nodes[index]
    

class HNSW(KNNSBase):
    def __init__(self, m=5, m_max=None, ef_construction=30, mL=None, ef=30) -> None:
        super().__init__()
        self.graph = HNSW_Graph()
        
        self.m = m
        if m_max == None:
            self.m_max = m
        else:
            self.m_max = m_max
        self.ef_construction = ef_construction
        if mL == None:
            self.mL = 1/log(m)
        else:
           self.mL = mL 

        self.ef = ef

        self.use_ui = True

    def insert(self, data):
        self.graph.insert_data(data)
        start = time()
        data_size = len(data)
        for i, e in enumerate(data):
            self.insert_element(i, e)
            if self.use_ui and i%100 == 0:
                end = time()
                print(f"{i}: {end-start}s")
                start = time()
                #print_indexing_frame(i, data_size)
        #if self.use_ui:
            #print_indexing_frame(data_size, data_size)

    def search(self, query, k=1):
        W = []
        ep = self.graph.enter_point_index
        L = self.graph.height-1
        for lc in range(L, 0, -1):
            W = self.search_layer(query, ep=ep, ef=1, lc=lc)
            ep = self.get_nearest(W)
        W = self.search_layer(query, ep=ep, ef=self.ef, lc=0)
        return self.get_k_nearest(W, k, return_distances=True)
            
    def insert_element(self, index, q):
        w = []
        ep = self.graph.enter_point_index
        L = self.graph.height-1
        l = floor(-log(uniform(0, 1))*self.mL)
        for lc in range(L, l, -1):
            w = self.search_layer(q, ep=ep, ef=1, lc=lc)
            ep = self.get_nearest(w)
        for lc in range(min(L, l), -1, -1): 
            w = self.search_layer(q, ep=ep, ef=self.ef_construction, lc=lc)
            neighbors = self.select_neighbors(q, w, self.m)
            self.graph.set_bidirectional_links(index, neighbors, lc)
            for e in neighbors:
                e_node = self.graph.get_node(e)
                e_conn = e_node.get_neighbors(lc)
                if len(e_conn) > self.m_max:
                    e_conn = [(e_c, self.get_distance(e_node.embedding, self.graph.get_node(e_c).embedding)) for e_c in e_conn]
                    e_new_conn = self.select_neighbors(e, e_conn, self.m_max)
                    self.graph.remove_bidirectional_links(e, lc)
                    self.graph.set_bidirectional_links(e, e_new_conn, lc)
            ep = w[0][0]
        if l > L:
            neighbors = []
            self.graph.get_node(index).set_neighbors(neighbors, l)
            self.graph.height = l+1
            self.graph.enter_point_index = index

    def search_layer(self, q, ep, ef, lc):
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
                if not e in v:
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
    
    def select_neighbors(self, q, C, M):
        return self.get_k_nearest(C, M)
    
    def get_k_nearest(self, node_indexes_and_distances, k, return_distances=False):
        results = []
        for index, distance in node_indexes_and_distances:
            if len(results) < k:
                results.append((index, distance))
                results.sort(key=lambda tup: tup[1])
            else:
                if distance < results[-1][1]:
                    results.remove(results[-1])
                    results.append((index, distance))
                    results.sort(key=lambda tup: tup[1])
        if not return_distances:
            results = [tup[0] for tup in results]
        return results
    
    def get_nearest(self, node_indexes_and_distances, return_distances=False):
        result = node_indexes_and_distances[0]
        for index, distance in node_indexes_and_distances[1:]:
            if distance < result[1]:
                result = (index, distance)
        if not return_distances:
            result = result[0]
        return result
    
    def get_furthest(self, node_indexes_and_distances):
        result = node_indexes_and_distances[0]
        for index, distance in node_indexes_and_distances[1:]:
            if distance > result[1]:
                result = (index, distance)
        return result