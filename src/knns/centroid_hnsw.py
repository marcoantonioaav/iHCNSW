from knns.hnsw import HNSW
from sklearn.cluster import KMeans
from math import floor, log
from random import uniform, shuffle

from ui import print_indexing_frame

class CentroidHNSW(HNSW):
    def __init__(self, m=5, m_max=None, ef_construction=30, mL=None, ef=30, n_clusters=2) -> None:
        super().__init__(m, m_max, ef_construction, mL, ef)
        self.n_clusters = n_clusters
        self.centroids = []

    def insert(self, data):
        self.graph.insert_data(data)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)
        kmeans.fit(data)
        self.centroids = kmeans.cluster_centers_

        data_distances = [(i, min([self.get_distance(d, c) for c in self.centroids])) for i, d in enumerate(data)]
        data_distances.sort(key=lambda tup: tup[1], reverse=True)

        layers = [floor(-log(uniform(0, 1))*self.mL) for i in range(len(data))]
        layers.sort()

        indexes_and_layers = [(data_distances[i][0], layers[i]) for i in range(len(data))]
        shuffle(indexes_and_layers) 

        total_inserted = 0
        for index, layer in indexes_and_layers:
            self.insert_element(index, data[index], layer)
            total_inserted +=1
            if self.use_ui and total_inserted%100 == 0:
                print_indexing_frame(total_inserted, len(data))
        if self.use_ui:
            print_indexing_frame(len(data), len(data))

    def insert_element(self, index, q, l):
        w = []
        ep = self.graph.enter_point_index
        L = self.graph.height-1 
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
        
