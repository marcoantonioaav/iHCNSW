from knns.hnsw import HNSW
from sklearn.cluster import KMeans


class IterativeHNSW(HNSW):
    def __init__(self, m=5, m_max0='auto', ef_construction=30, mL='auto', ef=30, n_clusters=1024, n_probes=8) -> None:
        super().__init__(m, m_max0, ef_construction, mL, ef)
        self.n_clusters = n_clusters
        self.n_probes = n_probes
        self.centroids = []
        self.labels = []

    def insert(self, data):
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)
        kmeans.fit(data)
        self.centroids = kmeans.cluster_centers_
        self.labels = kmeans.labels_
        return super().insert(data)
    
    def search(self, query, k=1):
        previous_result = []
        result = self.search_iteration(query, k)
        explored_clusters = self.add_clusters_to_list(result, [])
        self.i = 0
        while len(explored_clusters) < self.n_probes and not self.equal_lists(result, previous_result):
            previous_result = result
            iteration_result = self.search_iteration(query, k, prune=explored_clusters)
            result = self.get_k_nearest(self.merge_results(iteration_result, previous_result), k, return_distances=True)
            explored_clusters = self.add_clusters_to_list(result, explored_clusters)
            self.i += 1
        return result
    
    def add_clusters_to_list(self, result, l):
        for index, distance in result:
            label = self.labels[index]
            if not label in l:
                l.append(label)
        return l
    
    def equal_lists(self, l1, l2):
        for e in l1:
            if not e in l2:
                return False
        return len(l1)==len(l2)
    
    def merge_results(self, l1, l2):
        l3 = [e for e in l1]
        for e in l2:
            if not e in l3:
                l3.append(e)
        return l3
    
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
        return self.labels[e] in prune