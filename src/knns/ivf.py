from knns.base import KNNSBase
from sklearn.cluster import KMeans

class IVF(KNNSBase):
    def __init__(self, n_buckets=4, n_probes=1) -> None:
        super().__init__()
        self.buckets = []
        for i in range(n_buckets):
            self.buckets.append([])
        self.centroids = [None]*n_buckets
        self.n_probes = n_probes

    def insert(self, data):
        model = KMeans(n_clusters=len(self.buckets))
        labels = model.fit_predict(data)
        for i, e in enumerate(data):
            self.buckets[labels[i]].append((i, e))            
        for i, bucket in enumerate(self.buckets):
            self.centroids[i] = [c for c in model.cluster_centers_[i]]
    
    def search(self, query, k=1):
        probes = list(range(0, len(self.buckets)))
        sorted_probes = sorted(probes, key=lambda i: self.get_distance(query, self.centroids[i]))
        domain = []
        for probe in sorted_probes[:self.n_probes]:
            domain += self.buckets[probe]
               
        results = []
        for index, embedding in domain:
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