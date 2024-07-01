from knns.base import KNNSBase

class ExhaustiveKnn(KNNSBase):
    def __init__(self) -> None:
        super().__init__()
        self.data = []

    def insert(self, data):
        for e in data:
            self.data.append(e)
    
    def search(self, query, k=1):
        results = []
        for index, embedding in enumerate(self.data):
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
    