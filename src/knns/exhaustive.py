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
        for index, line in enumerate(self.data):
            line_similarity = self.get_distance(query, line)
            if len(results) < k:
                results.append((index, line_similarity))
                results.sort(key=lambda tup: tup[1], reverse=True)
            else:
                if line_similarity < results[0][1]:
                    results.remove(results[0])
                    results.append((index, line_similarity))
                    results.sort(key=lambda tup: tup[1], reverse=True)
        return sorted(results, key=lambda tup: tup[1])

#def knn(query, data, k):
#    comparisons = 0
#    results = []
#    for line in data:
#        line_similarity = util.get_similarity(query.e, line.e)
#        if len(results) < k:
#            results.append((line, line_similarity))
#            results.sort(key=lambda tup: tup[1])
#        else:
#            comparisons += 1
#            if line_similarity > results[0][1]:
#                results.remove(results[0])
#               results.append((line, line_similarity))
#                results.sort(key=lambda tup: tup[1])
#    return sorted(results, key=lambda tup: tup[1], reverse=True), comparisons