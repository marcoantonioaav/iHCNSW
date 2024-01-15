import util

def knn(query, data, k):
    comparisons = 0
    results = []
    for line in data:
        line_similarity = util.get_similarity(query, line)
        if len(results) < k:
            results.append((line, line_similarity))
            results.sort(key=lambda tup: tup[1])
        else:
            comparisons += 1
            if line_similarity > results[0][1]:
                results.remove(results[0])
                results.append((line, line_similarity))
                results.sort(key=lambda tup: tup[1])
    return sorted(results, key=lambda tup: tup[1], reverse=True), comparisons