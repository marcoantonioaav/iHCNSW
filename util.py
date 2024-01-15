import numpy as np

def get_similarity(e1, e2):
    return np.dot(e1, e2)/(np.linalg.norm(e1)*np.linalg.norm(e2)) ## cossine similarity

def generate_embeddings(n, size=16):
    embeddings = []
    for i in range(n):
        embeddings.append(generate_embedding(size))
    return embeddings

def generate_embedding(size=16):
    return np.random.rand(size)

def print_results(query, results, comparsions, data, algorithm):
    print(f"{algorithm}\nQuery: {query}\nComparsions: {comparsions}\n")
    for i, result in enumerate(results):
        print(f"Result {i}: {result[0]}\nSimilarity: {result[1]}\nRanking: {rank_result(result[0], query, data)}\n")
    print("------------------------------------------------------")

def rank_result(result, query, data):
    ranked_data = rank_data(query, data)
    rank = 0
    while not np.array_equal(ranked_data[rank][1], result):
        rank += 1
    return rank

def rank_data(query, data):
    dict = {}
    for line in data:
        dict[get_similarity(line, query)] = line
    return sorted(dict.items(), reverse=True)