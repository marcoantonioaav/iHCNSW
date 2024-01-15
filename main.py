import numpy as np
from knn import knn
from nsw import Nsw
import util
import tree

if __name__=="__main__":
    n = 200
    k = 1
    query = util.generate_embedding()
    data = util.generate_embeddings(n)

    results, comparsions = knn(query, data, k)
    util.print_results(query, results, comparsions, data, "KNN")

    #data_tree = tree.generate_data_tree(data, int(np.log(n)))
    #result, similarity, comparsions = tree.tree_search(query, data_tree)
    #ranking = util.rank_result(result, query, data)
    #print(f"Tree search\nQuery: {query}\nComparsions: {comparsions}\nResult: {result}\nSimilarity: {similarity}\nRanking: {ranking}\n")

    nsw = Nsw(data)
    results, comparsions = nsw.search(query, k)
    util.print_results(query, results, comparsions, data, "NSW")
