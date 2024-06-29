from knns.exhaustive import ExhaustiveKnn
import data

def test_w_generated_embeddings(k=5, n=200):
    query = data.generate_embedding()
    data = data.generate_embeddings(n)

    knn = ExhaustiveKnn()
    knn.insert(data)
    gold_results = knn.search(query, k)

    for gold_result in gold_results:
        print(gold_result)

if __name__=="__main__":
    ds = data.Dataset()
    ds.load_from_tfds('sift1m') 
    
    data = ds.get_db_embeddings()
    knn = ExhaustiveKnn()
    knn.insert(data)

    k = 5
    test_size = ds.get_test_size()
    print(f"testing {test_size} query embeddings")
    for i in range(test_size):
        query = ds.get_test_embedding(i)
        knn_results = knn.search(query, k)
        recall = ds.get_test_recall(i, knn_results)
        print(f"test {i}: recall = {recall}")
    
