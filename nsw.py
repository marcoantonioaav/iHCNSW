from numpy import zeros
from random import choice
import util

class Node:
    def __init__(self, e=zeros(16)) -> None:
        self.e = e
        self.neighbors = []

    def search(self, query, comparisons=0):
        next_node = self
        for neighbor in self.neighbors:
            comparisons += 1
            if util.get_similarity(neighbor.e, query) > util.get_similarity(next_node.e, query):
                next_node = neighbor
        if next_node == self:
            return self, comparisons
        return next_node.search(query, comparisons)

    def insert(self, node):
        neighbor = self.search(node.e)[0]
        neighbor.neighbors.append(node)
        node.neighbors.append(neighbor)

class Nsw:
    def __init__(self, embeddings=[]) -> None:
        self.nodes = []
        self.insert_embeddings(embeddings)

    def search(self, query, k):
        results = []
        comparsions = 0
        for i in range(k):
            entry_point = choice(self.nodes)
            result, it_comparsions = entry_point.search(query)
            results.append((result.e, util.get_similarity(result.e, query)))
            comparsions += it_comparsions
        return sorted(results, key=lambda tup: tup[1], reverse=True), comparsions

    def insert(self, node):
        if len(self.nodes) > 0:
            entry_point = choice(self.nodes)
            entry_point.insert(node)
        self.nodes.append(node)

    def insert_embeddings(self, embeddings):
        for embedding in embeddings:
            self.insert(Node(e=embedding))