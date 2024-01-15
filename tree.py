import numpy as np
import util

def tree_search(query, tree):
    comparisons = 0
    node = tree
    while len(node.children) > 0:
        next_node = node.children[0]
        similarity = 0
        for child in node.children:
            node_similarity = util.get_similarity(query, child.e)
            comparisons += 1
            if node_similarity > similarity:
                similarity =  node_similarity
                next_node = child
        node = next_node
    return node.e, util.get_similarity(node.e, query), comparisons

class TreeNode:
    def __init__(self, e=np.zeros(16), children=[]) -> None:
        self.e = e
        self.children = children
        for child in children:
            self.e = self.e + child.e
        if len(children) > 0:
            self.e = self.e / len(children)

def generate_data_tree(data, height=30):
    nodes = []
    for line in data:
        nodes.append(TreeNode(e=line))
    for level in range(height-2, -1, -1):
        width = get_level_width(level, len(data), height)
        children_size = int(np.ceil(len(nodes)/width))
        new_nodes = []
        for i in range(width):
            if (i*children_size)+children_size > len(nodes):
                children = [nodes[-1]]
                #print(f"{width}[-1] -> {len(children)}")
            else:
                children = nodes[i*children_size:(i*children_size)+children_size]
                #print(f"{width}[{i*children_size}:{(i*children_size)+children_size}] -> {len(children)}")
            new_nodes.append(TreeNode(children=children))
        nodes = new_nodes
    return nodes[0]

def get_level_width(level, data_size, height):
    if level == 0:
        return 1
    if level == height-1:
        return data_size
    return data_size - int(np.ceil(data_size/(height))*(height-level))