"""
Pawlak Iga, Michalski Marcin
PSZT Projekt 2
Alcohol Consumption Prediction
Algorithm C4.5
"""


import ID3
import numpy as np
import pickle

"""
Class implementing the C45 tree pruning algorithm
Fields:     id3 - ID3 object, used to build an ID3 tree if one is note provided by the user
            id3tree - represents the tree built using ID3 algorithm at the same training data set
            
Methods:    read_id3_tree - loads a tree into field id3tree from file named file_name, returns 1 if success and -1 if the loaded 
                            is not a tree
            get_subtree_training_error(core_node) - counts the error with classification of a subtree under core_node 
            
            estimate_subtree_test_error (core_node) - estimates the classification error in a subtree under core_node on test data set
            
            get_subtree_objects (node) - returns the data set used to build the subtree under node
            
            most_frequent_class_in_subtree(core_node) - returns the most frequent data class in subtree under core_node
            
            prune tree - prunes the tree built by ID3, returns a Tree object
"""


class C45:

    def __init__(self, data, class_name, attributes):
        self.objects = data
        self.class_name = class_name
        self.attributes = attributes
        self.classes = np.sort(self.objects[class_name].unique())
        self.id3 = ID3.ID3(data, class_name, attributes)
        self.id3tree = ID3.Tree(None)

    def set_id3_tree(self, tree):
        self.id3tree = tree

    def get_subtree_training_error(self, core_node):
        subtree = ID3.Tree(core_node)
        error = 0
        for index in self.objects.index:
            o = self.objects.loc[index]
            class_val = subtree.classify(o)
            if class_val != o[self.class_name]:
                error = error + 1
        return error / len(self.objects)

    def estimate_subtree_test_error(self, core_node):
        training_error = self.get_subtree_training_error(core_node)
        return training_error + np.sqrt(training_error * (1 - training_error)) / len(self.objects)

    def get_subtree_objects(self, node):
        current_node = node
        subtree_objects = self.objects
        while current_node.parent is not None:
            subtree_objects = self.id3.attribute_value_members(
                current_node.parent.attribute,
                current_node.parent.attribute_values[current_node.parent.children.index(current_node)],
                subtree_objects
            )
            current_node = current_node.parent
        return subtree_objects

    def most_frequent_class_in_subtree(self, core_node):
        subtree_objects = self.get_subtree_objects(core_node)
        dom_class = self.id3.get_most_frequent_class(subtree_objects)
        return dom_class

    def prune_tree(self):
        if self.id3tree.core_node is None:
            self.id3tree = self.id3.build_tree(None, self.attributes, self.objects)
        leaves = self.id3tree.leaves()
        for leaf in leaves:
            current_node = leaf.parent
            while current_node.parent is not None:
                e_0 = self.estimate_subtree_test_error(current_node)
                alternative_leaf = ID3.Leaf(current_node.parent, self.most_frequent_class_in_subtree(current_node))
                e_1 = self.estimate_subtree_test_error(leaf)
                if e_0 > e_1:
                    index = current_node.parent.children.index(current_node)
                    current_node.parent.children[index] = alternative_leaf
                    for l in current_node.leaves:
                        leaves.remove(l)
                current_node = current_node.parent
        return ID3.Tree(current_node)
