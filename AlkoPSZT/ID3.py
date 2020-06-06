"""
Pawlak Iga, Michalski Marcin
PSZT Projekt 2
Alcohol Consumption Prediction
Algorithm ID3
"""
import Data
import numpy as np
import pickle

"""
Class representing the nodes in the tree 
Fields:     parent : Node - parent node
            attribute : String - name of the attribute checked in that node 
            attribute_values : List  - values of this attribute, sorted ascending 
            children : List - list of child nodes with indexes corresponding to index of value in attribute_values

Methods:    add_child_node( Int value, Node child) : void
            get_child_node( Int value ) : Node
            has_children : Boolean

"""


class Node:
    def __init__(self, parent, attribute, values):
        self.parent = parent
        self.attribute = attribute
        self.attribute_values = list(values)
        self.children = list()
        self.leaves = list()

    def add_child_node(self, value, child):
        index = self.attribute_values.index(value)
        self.children.insert(index, child)
        if isinstance(child, Leaf):
            self.leaves.append(child)
        else:
            for l in child.leaves:
                self.leaves.append(l)

    def get_child_node(self, value):
        if self.attribute_values.__contains__(value) == False:
            return -1
        index = self.attribute_values.index(value)
        return self.children[index]

    def has_children(self):
        if len(self.children) == 0:
            return False
        else:
            return True


"""
Class representing the Leafs in the tree 
Fields:     parent : Node 
            class_val - the class that is assigned to objects in that leaf 

"""


class Leaf:
    def __init__(self, parent, class_val):
        self.parent = parent
        self.class_val = class_val


"""
Class representing the tree used for classification
Fields:     core_node  : Node  - the first node in the tree 
            nodes : List<Nodes>  - list of nodes (not sure if necessary) 

Methods:    add_node (Node node) : void 
            classify (object) : class_value - function that classifies an object, starts at the core node and chooses the next node 
                                              based on object's value of the attribute checked in that node 
                                              -> stops when current_node is a leaf and returns the class_val from that leaf 
            leaves - returns a list of all leaves in the tree 
            add_all_nodes(start_node : Node) - adds all nodes under start_node to field 'nodes' - used when saving to file, probably unnecessary
            save_to_file(filename_nodes, filename_tree) - saves the tree and its nodes to files with given names 
"""


class Tree:
    def __init__(self, core_node):
        self.core_node = core_node
        self.nodes = list()

    def add_node(self, node):
        self.nodes.append(node)

    def leaves(self):
        return self.core_node.leaves

    def classify(self, object):
        current_node = self.core_node

        while not isinstance(current_node, Leaf):
            if current_node.has_children():
                child = current_node.get_child_node(object[current_node.attribute])
                if child == -1:
                    return -1
                current_node = child

        return current_node.class_val

    def add_all_nodes(self, start_node):
        current_node = start_node
        if isinstance(current_node, Leaf):
            return
        for node in current_node.children:
            self.nodes.append(node)
            self.add_all_nodes(node)

    def save_to_file(self, filename_nodes, filename_tree):
        self.add_all_nodes(self.core_node)
        if len(self.nodes) == 0:
            return -1
        nodes_file = open(filename_nodes, 'wb')
        tree_file = open(filename_tree, 'wb')
        pickle.dump(self.nodes, nodes_file)
        pickle.dump(self, tree_file)
        nodes_file.close()
        tree_file.close()


""""
Class implementing the ID3 algorithm 
Fields:     objects - training data set
            objects_number - number of objects (unused) 
            class_name - the name of the column with class values in training data 
            attributes - attributes other than class 
            classes - list of classes

Methods:    all_attribute_values(String column_name, Objects)  - returns all unique values of attribute in column : column_name, found in objects 
            object_attribute_value(Int index, String column_name) - returns the value of attribute in column_name of object at index
            object_class(index) - returns the class of the object at index
            attribute_value_members(String attribute_name, attribute_value, Objects) - returns all objects from Objects where the value of 
                                                                                       attribute == attribute_value
                                                                                       
            class_members(class_val, objects) - same as former but with classes
            class_count(class_val,objects) - the number of class_members in Objects 
            
            count_enthropy(objects) - returns the value of enthropy in Objects
            inf_gain(attribute,objects) - returns the value of information gain for the specified attribute and objects 
            get_D(attributes,objects) - returns the attribute from given attributes with maximum inf gain for objects
            build_tree(start_node,attributes,objects) - builds a classification tree under start_node
"""


class ID3:

    def __init__(self, data, class_name, attributes):
        self.objects = data
        self.objects_number = len(data)
        self.class_name = class_name
        self.attributes = attributes
        self.classes = np.sort(self.objects[class_name].unique())

    def all_attribute_values(self, column_name, objects):
        return np.sort(objects[column_name].unique())

    def object_attribute_value(self, index, column_name):
        return self.objects.loc[index, column_name]

    def object_class(self, index):
        return self.objects.loc[index, self.class_name]

    def attribute_value_members(self, attribute_name, attribute_value, objects):
        return objects.query(attribute_name + "==\"" + str(attribute_value) + "\"")

    def class_members(self, class_val, objects):
        return objects.query(self.class_name + '==\"' + str(class_val) + "\"")

    def class_count(self, class_val, objects):
        return len(self.class_members(class_val, objects))

    def count_enthropy(self, objects):
        ent = 0
        for i in self.classes:
            class_count = len(self.class_members(i, objects))
            class_frequency = class_count / len(objects)
            if class_frequency == 0:
                continue
            ent = ent - class_frequency * np.log(class_frequency)
        return ent

    def inf_gain(self, attribute, objects):
        inf = 0
        for j in self.all_attribute_values(attribute, objects):
            # members - Sj
            members = self.attribute_value_members(attribute, j, objects)
            if len(members) == 0:
                continue
            # I(Sj)
            i = self.count_enthropy(members)
            # Inf(D,S) == sum j (I(Sj)*|Sj|/|S|
            inf = inf + i * len(members) / len(objects)
        # InfGain(D,S) = I(S) - Inf(D,S)
        inf_gain = self.count_enthropy(objects) - inf
        return inf_gain

    def get_D(self, attributes, objects):
        max_inf_gain = 0
        D = 0
        for a in attributes:
            if self.inf_gain(a, objects) > max_inf_gain:
                max_inf_gain = self.inf_gain(a, objects)
                D = a
        # No attribute gives any information - choose the first one
        if D == 0:
            D = attributes[0]
        return D

    def get_most_frequent_class(self, objects):
        max_count = 0
        dom_class = 0
        for c in self.classes:
            if self.class_count(c, objects) > max_count:
                dom_class = c
                max_count = self.class_count(c, objects)
        return dom_class

    def build_tree(self, start_node, attributes, objects):
        av_attributes = attributes.copy()
        # if S is empty return error
        if len(objects) == 0:
            return -1
        # if all objects are from the same class return a leaf with that class
        for c in self.classes:
            if len(self.class_members(c, objects)) == len(objects):
                return Tree(Leaf(start_node, c))
        # if R is empty return the most frequent class in S
        if len(attributes) == 0:
            c = self.get_most_frequent_class(objects)
            return Tree(Leaf(start_node, c))
        D = self.get_D(av_attributes, objects)
        # Remove D from R
        av_attributes.remove(D)
        # Create node with D and its values in objects
        node = Node(start_node, D, self.all_attribute_values(D, objects))
        # Build sub trees under node and add their core nodes as children to the current node
        for j in node.attribute_values:
            # attribute_value_members - Sj
            subtree = self.build_tree(node, av_attributes, self.attribute_value_members(D, j, objects))
            if subtree != -1:
                node.add_child_node(j, subtree.core_node)
        return Tree(node)
