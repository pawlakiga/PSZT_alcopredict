
"""
Pawlak Iga, Michalski Marcin
Algorytm ID3
"""
import Data
import numpy as np

class Node :
    def __init__(self, parent, attribute, values):
        self.parent = parent
        self.attribute = attribute
        self.attribute_values = list(values)
        self.children = list()

    def add_child_node(self,value, child):
        index = self.attribute_values.index(value)
        self.children.insert(index,child)

    def get_child_node(self,value):
        index = self.attribute_values.index(value)
        return self.children[index]

    def has_children(self):
        if len(self.children) == 0 :
            return False
        else :
            return True


class Leaf :
    def __init__(self,parent,class_val):
        self.parent = parent
        self.class_val = class_val

class Tree :
    def __init__(self,core_node):
        self.core_node = core_node

    def add_node(self, node):
        self.nodes.append(node)

    def classify(self,object):
        current_node = self.core_node

        while not isinstance(current_node,Leaf):
            if current_node.has_children() :
                current_node = current_node.get_child_node(object.loc[current_node.attribute])

        return current_node.class_val



class ID3 :

    def __init__(self, data, class_name, attributes):
        self.objects = data
        self.objects_number = len(data)
        self.class_name = class_name
        self.attributes = attributes
        self.set_classes(class_name)


    def set_classes(self, column_name):
        self.classes = np.sort(self.objects[column_name].unique())
        self.class_name = column_name
        return 1

    def all_attribute_values(self, column_name):
        return self.objects[column_name].unique()

    def object_attribute_value(self, index, column_name):
        return self.objects.loc[index, column_name]

    def object_class(self, index):
        return self.objects.loc[index, self.class_name]

    def attribute_value_members(self, attribute_name, attribute_value, objects):
        return objects.query(attribute_name + "==\"" + str(attribute_value) + "\"")

    def class_members(self, class_val, objects):
        return objects.query(self.class_name + '==\"' + str(class_val) + "\"")

    def class_count(self,class_val,objects):
        return len(self.class_members(class_val, objects))

    def count_enthropy(self, objects):
        ent = 0
        for i in self.classes:
            class_count = len(self.class_members(i, objects))
            class_frequency = class_count/len(objects)
            if class_frequency == 0 :
                continue
            ent = ent - class_frequency * np.log(class_frequency)
        return ent

    def inf_gain(self,attribute,objects):
        inf = 0
        for j in self.all_attribute_values(attribute) :
            members = self.attribute_value_members(attribute,j,objects)
            if len(members) == 0 :
                continue
            i = self.count_enthropy(members)
            inf = inf + i*len(members)/len(objects)
        inf_gain = self.count_enthropy(objects) - inf
        return inf_gain

    def get_D(self,attributes,objects):
        max_inf_gain = 0
        D = 0
        for a in attributes :
            if self.inf_gain(a,objects) > max_inf_gain :
                max_inf_gain = self.inf_gain(a,objects)
                D = a
            if (D==0) :
                print(objects)
        return D


    def build_tree(self, start_node,attributes,objects):
        if len(objects) == 0 :
            return -1
        for c in self.classes :
            if len(self.class_members(c,objects)) == len(objects) :
               return Tree(Leaf(start_node,c))
        if len(self.attributes) == 0 :
            max_count = 0
            dom_class = 0
            for c in self.classes:
                if self.class_count(c,objects) > max_count :
                    dom_class = c
                    max_count = self.class_count(c,objects)
            return Tree(Leaf(start_node,c))
        D = self.get_D(attributes,objects)
        if D == 0 :
            D = attributes[0]
        attributes.remove(D)
        print(attributes)
        node = Node(start_node,D,self.all_attribute_values(D))
        self.current_node = node
        for j in node.attribute_values:
            #print(self.attribute_value_members(D,j,objects))
            subtree = self.build_tree(node, attributes, self.attribute_value_members(D,j,objects))
            if subtree != -1 :
                node.add_child_node(j,subtree.core_node)
        return Tree(node)






















