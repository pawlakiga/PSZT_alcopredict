
"""
Pawlak Iga, Michalski Marcin
Algorytm ID3
"""
import Data
import numpy as np
class ID3 :

    def __init__(self, data, objects_number):
        self.data = data
        self.objects_number = objects_number
        self.classes = data.classes

    def enthropy(self, indexes):
        ent = 0
        class_count = [0] * len(self.classes)
        for c in range(len(self.classes)) :
            class_count[c] = len(get_class_members(c, indexes))
            ent = ent - class_count[c] * np.log(class_count[c])
        return ent



    def get_class_members(self,class_index,indexes):
        members = [0]
        for ix in indexes :
            if data.get_object_class(ix) == self.classes[class_index] :
               member.append(ix)
        return members

    def get_attribute_value_members(self,attribute,value):
        members = [0]
        for ix in range(self.objects_number) :
            if data.get_object_attribute_value(ix,attribute) == value :
                members.append(ix)
        return members

    def inf_gain(self,attribute):

        inf = 0
        self.enthropy = self.enthropy(self.objects_number)
        for j in data.get_all_attribute_values(attribute) :
            members = self.get_attribute_value_members(attribute,j)
            inf = inf + self.enthropy(members) * len(members)/self.objects_number

        inf_gain = self.enthropy - inf


















