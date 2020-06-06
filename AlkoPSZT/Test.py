""""
Pawlak Iga, Michalski Marcin
PSZT Projekt 2
Alcohol Consumption Prediction
Test
"""
import Data
import C45
import pickle
import ID3
import random


class Test :

    def __init__(self):
        self.file_path_mat = ''
        self.file_path_port = ''
        self.id3tree = None
        self.data = None
        self.training_data = None
        self.test_data = None

    def set_paths(self, path_mat, path_port):
        self.file_path_mat = path_mat
        self.file_path_port = path_port

    def load_data(self):
        self.data = Data.StudentsData()
        self.data.load_data(self.file_path_mat,self.file_path_port)

    def read_id3_tree(self, file_name):
        file = open(file_name, 'rb')
        tree = pickle.load(file)
        if isinstance(tree, ID3.Tree):
            self.id3tree = tree
            return 1
        else:
            return -1

    def set_training_test_subsets(self, bias):
        training_indexes = random.sample(self.data.index,int(len(self.data.objects)*bias))
        self.training_data = self.data.objects.iloc[training_indexes]
        self.test_data = self.data.objects.drop(training_indexes)












file_path1 = 'C:\\Users\\Iga\\Documents\\GitHub\\PSZT_alcopredict\\student-alcohol-consumption\\student-mat.csv'
file_path2 = 'C:\\Users\\Iga\\Documents\\GitHub\\PSZT_alcopredict\\student-alcohol-consumption\\student-por.csv'

data = Data.StudentsData()
data.load_data(file_path1,file_path2)
sample = data.objects.query("index % 30 == 7")
file = open('nodes','rb')
nodes = pickle.load(file)
file.close()
file2 = open('tree','rb')
tree = pickle.load(file2)
file2.close()

print(tree.classify(data.objects.loc[67]))
print('Faktyczna klasa: ' + str(data.objects.loc[67,'Walc']))
c45 = C45.C45(sample,'Walc', data.attributes)
c45.set_id3_tree(tree)
c45tree = c45.prune_tree()
print(c45tree.classify(data.objects.loc[67]))
c45tree.save_to_file('c45nodes_walc','c45tree_dalc')

"""

id3 = ID3.ID3(sample,'Walc', data.attributes)
tree = id3.build_tree(None, data.attributes, sample)
tree.save_to_file('nodes', 'tree')

object = data.objects.loc[14]
print(object['Walc'])

print(id3.class_count(2,data.objects))
#print(len(id3.attribute_value_members('absences',10,data.objects)))
tree = id3.build_tree(None, data.attributes, data.objects)
print(tree)
print(tree.classify(object))
print(object['Walc'])
object = data.objects.loc[104]
print(tree.classify(object))
print(object['Walc'])
object = data.objects.loc[600]
print(tree.classify(object))
print(object)
c45 = C45.C45(data.objects,'Walc', data.attributes)
c45.set_id3_tree(tree)
c45tree = c45.prune_tree()
print(c45tree.classify(object))"""

