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
import os


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



attributes = [['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu','Mjob'],
                ['sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu','Mjob', 'Fjob'],
                ['age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu','Mjob', 'Fjob', 'reason'],
                ['address', 'famsize', 'Pstatus', 'Medu', 'Fedu','Mjob', 'Fjob', 'reason', 'guardian'],
                ['famsize', 'Pstatus', 'Medu', 'Fedu','Mjob', 'Fjob', 'reason', 'guardian', 'traveltime'],
                ['Pstatus', 'Medu', 'Fedu','Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime'],
                ['Medu', 'Fedu','Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime','failures'],
                ['Fedu','Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime','failures', 'schoolsup'],
                ['Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime','failures', 'schoolsup', 'famsup'],
                ['Fjob', 'reason', 'guardian', 'traveltime', 'studytime','failures', 'schoolsup', 'famsup', 'paid'],
                ['reason', 'guardian', 'traveltime', 'studytime','failures', 'schoolsup', 'famsup', 'paid', 'activities'],
                ['guardian', 'traveltime', 'studytime','failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery'],
                ['traveltime', 'studytime','failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery','higher'],
                ['studytime','failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery','higher', 'internet'],
                ['failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery','higher', 'internet', 'romantic'],
                ['schoolsup', 'famsup', 'paid', 'activities', 'nursery','higher', 'internet', 'romantic', 'famrel'],
                ['famsup', 'paid', 'activities', 'nursery','higher', 'internet', 'romantic', 'famrel', 'freetime'],
                ['paid', 'activities', 'nursery','higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout'],
                ['activities', 'nursery','higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'health'],
                ['nursery','higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'health', 'absences'],
                ['higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'health', 'absences', 'G1'],
                ['internet', 'romantic', 'famrel', 'freetime', 'goout', 'health', 'absences', 'G1','G2'],
                ['romantic', 'famrel', 'freetime', 'goout', 'health', 'absences', 'G1','G2', 'G3']]



'''
object = data.objects.loc[14]
print('True value: 'object['Walc'])

print(id3.class_count(2,data.objects))
#print(len(id3.attribute_value_members('absences',10,data.objects)))
tree = id3.build_tree(None, attributes[it], data.objects)

print(tree)
print(tree.classify(object))
print(object['Walc'])
object = data.objects.loc[104]
print(tree.classify(object))
print(object['Walc'])
object = data.objects.loc[600]
print(tree.classify(object))
print(object)


c45 = C45.C45(sample,'Walc', attributes[it])
c45.set_id3_tree(tree)
c45tree = c45.prune_tree()
print(c45tree.classify(object))
'''

file_path1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "\\student-alcohol-consumption\\student-mat.csv"
file_path2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "\\student-alcohol-consumption\\student-por.csv"

data = Data.StudentsData()
data.load_data(file_path1,file_path2)
sample_learn = data.objects.query("index % 3 == 0") # test batch 
sample_test = data.objects.query("index % 3 != 0") # test batch


it = 0
build_trees = False
if build_trees is True:
    for it in range(22):
        # Get absolute path


        id3 = ID3.ID3(sample_learn,'Walc', attributes[it])
        tree = id3.build_tree(None, attributes[it], sample_learn)
        tree.save_to_file('AlkoPSZT/nodes' + str(it), 'AlkoPSZT/tree' + str(it))



make_test = True
if make_test is True:
    
    for it in range(22):
        good_id3 = 0
        good_c45 = 0
        delta_id3 = 0
        delta_c45 = 0
        print('set: ' + str(it))

        file = open(os.path.dirname(os.path.abspath(__file__)) + '\\nodes' + str(it),'rb')
        nodes = pickle.load(file)
        file.close()
        file2 = open(os.path.dirname(os.path.abspath(__file__)) + '\\tree' + str(it),'rb')
        tree = pickle.load(file2)
        file2.close()
        c45 = C45.C45(sample_test,'Walc', attributes[it])
        c45.set_id3_tree(tree)
        c45tree = c45.prune_tree()

        for i in range(660):
            id3_temp = tree.classify(data.objects.loc[i]) - data.objects.loc[i,'Walc']
            c45_temp = c45tree.classify(data.objects.loc[i]) - data.objects.loc[i,'Walc']
            delta_id3 = delta_id3 + abs(id3_temp)
            delta_c45 = delta_c45 + abs(c45_temp)

            if id3_temp == 0:
                good_id3 = good_id3 + 1
            if  c45_temp == 0:
                good_c45 = good_c45 + 1

        print('correctly classified id3: ' + str(good_id3))
        print('correctly classified c45: ' + str(good_c45))
        print('delta id3: ' + str(delta_id3))
        print('delta c45: ' + str(delta_c45))


