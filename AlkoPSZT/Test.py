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

build_trees = True # if true algorythm id3 will be invoked, it's time consuming, so after building trees it's recomended to set build_trees to False
make_test = True # if true, tests will be executed. Results of tests are in console 


# list of atribute groups, on which tests will be executed
attributes = [['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu','Mjob'], #1
                ['sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu','Mjob', 'Fjob'], #2
                ['age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu','Mjob', 'Fjob', 'reason'], #3
                ['address', 'famsize', 'Pstatus', 'Medu', 'Fedu','Mjob', 'Fjob', 'reason', 'guardian'], #4
                ['famsize', 'Pstatus', 'Medu', 'Fedu','Mjob', 'Fjob', 'reason', 'guardian', 'traveltime'], #5
                ['Pstatus', 'Medu', 'Fedu','Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime'], #6
                ['Medu', 'Fedu','Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime','failures'], #7
                ['Fedu','Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime','failures', 'schoolsup'], #8
                ['Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime','failures', 'schoolsup', 'famsup'], #9
                ['Fjob', 'reason', 'guardian', 'traveltime', 'studytime','failures', 'schoolsup', 'famsup', 'paid'], #10
                ['reason', 'guardian', 'traveltime', 'studytime','failures', 'schoolsup', 'famsup', 'paid', 'activities'], #11
                ['guardian', 'traveltime', 'studytime','failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery'], #12
                ['traveltime', 'studytime','failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery','higher'], #13
                ['studytime','failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery','higher', 'internet'], #14
                ['failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery','higher', 'internet', 'romantic'], #15
                ['schoolsup', 'famsup', 'paid', 'activities', 'nursery','higher', 'internet', 'romantic', 'famrel'], #16
                ['famsup', 'paid', 'activities', 'nursery','higher', 'internet', 'romantic', 'famrel', 'freetime'], #17
                ['paid', 'activities', 'nursery','higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout'], #18
                ['activities', 'nursery','higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'health'], #19
                ['nursery','higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'health', 'absences'], #20
                ['higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'health', 'absences', 'G1'], #21
                ['internet', 'romantic', 'famrel', 'freetime', 'goout', 'health', 'absences', 'G1','G2'], #22
                ['romantic', 'famrel', 'freetime', 'goout', 'health', 'absences', 'G1','G2', 'G3']] #23



file_path1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "\\student-alcohol-consumption\\student-mat.csv"
file_path2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "\\student-alcohol-consumption\\student-por.csv"

data = Data.StudentsData()
data.load_data(file_path1,file_path2)
sample_learn = data.objects.query("index % 3 == 0") # learning batch 
sample_test = data.objects.query("index % 3 != 0") # test batch


it = 0

if build_trees is True:
    for it in range(23):
        # Get absolute path


        id3 = ID3.ID3(sample_learn,'Walc', attributes[it])
        tree = id3.build_tree(None, attributes[it], sample_learn)
        tree.save_to_file('AlkoPSZT/nodes' + str(it), 'AlkoPSZT/tree' + str(it))


if make_test is True:
    lenght_t, _ = sample_test.shape
    lenght_l, _ = sample_learn.shape
    print(str(lenght_t))
    print(str(lenght_l))
    for it in range(23):
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
        c45 = C45.C45(sample_learn,'Walc', attributes[it])
        c45.set_id3_tree(pickle.loads(pickle.dumps((tree))))
        c45tree = c45.prune_tree()

        for index, row in sample_test.iterrows():
            id3_temp = tree.classify(row) - row['Walc']
            c45_temp = c45tree.classify(row) - row['Walc']
            delta_id3 = delta_id3 + id3_temp**2
            delta_c45 = delta_c45 + c45_temp**2

            if id3_temp == 0:
                good_id3 = good_id3 + 1
            if  c45_temp == 0:
                good_c45 = good_c45 + 1

        print('correctly classified id3: ' + str(good_id3))
        print('correctly classified c45: ' + str(good_c45))
        print('delta id3: ' + str(delta_id3/lenght_t))
        print('delta c45: ' + str(delta_c45/lenght_t))
        print()


